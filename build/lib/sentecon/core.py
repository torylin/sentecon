import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, models
from torch import cuda
from sklearn.cluster import KMeans
from empath import Empath
import pdb
from tqdm import tqdm
import liwc
import os
import torch
import warnings
warnings.filterwarnings("ignore")

tqdm.pandas()

class SenteCon:
    def __init__(self, lexicon, lm, lm_library, data_dir, liwc_path, num_centroids=1, seed=230706):
        self.device = 'cuda' if cuda.is_available() else 'cpu'
        
        self.data_dir = data_dir
        self.lexicon = lexicon
        self.lm = lm
        self.lm_library = lm_library
        self.num_centroids = num_centroids

        if self.lexicon == 'Empath':
            lexicon = Empath()
            self.category_names = list(lexicon.cats.keys())
        elif self.lexicon == 'LIWC':
            parse, category_names = liwc.load_token_parser(liwc_path)
            self.category_names = category_names[21:]

        if self.num_centroids > 1:
            self.category_headers = []
            for cluster in range(self.num_centroids):
                self.category_headers += ['{}_c{}'.format(category, cluster) for category in category_names]
        else:
            self.category_headers = self.category_names
        
        # corpus_df = pd.read_pickle(os.path.join(self.data_dir, '{}Vocab_corpus_{}_{}.pkl'.format(
        #     self.lexicon, self.lexicon, self.lm)))
        # corpus_df.dropna(inplace=True)

        # self.centroids = self.__compute_centroids(corpus_df, self.num_centroids, seed)
        # self.centroids.to_csv(os.path.join(self.data_dir,'{}Vocab_centroids_{}_{}.csv'.format(self.lexicon, self.lexicon, self.lm)), index=False)
        # pdb.set_trace()
        self.centroids = pd.read_csv(os.path.join(self.data_dir,'{}Vocab_centroids_{}_{}.csv'.format(self.lexicon, self.lexicon, self.lm)))
        self.centroids = self.centroids.reindex(sorted(self.centroids.columns), axis=1)
        self.centroid_names = sorted(list(set([col.split('_')[0] for col in self.centroids.columns])))
        
    def __compute_centroids(self, corpus_df, num_centroids, seed):
        centroids = dict()
        categories = set(corpus_df['category'].values)
        
        for category in categories:
            cat_embeds = np.vstack(corpus_df[corpus_df['category'] == category]['sentence_embedding'].values)
            if self.num_centroids == 1:
                centroids[category] = cat_embeds.mean(axis=0)
            else:
                cluster_model = KMeans(n_clusters=self.num_centroids, random_state=seed)
                try:
                    cluster_model.fit(cat_embeds)
                except:
                    print('Empty centroids')
                    pdb.set_trace()
                clusters = cluster_model.predict(cat_embeds)
                for cluster in set(clusters):
                    centroids['{}_c{}'.format(category, cluster)] = cat_embeds[clusters == cluster].mean(axis=0)  

            centroids = pd.DataFrame.from_dict(centroids)

        return centroids
    
    def __compute_representation(self, row, embeddings, centroids, num_centroids, rel_cats, rel_cats_header, centroid_names, device):
        embedding = embeddings[row.name]
        similarities = dict.fromkeys(rel_cats, 0)

        rel_centroids = centroids[sorted(list(set(rel_cats_header) & set(centroids.columns)))]
        centroids_array = rel_centroids.to_numpy()
        newsim = torch.nn.CosineSimilarity(dim=1)

        new_similarities = newsim(torch.tensor(embedding).to(device), torch.tensor(centroids_array.T).to(device))
        new_similarities = new_similarities.view(-1, num_centroids).max(dim=1).values
        
        cols = sorted(list(set(rel_cats) & set(centroid_names)))
        new_similarities = dict(zip(cols, new_similarities.tolist()))
        similarities.update(new_similarities)

        return similarities
    

    def embed(self, text):
        if self.lm_library == 'sentence-transformers':
            model = SentenceTransformer(self.lm)
        elif self.lm_library == 'transformers':
            word_model = models.Transformer(self.lm)
            pooling = models.Pooling(word_model.get_word_embedding_dimension())
            model = SentenceTransformer(modules=[word_model, pooling])
    
        model.to(self.device)
        sentence_embeds = model.encode(text, show_progress_bar=True)
        text_df = pd.DataFrame({'text': text})

        text_df['representation'] = text_df.progress_apply(
            self.__compute_representation,
            args=(sentence_embeds, self.centroids, self.num_centroids, self.category_names, self.category_headers, self.centroid_names, self.device),
            axis=1
        )

        rep_df = pd.DataFrame(text_df['representation'].values.tolist())

        return rep_df
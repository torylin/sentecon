from context import SenteCon

sentecon = SenteCon(lexicon='LIWC', lm='all-mpnet-base-v2', lm_library='sentence-transformers', data_dir='/home/victorialin/Documents/2021-2022/Research/sentecon/sentecon/data/',
                    liwc_path='/home/victorialin/Documents/liwc_dict/LIWC2015_English_Flat.dic')
df = sentecon.embed(['this is a test', 'what do you mean'])

print(df)
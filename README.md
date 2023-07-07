# SenteCon

SenteCon is a method for introducing human interpretability in deep language representations using lexicons. Given a passage of text, SenteCon encodes the text as a layer of interpretable categories over an existing deep language model, offering interpretability at little to no cost to downstream performance. For more information, please see our paper, [SenteCon: Leveraging Lexicons to Learn Human-Interpretable Language Representations](https://arxiv.org/pdf/2305.14728.pdf) (to appear in the Findings of ACL 2023).

SenteCon can be installed via `pip` from [PyPi](https://pypi.org/project/sentecon/):
```
pip install sentecon
```

To use SenteCon, import as follows:
```
from sentecon import SenteCon
sentecon = SenteCon(lexicon='LIWC', lm='all-mpnet-base-v2', lm_library='sentence-transformers', data_dir=$DATA_DIR, liwc_path=$LIWC_PATH)
```

Pre-built `SenteCon` class' argument options for `lexicon` are `['LIWC', 'Empath']`. For LIWC, pre-built options for `lm` are `['all-mpnet-base-v2', 'all-MiniLM-L6-v2', 'all-distilroberta-v1']` (corresponding to `lm_library=sentence-transformers`) and `['bert-base-uncased', 'roberta-base']` (corresponding to `lm_library=transformers`). For Empath, pre-built options for `lm` are `['all-mpnet-base-v2', 'all-MiniLM-L6-v2']` (corresponding to `lm_library=sentence-transformers`). Details on how to use custom models will be added soon.

To generate SenteCon representations:
```
sentecon.embed(['this is a test', 'what do you mean'])
#      affect    posemo    negemo       anx     anger       sad    social    family    friend    female      male   cogproc  ...      work   leisure      home     money     relig     death  informal     swear  netspeak    assent    nonflu    filler
# 0  0.257607  0.241576  0.254157  0.254629  0.239016  0.217544  0.270090  0.198617  0.185521  0.209694  0.198943  0.321512  ...  0.360601  0.289882  0.240001  0.255045  0.212570  0.243143  0.249118  0.240042  0.244287  0.156011  0.232672  0.117097
# 1  0.298780  0.268381  0.305567  0.259470  0.291800  0.267315  0.269026  0.150925  0.249551  0.171428  0.190681  0.347195  ...  0.245808  0.229228  0.181226  0.210201  0.186952  0.246317  0.304549  0.286534  0.294494  0.239390  0.256054  0.236797
```

Please note that the LIWC lexicon is proprietary, so it is not included in this repository. Users must have access to a LIWC `.dic` file, which can be purchased from [liwc.app](https://www.liwc.app/).

To run SenteCon on the evaluation datasets from the paper, first clone this repository. Then use the command

```
.experiments/bash/run_sentecon.sh $SCRIPT_DIRECTORY
```

Human annotations of LIWC categories for the MELD dataset can be found under `sentecon/experiments/data/MELD/annotation_scripts/`. These annotations are indexed by `S1` through `S5`, which correspond to sentence batches 1-5 (also under the same directory), and `C1` through `C5`, which correspond to category batches (listed in the paper appendix, Section B.3).


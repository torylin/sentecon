# SenteCon

SenteCon is a method for introducing human interpretability in deep language representations using lexicons. Given a passage of text, SenteCon encodes the text as a layer of interpretable categories over an existing deep language model, offering interpretability at little to no cost to downstream performance. For more information, please see our paper, [SenteCon: Leveraging Lexicons to Learn Human-Interpretable Language Representations](https://aclanthology.org/2023.findings-acl.264.pdf) (Findings of ACL 2023).

## Setup

SenteCon can be installed via `pip` from [PyPI](https://pypi.org/project/sentecon/):
```
pip install sentecon
```
## Usage

To use SenteCon, import the `SenteCon` class, which takes the arguments `lexicon` and `lm`.
```
from sentecon import SenteCon
```

Pre-built options for `lexicon` are `['LIWC', 'Empath']`. 

Pre-built options for `lm` (all pre-trained models) are:
- LIWC: `['all-mpnet-base-v2', 'all-MiniLM-L6-v2', 'all-distilroberta-v1', 'bert-base-uncased', 'roberta-base']`
- Empath: `['all-mpnet-base-v2', 'all-MiniLM-L6-v2']`

The following code produces SenteCon representations (returned as a `pandas` dataframe) that use [Empath](https://github.com/Ejhfast/empath-client/) as the base lexicon $L$ and pre-trained [MPNet](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) as the embedding language model $M_\theta$:

```
sentecon = SenteCon(lexicon='Empath', lm='all-mpnet-base-v2')
sentecon.embed(['this is a test', 'what do you mean'])

       help    office     dance     money   wedding  domestic_work     sleep  ...     ocean    giving  contentment   writing     rural  positive_emotion   musical
0  0.284190  0.320671  0.267699  0.277306  0.273392       0.311223  0.305355  ...  0.277074  0.270200     0.265807  0.356591  0.266273          0.278889  0.283758
1  0.244075  0.237357  0.220706  0.197963  0.222953       0.217883  0.219400  ...  0.180234  0.222138     0.246295  0.275586  0.183908          0.263977  0.220248
```

Please note that the LIWC lexicon is proprietary, so it is not included in this repository. To use the LIWC option, users must have access to a LIWC `.dic` file, which can be purchased from [liwc.app](https://www.liwc.app/). The path to this `.dic` file must be specified in the `liwc_path` argument when calling the `SenteCon` class, e.g.,

```
sentecon = SenteCon(lexicon='LIWC', lm='all-mpnet-base-v2', liwc_path=$LIWC_PATH)
```

**When using SenteCon representations for predictive tasks, it is often helpful to standardize over columns (and sometimes also helpful to standardize over rows).**

Some features that will be added soon:
- The ability to use custom (e.g., fine-tuned) models for `lm`
- Support for SenteCon+

## Rerunning experiments

To run SenteCon and SenteCon+ on the evaluation datasets from the paper, first clone this repository. Then use the command

```
./experiments/bash/run_sentecon.sh $SCRIPT_DIRECTORY
```

Human annotations of LIWC categories for the MELD dataset can be found under `experiments/data/MELD/annotation_scripts/`. These annotations are indexed by `S1` through `S5`, which correspond to sentence batches 1-5 (also under the same directory), and `C1` through `C5`, which correspond to category batches (listed in the paper appendix, Section B.3).


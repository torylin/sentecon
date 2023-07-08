# SenteCon

SenteCon is a method for introducing human interpretability in deep language representations using lexicons. Given a passage of text, SenteCon encodes the text as a layer of interpretable categories over an existing deep language model, offering interpretability at little to no cost to downstream performance. For more information, please see our paper, [SenteCon: Leveraging Lexicons to Learn Human-Interpretable Language Representations](https://arxiv.org/pdf/2305.14728.pdf) (to appear in the Findings of ACL 2023).

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

Pre-built options for `lm` are:
- LIWC: `['all-mpnet-base-v2', 'all-MiniLM-L6-v2', 'all-distilroberta-v1', 'bert-base-uncased', 'roberta-base']`
- Empath: `['all-mpnet-base-v2', 'all-MiniLM-L6-v2']`

The following code produces SenteCon representations (returned as a `pandas` dataframe) that use [Empath](https://github.com/Ejhfast/empath-client/) as the base lexicon $L$ and [MPNet](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) as the embedding language model $M_\theta$:

```
sentecon = SenteCon(lexicon='Empath', lm='all-mpnet-base-v2')
sentecon.embed(['this is a test', 'what do you mean'])

       help    office     dance     money   wedding  domestic_work     sleep  medical_emergency      cold      hate  ...    weapon  children   monster     ocean    giving  contentment   writing     rural  positive_emotion   musical
0  0.365208  0.304747  0.318054  0.335406  0.274722              0  0.266293                  0  0.269468  0.298770  ...  0.278442  0.265399  0.305324  0.287262  0.283115     0.345869  0.297544  0.278183                 0  0.282828
1  0.235688  0.261899  0.200191  0.243028  0.235971              0  0.183897                  0  0.215329  0.244065  ...  0.287771  0.229656  0.247356  0.285539  0.245884     0.258005  0.279332  0.269617                 0  0.249533
```

Please note that the LIWC lexicon is proprietary, so it is not included in this repository. To use the LIWC option, users must have access to a LIWC `.dic` file, which can be purchased from [liwc.app](https://www.liwc.app/). The path to this `.dic` file must be specified in the `liwc_path` argument when calling the `SenteCon` class, e.g.,

```
sentecon = SenteCon(lexicon='LIWC', lm='all-mpnet-base-v2', liwc_path=$LIWC_PATH)
```

**When using SenteCon representations for predictive tasks, it is often helpful to standardize over columns (and sometimes also helpful to standardize over rows).**

Some features that will be added soon:
- The ability to use custom models for `lm`
- Support for SenteCon+

## Rerunning experiments

To run SenteCon on the evaluation datasets from the paper, first clone this repository. Then use the command

```
.experiments/bash/run_sentecon.sh $SCRIPT_DIRECTORY
```

Human annotations of LIWC categories for the MELD dataset can be found under `sentecon/experiments/data/MELD/annotation_scripts/`. These annotations are indexed by `S1` through `S5`, which correspond to sentence batches 1-5 (also under the same directory), and `C1` through `C5`, which correspond to category batches (listed in the paper appendix, Section B.3).


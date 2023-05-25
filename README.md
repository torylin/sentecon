# SenteCon

SenteCon is a method for introducing human interpretability in deep language representations using lexicons. Given a passage of text, SenteCon encodes the text as a layer of interpretable categories over an existing deep language model, offering interpretability at little to no cost to downstream performance. For more information, please see our paper, [SenteCon: Leveraging Lexicons to Learn Human-Interpretable Language Representations](https://arxiv.org/pdf/2305.14728.pdf) (to appear in the Findings of ACL 2023).

To run SenteCon on the evaluation datasets from the paper, first clone this repository. Then use the command

```
./bash/run_sentecon.sh $SCRIPT_DIRECTORY
```

Human annotations of LIWC categories for the MELD dataset can be found under `data/MELD/annotation_scripts/`. These annotations are indexed by `S1` through `S5`, which correspond to sentence batches 1-5 (also under the same directory), and `C1` through `C5`, which correspond to category batches (listed in the paper appendix, Section B.3).

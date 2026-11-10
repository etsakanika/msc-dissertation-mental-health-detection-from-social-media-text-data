# Detection of Mental Health Disorders from Social Media Text Data

**MSc Dissertation** · Evangelia Tsakanika · MSc Artificial Intelligence and Data Science <br>
CITY College, University of York Europe Campus · 2026

## Overview

Machine learning models trained on Reddit posts report near-perfect accuracy in detecting signs of mental health conditions. Their labels, however, usually come from the community (subreddit) in which a post was written, not from what the post expresses. This dissertation examines what such models actually learn. It compares five classical models and a fine-tuned RoBERTa transformer on two public Reddit corpora whose labels were produced in different ways, by community membership and by human annotation — and evaluates them in-domain, on unseen communities and across the two corpora.

The central finding is an asymmetry: models trained on community labels generalise well to new communities but far less well to human-annotated text, where their errors follow the topic of each community, whereas a model trained on fewer than 3,000 human-annotated excerpts transfers to the community-labelled data almost without loss.

## Research questions

- **RQ1.** How accurately can classical and transformer models detect signs of mental health conditions in Reddit posts?
- **RQ2.** How much of this accuracy comes from stylistic features rather than from the content of the posts?
- **RQ3.** How well do the models generalise to unseen communities and to data labelled in a different way?
- **RQ4.** Where do the models fail, and which conditions do they confuse?

## Repository contents

| Path | Description |
| --- | --- |
| `Detection_of_Mental_Health_Disorders_from_Social_Media_Text_Data_Evangelia_Tsakanika_Master_Dissertation_2026.ipynb` | The complete experimental pipeline, fully executed, with all tables and figures |
| `figures/` | All figures of the dissertation (PNG, 300 dpi) |
| `results/tables/` | All result tables (CSV) |
| `requirements.txt` | Library versions used in the final run |

## Data

The datasets are **not included** in this repository, in line with their terms of release and to protect the privacy of the authors of the posts. Both are publicly available for research:

- **Hasan and Saquer benchmark suite** (four tasks; community labels) - Zenodo, DOI [10.5281/zenodo.17114739](https://doi.org/10.5281/zenodo.17114739)
- **Dreaddit** (stress; human annotation) - E. Turcan and K. McKeown, LOUHI workshop at EMNLP 2019, available from the [authors' page](http://www.cs.columbia.edu/~eturcan/data/dreaddit.zip)

The notebook expects the released CSV files, unchanged, in:

```
<project root>/data/hasan_saquer/
<project root>/data/dreaddit/
```

## How to run

1. Open the notebook in Google Colab and select a **T4 GPU** runtime.
2. Place the data as shown above, and set `PROJECT_ROOT` in section 0.1 (or the environment variable `DISS_ROOT`) to the project root.
3. Run all cells. Tables are written to `results/tables/`, figures to `figures/`, and library versions to `results/environment.json`.

**Caching.** Every trained model's predictions are stored in `results/cache/`, together with a fingerprint of the exact data and settings that produced them. A result is reused only when its fingerprint matches the current configuration, so later runs take minutes instead of hours. To recompute everything from scratch, set `FORCE_RERUN = True` in section 0.1. A full run takes about 15 minutes for the classical models on CPU and about 100 minutes for the five RoBERTa fine-tuning runs on a T4 GPU.

**Reproducibility.** All random processes use seed 42, and the published data splits are used unchanged. GPU computation is not fully deterministic, so re-training RoBERTa may change scores slightly (by about 0.005 macro-F1 in our checks).

## Key results

| Evaluation | Model | Macro-F1 |
| --- | --- | --- |
| In-domain, across the five tasks | RoBERTa | 0.816 – 0.995 |
| In-domain, *general* task, no words of content | Surface (8 stylistic features) | 0.849 |
| *general* → unseen communities | RoBERTa | 0.961 |
| *general* → Dreaddit (human labels) | RoBERTa | 0.631 |
| Dreaddit → *general* (test / external) | RoBERTa | 0.821 / 0.880 |

## Environment

Python 3.13 · PyTorch 2.11 · Hugging Face Transformers 5.21 · scikit-learn 1.6.1 · pandas 2.2.3 · NumPy 2.1.3 · Matplotlib 3.10 · NVIDIA T4 GPU (Google Colab)

## Ethics

The study uses only data publicly released for research. No attempt was made to identify, contact or profile any author, and no user information beyond the released text was used. Notebook outputs that display the text of posts have been cleared, and no posts are quoted in the dissertation. The models are research artefacts for studying detection and generalisation; they have not been validated for clinical or monitoring use.


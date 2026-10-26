# Do Market and Political Risks Narrow the Diversity of Voices? A Comparative Analysis of Actor Representation in the News Prior to Elections

This repository contains the code and aggregated data for a comparative content analysis of **actor diversity** in election news across five countries: Belgium (Flanders), Ireland, the Netherlands, Turkey and the United Kingdom.

## Abstract

Citizens’ access to diverse voices matters most in the run-up to elections, when it supports well-informed voting. Yet we know little about the relationship between system-level conditions and diversity of actors represented in news. The few comparative studies available select cases on the media systems typology rather than on indicators of pluralism itself. We address this gap by assessing actor diversity across five countries that vary on two Media Pluralism Monitor dimensions, market plurality and political independence: Belgium, Ireland, the Netherlands, Turkey and the UK. Using open-weight large language models, we extract 149,752 news actors from 40,656 articles in the year preceding each country’s general election (2023-2024) and assess societal actor diversity across actor functions and political actor diversity in terms of attention to governing and opposition parties. We find that none of the five systems meets the expectations of participatory or deliberative models of democracy. Governing parties supply more than half of all political actors in the news in every country, well above their parliamentary seat share, while opposition parties as well as civil society actors and citizens remain marginally represented. These patterns are broadly similar across systems with varying pluralism risk profiles. Electoral campaigns broaden representation of the opposition parties across all selected countries.

## What this repository contains

The pipeline runs in three stages, mirrored by the top-level folders.

| Folder            | Stage                                                             | Produces                                      |
| ----------------- | ----------------------------------------------------------------- | --------------------------------------------- |
| `1_annotation/` | Topic filtering, LLM actor extraction, party entity-linking       | Actor-level annotations                       |
| `2_validation/` | Human intercoder reliability; LLM-vs-human validation             | Appendix A, Appendix B                        |
| `3_analysis/`   | Deduplication, diversity indices, tables, figures, tests          | Tables 3–5, Figures 1–2, Appendices C and D |
| `codebooks/`    | Manual annotation codebooks (immigration, rising prices)          | —                                            |
| `data/`         | Expected inputs and data-availability statement (no data shipped) | —                                            |

### Reproducing the results

This repository contains **code only**. No study data is distributed with it (see
[Data](#data) below), and no generated tables or figures are distributed either: running
the notebooks writes them to `results/`, which is git-ignored. The tables and figures in
the article are the authoritative versions. Once the required inputs are in place, the
pipeline runs as:

```bash
cd 3_analysis
jupyter lab      # 01 → 02 → 03 → 04 in order
```

| Notebook                       | Kernel      | Produces                                     |
| ------------------------------ | ----------- | -------------------------------------------- |
| `01_clean_corpus.ipynb`      | Python      | Deduplicated article-level inputs; Table C1  |
| `02_diversity_indices.ipynb` | Python      | Weekly *D*z scores for both sub-dimensions |
| `03_analysis_R.ipynb`        | **R** | Tables 3–5, C2–C3, D1–D4                  |
| `04_visualizations.ipynb`    | Python      | Figures 1–2                                 |

`01_clean_corpus.ipynb` deduplicates the corpus and writes the article-level inputs that
`02`–`04` consume to produce every table, figure and test in the paper. Notebook `03`
runs on an R kernel; the other three are Python. The expected layout of the input folders
is documented in [`data/README.md`](data/README.md).

Notebook `04` writes the two figures reported in the article to `results/figures/`, plus
15 exploratory figures to `results/figures/supplementary/` that the article does not
report. Nothing under `results/` is committed.

All paths resolve through `3_analysis/helpers.py`, which also holds the shared constants
(country and outlet labels, the deduplication threshold, the diversity formulas). Change a
label or threshold there and every notebook picks it up.

## Measures

Diversity is measured with the standardized Simpson's index, *D*z, computed per outlet and
week and then averaged within each country and week:

- *D* = 1 − Σ *p*ᵢ², the probability that two randomly drawn actors fall in different categories
- *D*z rescales *D* by its maximum, `1 − 1/k`, so the four-category societal and two-category
  political dimensions are on a 0–1 range

*D*z = 1 denotes equal representation across categories (the open-diversity benchmark) and
approaches 0 as coverage concentrates on a single category. Political actor diversity is
additionally evaluated against the reflective benchmark, operationalized as the governing
parties' share of party-linked political actors minus their share of parliamentary seats.

## Models

Annotation used open-source models served locally through [Ollama](https://ollama.com),
with temperature 0 and fixed seeds for reproducibility.

| Task                                  | Model                                                     | Notes                                                                  |
| ------------------------------------- | --------------------------------------------------------- | ---------------------------------------------------------------------- |
| Main topic and home-country relevance | [`phi4-mini:3.8b`](https://ollama.com/library/phi4-mini) | Selected over `llama3:8b` and `gpt-oss:20b` on precision and speed |
| Actor extraction and classification   | [`gpt-oss:20b`](https://ollama.com/library/gpt-oss)      | F1 .71 (Dutch), .81 (English), .84 (Turkish) against human coding      |

Per-language validation results are in `2_validation/`; the paper reports them in
Appendix B.

## Data

**No data is distributed with this repository.** The article corpus was retrieved under
licence from NexisUni and Belgapress and by direct collection from outlet websites, and
its redistribution is not permitted. The data derived from it (annotations, diversity
scores, reference files) is available **on request from the authors** only for
research purposes. The expected inputs, their formats, and the reference sources are
documented in [`data/README.md`](data/README.md).

## Environment

```bash
conda env create -f environment.yml
conda activate actor-diversity
```

or, with pip:

```bash
pip install -r requirements.txt
```

Notebook `03_analysis_R.ipynb` runs on R (≥ 4.3) rather than Python. Install the R
packages and the Jupyter R kernel with:

```r
install.packages(c("lme4", "emmeans", "performance", "rempsyc", "flextable", "IRkernel"))
IRkernel::installspec()
```

Stage 1 additionally needs a running Ollama server and the Stanza language models for
English, Dutch and Turkish; see `data/README.md`.

## How to cite

> [Author(s)]. (forthcoming). *Do Market and Political Risks Narrow the Diversity of Voices? A Comparative Analysis of Actor Representation in the News Prior to Elections.*

A DOI will be added on publication.

## Acknowledgements

The analysis was conducted on the SURF Research Cloud, provided and administered by SURF,
the IT cooperative for Dutch education and research institutions.

## License

[MIT](LICENSE) for the code. Party and politician reference data derive from
EveryPolitician, Wikidata, ParlGov and CHES, and remain subject to their original terms.

# Whose Voice Do We Hear Before the Vote? A Cross-country Comparison of Actor Diversity in News

Code and aggregated data for a comparative content analysis of **actor diversity** in
election news across five countries: Belgium (Flanders), Ireland, the Netherlands,
Turkey and the United Kingdom.

## Abstract

News media are expected to represent a diversity of voices, yet how far this expectation
is met, and how far it depends on the political conditions under which media operate,
remains largely untested outside Western democracies. This study analyses news coverage of
immigration and rising prices in the year preceding the 2023–2024 parliamentary elections
in five countries selected for their contrasting Media Pluralism Monitor risk scores on
political independence and market plurality. Actor diversity is assessed along two
sub-dimensions: *societal* actor diversity, the spread of coverage across four functional
categories (government and politics, professionals and experts, civil society, and
citizens), and *political* actor diversity, the balance between governing and opposition
parties, evaluated against both open and reflective normative benchmarks. Because the
corpus is large and multilingual, news actors were extracted and classified with
open-source generative large language models, validated throughout against human coding.

## What this repository contains

The pipeline runs in three stages, mirrored by the top-level folders.

| Folder | Stage | Produces |
|---|---|---|
| `1_annotation/` | Topic filtering, LLM actor extraction, party entity-linking | Actor-level annotations |
| `2_validation/` | Human intercoder reliability; LLM-vs-human validation | Appendix A, Appendix B |
| `3_analysis/` | Deduplication, diversity indices, tables, figures, tests | Tables 3–5, Figures 1–8, Appendix C |
| `codebooks/` | Manual annotation codebooks (immigration, rising prices) | — |
| `data/` | Expected inputs and data-availability statement (no data shipped) | — |
| `output/` | Generated tables and figures; safe to delete and rebuild | — |

### Reproducing the results

This repository contains **code only**. No study data is distributed with it (see
[Data](#data) below). Once the required inputs are in place under `data/`, the pipeline
runs as:

```bash
cd 3_analysis
jupyter lab      # 01 → 02 → 03 → 04 → 05 in order
```

`01_clean_corpus.ipynb` deduplicates the corpus and writes the article-level inputs that
`02`–`05` consume to produce every table, figure and test in the paper. The expected
layout of `data/` is documented in [`data/README.md`](data/README.md).

All paths resolve through `3_analysis/helpers.py`, which also holds the shared constants
(country and outlet labels, the deduplication threshold, the diversity formulas). Change a
label or threshold there and every notebook picks it up.

## Measures

Diversity is measured with the standardized Simpson's index, *D*z, computed per outlet and
week and then averaged within each country and week:

- *D* = 1 − Σ *p*ᵢ², the probability that two randomly drawn actors fall in different categories
- *D*z rescales *D* by its maximum, `1 − 1/k`, so the four-category societal and two-category
  political dimensions are comparable on a 0–1 range

*D*z = 1 denotes equal representation across categories (the open-diversity benchmark) and
approaches 0 as coverage concentrates on a single category. Political actor diversity is
additionally evaluated against the reflective benchmark, operationalized as the governing
parties' share of party-linked political actors minus their share of parliamentary seats.

## Models

Annotation used open-source models served locally through [Ollama](https://ollama.com),
with temperature 0 and fixed seeds for reproducibility.

| Task | Model | Notes |
|---|---|---|
| Main topic and home-country relevance | [`phi4-mini:3.8b`](https://ollama.com/library/phi4-mini) | Selected over `llama3:8b` and `gpt-oss:20b` on precision and speed |
| Actor extraction and classification | [`gpt-oss:20b`](https://ollama.com/library/gpt-oss) | F1 .71 (Dutch), .81 (English), .84 (Turkish) against human coding |

Per-language validation results are in `2_validation/`; the paper reports them in
Appendix B.

## Data

**No data is distributed with this repository.** The article corpus was retrieved under
licence from NexisUni and Belgapress and by direct collection from outlet websites, and
its redistribution is not permitted. The data derived from it (annotations, diversity
scores, reference files) is available **on reasonable request from the authors** for
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

Stage 1 additionally needs a running Ollama server and the Stanza language models for
English, Dutch and Turkish; see `data/README.md`.

## How to cite

> [Author(s)]. (forthcoming). *Whose voice do we hear before the vote? A cross-country
> comparison of actor diversity in news.*

A DOI will be added on publication.

## Acknowledgements

The analysis was conducted on the SURF Research Cloud, provided and administered by SURF,
the IT cooperative for Dutch education and research institutions.

## License

[MIT](LICENSE) for the code. Party and politician reference data derive from
EveryPolitician, Wikidata, ParlGov and CHES, and remain subject to their original terms.

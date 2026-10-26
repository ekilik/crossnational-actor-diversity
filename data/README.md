# Data

**No data is distributed with this repository.** This folder documents the inputs the
code expects and how the data can be obtained. Everything under `data/` other than this
file is git-ignored.

## Availability

The article corpus was retrieved under institutional licence from NexisUni (Ireland, the
Netherlands, Turkey, the United Kingdom) and Belgapress (Belgium), and by direct
collection from the public-service outlets' websites. Redistribution of the article text
is not permitted.

The data derived from that corpus (LLM and human annotations, actor-level enriched files,
and the aggregated weekly diversity scores), together with the party/politician reference
files, is available **on reasonable request from the authors** for non-commercial research
purposes. Requests are handled subject to the licences above.

## Expected layout

`3_analysis/helpers.py` resolves its paths from the file's own location, so the analysis
stage reads and writes **at the repository root**, not inside `data/`. Populate the tree as
follows; all three generated folders are git-ignored.

```
<repo root>/
├── analysis_data/     # inputs to notebook 01, and its cleaned outputs
│   ├── *_df*.csv                                  # per-country topic files (with body text)
│   ├── df_actors_socialdiv_analysis.csv            # article-level, one row per article
│   ├── df_actors_politicaldiv_analysis.csv         # actor-level, one row per political actor
│   ├── df_actors_socialdiv_analysis_clean.csv      # written by notebook 01
│   └── df_actors_politicaldiv_analysis_clean.csv   # written by notebook 01
├── model_dfs/         # weekly Dz scores, written by notebook 02 and read by 03 and 04
├── results/           # tables (docx + csv) and figures, written by 03 and 04
└── data/              # annotation-stage inputs only (stage 1); see below
    ├── actor_annotations/ # enriched actor files (output of the enrichment stage)
    ├── reference/         # party/politician reference files
    │   ├── {cc}_parties_reference.xlsx
    │   ├── {cc}_politicians_reference.xlsx
    │   ├── raw/           # election rosters (input to prepare_politicians_reference.py)
    │   └── intermediate/  # pre-matched actor lists
    └── raw_annotations/   # stage-2 LLM output, keyed to article text
        ├── immigration/
        └── inflation/
```

Notebook `02` writes the aggregated diversity scores to `model_dfs/`. These files are the
analysis-ready grain: pooled actor counts and *D*z per outlet-week-issue, with no article
text, headlines or identifiers. They are the smallest set sufficient to reproduce every
table and figure in the paper from notebook `03` onward, and are included in the
on-request release.

Country codes: `be` Belgium · `ir` Ireland · `nl` Netherlands · `tr` Turkey · `uk` United Kingdom

## Input format for stage 1

All project CSVs are semicolon-separated (`helpers.CSV_SEP`). To run stage 1
(`1_annotation/`) or notebook `01_clean_corpus.ipynb`, supply your own article data as a
semicolon-separated CSV with at least:

| Column | Description |
|---|---|
| `article_id` | unique integer ID; the first digit encodes country (1 UK, 2 IE, 3 NL, 4 BE, 5 TR) |
| `title` | headline |
| `text` | article body |

### Software prerequisites for stage 1

```bash
ollama serve                      # models are pulled on first use
python -c "import stanza; [stanza.download(l) for l in ('en','nl','tr')]"
```

## Reference sources

| Source | Used for |
|---|---|
| [Media Pluralism Monitor](https://cmpf.eui.eu/media-pluralism-monitor/) | Country risk scores (market plurality, political independence) |
| [EveryPolitician](https://everypolitician.org/) / [Wikidata](https://www.wikidata.org/) | Politician-to-party linking |
| [ParlGov](https://www.parlgov.org/) | Party positions, seat shares |
| [CHES](https://www.chesdata.eu/) | Party ideology scores |
| [Eurobarometer](https://europa.eu/eurobarometer/) | Issue salience, for issue selection |

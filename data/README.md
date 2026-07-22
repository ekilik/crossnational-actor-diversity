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

To run the pipeline, populate `data/` as follows. The code resolves these paths through
`3_analysis/helpers.py` (analysis) and the `DATA_DIR` constants in `1_annotation/scripts/`
(annotation).

```
data/
├── analysis/          # article-level inputs to notebooks 02-05 (output of notebook 01)
│   ├── df_actors_socialdiv_analysis_clean.csv
│   └── df_actors_politicaldiv_analysis_clean.csv
├── diversity/         # weekly Dz scores (output of notebook 02)
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

Country codes: `be` Belgium · `ir` Ireland · `nl` Netherlands · `tr` Turkey · `uk` United Kingdom

## Input format for stage 1

To run stage 1 (`1_annotation/`) or notebook `01_clean_corpus.ipynb`, supply your own
article data as a semicolon-separated CSV with at least:

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

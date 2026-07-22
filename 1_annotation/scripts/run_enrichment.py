"""
Runs actor annotation enrichment for all countries × both topics (Immigration, Inflation).

Prerequisites:
  1. Run PoliticalParties/scripts/prepare_politicians_reference.py first to build reference CSVs.
  2. actor_enricher.py must be in the same directory as this script.

Usage:
  # All countries, all topics:
  python run_enrichment.py

  # Single country / topic:
  python run_enrichment.py --country nl --topic immigration

  # With Wikidata fallback (slower, but fills gaps):
  python run_enrichment.py --use_wikidata

Outputs per (country, topic):
  {topic}/actors/enriched/{cc}_{topic}_actors_enriched.csv   — actor-level, one row per actor
  {topic}/actors/enriched/{cc}_{topic}_actor_stats.csv       — article-level counts per function
"""

import os
import sys
import csv
import argparse
import pandas as pd

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPTS_DIR))
DATA_DIR = os.path.join(REPO_ROOT, 'data')
sys.path.insert(0, SCRIPTS_DIR)

from actor_enricher import ActorEnricher  # noqa: E402

# ---------------------------------------------------------------------------
# Per-country configuration
# ---------------------------------------------------------------------------
COUNTRY_CONFIG = {
    'nl': {'language': 'nl', 'center_parties': True,  'id_column': 'article_id'},
    'be': {'language': 'nl', 'center_parties': True,  'id_column': 'article_id'},
    'ir': {'language': 'en', 'center_parties': True,  'id_column': 'article_id'},
    'tr': {'language': 'tr', 'center_parties': False, 'id_column': 'article_id'},
    'uk': {'language': 'en', 'center_parties': False, 'id_column': 'article_id'},
}

# ---------------------------------------------------------------------------
# Input file resolution per topic
# ---------------------------------------------------------------------------
# Phase 2 output. Not distributed with the repository (it is keyed to article text
# the licences do not allow us to redistribute); see data/README.md.
IMMIGRATION_DIR = os.path.join(DATA_DIR, 'raw_annotations', 'immigration')
INFLATION_DIR = os.path.join(DATA_DIR, 'raw_annotations', 'inflation')
PARTIES_DIR = os.path.join(DATA_DIR, 'reference')


RAW_DIRS = {
    'immigration': IMMIGRATION_DIR,
    'inflation': INFLATION_DIR,
}


def resolve_input(cc: str, topic: str) -> pd.DataFrame:
    """Load annotation CSV(s) for a country/topic, concatenating batch files if present."""
    raw_dir = RAW_DIRS[topic]
    main_path = os.path.join(raw_dir, f'{cc}_{topic}_df_actors.csv')
    if not os.path.exists(main_path):
        raise FileNotFoundError(f"Input not found: {main_path}")

    df = pd.read_csv(main_path, sep=';', quoting=csv.QUOTE_NONNUMERIC, dtype=str)

    for batch_n in range(2, 10):
        batch_path = os.path.join(raw_dir, f'{cc}_{topic}_df_actors_batch{batch_n}.csv')
        if os.path.exists(batch_path):
            batch = pd.read_csv(batch_path, sep=';', quoting=csv.QUOTE_NONNUMERIC, dtype=str)
            df = pd.concat([df, batch], ignore_index=True)
            print(f"  [{cc}] Concatenated batch{batch_n}: +{len(batch)} rows")

    return df

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _save_outputs(enricher, actor_stats_df: pd.DataFrame, output_prefix: str, cc: str, suffix: str = '') -> None:
    """Save the three output files for one enrichment run (full or a single batch part)."""
    stats_path = f"{output_prefix}_actor_stats{suffix}.csv"
    actor_stats_df.to_csv(stats_path, sep=';', index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"  [{cc}] Saved article stats ({len(actor_stats_df)} rows) → {stats_path}")

    if hasattr(enricher, 'enriched_actors_df') and not enricher.enriched_actors_df.empty:
        actors_path = f"{output_prefix}_actors_enriched{suffix}.csv"
        enricher.enriched_actors_df.to_csv(actors_path, sep=';', index=False, quoting=csv.QUOTE_NONNUMERIC)
        print(f"  [{cc}] Saved enriched actors ({len(enricher.enriched_actors_df)} rows) → {actors_path}")

    if hasattr(enricher, 'actors_per_party_df') and not enricher.actors_per_party_df.empty:
        party_path = f"{output_prefix}_actors_per_party{suffix}.csv"
        enricher.actors_per_party_df.to_csv(party_path, sep=';', index=False, quoting=csv.QUOTE_NONNUMERIC)
        print(f"  [{cc}] Saved actors per party ({len(enricher.actors_per_party_df)} rows) → {party_path}")


def run_one(cc: str, topic: str, use_wikidata: bool, batch_size: int = 0) -> None:
    cfg = COUNTRY_CONFIG[cc]

    politicians_ref_path = os.path.join(PARTIES_DIR, f'{cc}_politicians_reference.xlsx')
    if not os.path.exists(politicians_ref_path):
        print(f"  [{cc}] WARNING: politicians reference not found: {politicians_ref_path}")
        print(f"  [{cc}]   Run PoliticalParties/scripts/prepare_politicians_reference.py first.")
        politicians_ref_path = None

    party_data_path = os.path.join(PARTIES_DIR, f'{cc}_parties_reference.xlsx')
    if not os.path.exists(party_data_path):
        print(f"  [{cc}] WARNING: party reference not found: {party_data_path}")
        party_data_path = None

    print(f"\n{'='*60}")
    print(f"  Country: {cc.upper()}  |  Topic: {topic.upper()}")
    print(f"{'='*60}")

    # Load input data
    try:
        actor_df = resolve_input(cc, topic)
    except FileNotFoundError as e:
        print(f"  [{cc}] Skipping — {e}")
        return

    print(f"  [{cc}] Loaded {len(actor_df)} articles")

    # Output directory
    out_dir = os.path.join(DATA_DIR, 'actor_annotations')
    os.makedirs(out_dir, exist_ok=True)
    output_prefix = os.path.join(out_dir, f'{cc}_{topic}')

    # Initialize enricher once (loads Stanza + reference data — expensive, do it once)
    enricher = ActorEnricher(
        actor_df=actor_df,
        id_column=cfg['id_column'],
        language=cfg['language'],
        center_parties=cfg['center_parties'],
        politicians_data_path=politicians_ref_path,
        party_data_path=party_data_path,
        function_map=cfg.get('function_map'),
    )

    if batch_size <= 0:
        # --- Single-run mode (original behaviour) ---
        actor_stats_df = enricher.run_full_enrichment(
            use_wikidata=use_wikidata,
            language=cfg['language'],
        )
        _save_outputs(enricher, actor_stats_df, output_prefix, cc)
        return

    # --- Batched mode ---
    id_col = cfg['id_column']
    article_ids = actor_df[id_col].unique()
    n_batches = (len(article_ids) + batch_size - 1) // batch_size
    print(f"  [{cc}] Batched mode: {len(article_ids)} unique articles → {n_batches} batches of ≤{batch_size}")

    all_stats: list[pd.DataFrame] = []
    all_enriched: list[pd.DataFrame] = []
    all_per_party: list[pd.DataFrame] = []

    for batch_i in range(n_batches):
        part_suffix = f'_part{batch_i + 1}of{n_batches}'
        part_stats_path = f"{output_prefix}_actor_stats{part_suffix}.csv"

        if os.path.exists(part_stats_path):
            # Resume: load already-completed batch
            print(f"\n  [{cc}] Batch {batch_i + 1}/{n_batches} already done — loading from disk")
            all_stats.append(pd.read_csv(part_stats_path, sep=';', quoting=csv.QUOTE_NONNUMERIC, dtype=str))
            part_enriched = f"{output_prefix}_actors_enriched{part_suffix}.csv"
            if os.path.exists(part_enriched):
                all_enriched.append(pd.read_csv(part_enriched, sep=';', quoting=csv.QUOTE_NONNUMERIC, dtype=str))
            part_party = f"{output_prefix}_actors_per_party{part_suffix}.csv"
            if os.path.exists(part_party):
                all_per_party.append(pd.read_csv(part_party, sep=';', quoting=csv.QUOTE_NONNUMERIC, dtype=str))
            continue

        batch_ids = article_ids[batch_i * batch_size : (batch_i + 1) * batch_size]
        batch_df = actor_df[actor_df[id_col].isin(batch_ids)].copy()

        print(f"\n  [{cc}] === Batch {batch_i + 1}/{n_batches} ({len(batch_df)} rows) ===")
        stats_df = enricher.run_full_enrichment(
            use_wikidata=use_wikidata,
            language=cfg['language'],
            actor_df=batch_df,
        )

        _save_outputs(enricher, stats_df, output_prefix, cc, suffix=part_suffix)

        all_stats.append(stats_df)
        if hasattr(enricher, 'enriched_actors_df') and not enricher.enriched_actors_df.empty:
            all_enriched.append(enricher.enriched_actors_df.copy())
        if hasattr(enricher, 'actors_per_party_df') and not enricher.actors_per_party_df.empty:
            all_per_party.append(enricher.actors_per_party_df.copy())

    # Merge all batches into final output files
    print(f"\n  [{cc}] Merging {n_batches} batches into final outputs…")

    if all_stats:
        merged_stats = pd.concat(all_stats, ignore_index=True)
        stats_path = f"{output_prefix}_actor_stats.csv"
        merged_stats.to_csv(stats_path, sep=';', index=False, quoting=csv.QUOTE_NONNUMERIC)
        print(f"  [{cc}] Saved merged article stats ({len(merged_stats)} rows) → {stats_path}")

    if all_enriched:
        merged_enriched = pd.concat(all_enriched, ignore_index=True)
        actors_path = f"{output_prefix}_actors_enriched.csv"
        merged_enriched.to_csv(actors_path, sep=';', index=False, quoting=csv.QUOTE_NONNUMERIC)
        print(f"  [{cc}] Saved merged enriched actors ({len(merged_enriched)} rows) → {actors_path}")

    if all_per_party:
        merged_party = pd.concat(all_per_party, ignore_index=True)
        party_path = f"{output_prefix}_actors_per_party.csv"
        merged_party.to_csv(party_path, sep=';', index=False, quoting=csv.QUOTE_NONNUMERIC)
        print(f"  [{cc}] Saved merged actors per party ({len(merged_party)} rows) → {party_path}")


def main():
    parser = argparse.ArgumentParser(description='Run actor enrichment for all countries × topics')
    parser.add_argument(
        '--country',
        nargs='+',
        default=list(COUNTRY_CONFIG.keys()),
        choices=list(COUNTRY_CONFIG.keys()) + ['all'],
        help='Country code(s) to process (default: all)',
    )
    parser.add_argument(
        '--topic',
        nargs='+',
        default=['immigration', 'inflation'],
        choices=['immigration', 'inflation', 'all'],
        help='Topic(s) to process (default: both)',
    )
    parser.add_argument(
        '--use_wikidata',
        action='store_true',
        help='Query Wikidata for actors not matched by reference data (slow)',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=0,
        help='Number of articles per batch (0 = no batching, process all at once)',
    )
    args = parser.parse_args()

    countries = list(COUNTRY_CONFIG.keys()) if 'all' in args.country else args.country
    topics = ['immigration', 'inflation'] if 'all' in args.topic else args.topic

    print(f"Countries:  {countries}")
    print(f"Topics:     {topics}")
    print(f"Wikidata:   {args.use_wikidata}")
    print(f"Batch size: {args.batch_size if args.batch_size > 0 else 'none (single run)'}")

    for topic in topics:
        for cc in countries:
            run_one(cc, topic, use_wikidata=args.use_wikidata, batch_size=args.batch_size)

    print(f"\n{'='*60}")
    print("ALL DONE")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

"""
Builds per-country politician reference Excel files for use with actor_enricher.py.

Output format per country: reference/{cc}_politicians_reference.xlsx
Columns: name, party, party_final, lrgen_cat

Two sources are combined per country:
  1. intermediate/{cc}_all_political_actors.csv  — actors already matched by previous notebooks
  2. raw/politicians_{year}_elections_{cc}.csv   — broader election roster

Both sources are joined with reference/{cc}_parties_reference.xlsx to attach
party_final and lrgen_cat. Source 1 takes precedence for duplicate names.
Only politicians whose party maps to a known lrgen_cat are kept.
"""

import os
import csv
import argparse
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(REPO_ROOT, 'data')
# raw/ and intermediate/ hold the election rosters and pre-matched actor lists used to
# build the reference files; only the built reference/ output is distributed.
RAW_DIR = os.path.join(DATA_DIR, 'reference', 'raw')
INTERMEDIATE_DIR = os.path.join(DATA_DIR, 'reference', 'intermediate')
REFERENCE_DIR = os.path.join(DATA_DIR, 'reference')

COUNTRY_CONFIG = {
    'nl': {
        'actors_file': 'nl_all_political_actors.csv',
        'elections_file': 'politicians_2023_elections_NL.csv',
        'parties_file': 'nl_parties_reference.xlsx',
    },
    'be': {
        'actors_file': 'be_all_political_actors.csv',
        'elections_file': 'politicians_2024_elections_BE.csv',
        'parties_file': 'be_parties_reference.xlsx',
    },
    'ir': {
        'actors_file': 'ir_all_political_actors.csv',
        'elections_file': 'politicians_2024_elections_IR.csv',
        'parties_file': 'ir_parties_reference.xlsx',
    },
    'tr': {
        'actors_file': 'tr_all_political_actors.csv',
        'elections_file': 'politicians_2023_elections_TR.csv',
        'parties_file': 'tr_parties_reference.xlsx',
    },
    'uk': {
        'actors_file': 'uk_all_political_actors.csv',
        'elections_file': 'politicians_2024_elections_UK.csv',
        'parties_file': 'uk_parties_reference.xlsx',
    },
}


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=';', quoting=csv.QUOTE_NONNUMERIC, dtype=str)


def load_parties(path: str) -> pd.DataFrame:
    """Load party reference, normalising the short-name column to 'party'."""
    df = pd.read_excel(path, dtype=str, engine='openpyxl')
    if 'party' in df.columns and 'party_name_short' in df.columns:
        df = df.drop(columns=['party'])
    if 'party_name_short' in df.columns:
        df = df.rename(columns={'party_name_short': 'party'})
    return df[df['party'].notna() & (df['party'].str.strip() != '')]


def party_lookup(parties_df: pd.DataFrame) -> pd.DataFrame:
    """Return a slim party→party_final/lrgen_cat lookup table."""
    cols = [c for c in ['party', 'party_final', 'lrgen_cat'] if c in parties_df.columns]
    return parties_df[cols].drop_duplicates(subset=['party'])


def build_source1(actors_path: str, lookup: pd.DataFrame) -> pd.DataFrame:
    """Extract name/party from matched-actors file, then attach party_final and lrgen_cat."""
    df = load_csv(actors_path)
    if 'party' in df.columns and 'party_name_short' in df.columns:
        df = df.drop(columns=['party'])
    df = df.rename(columns={'stanza_names': 'name', 'party_name_short': 'party'})
    df = df[['name', 'party']].copy()
    df = df[df['name'].notna() & (df['name'].str.strip() != '')]
    df = df[df['party'].notna() & (df['party'].str.strip() != '')]
    df = df.merge(lookup, on='party', how='left')
    return df[df['lrgen_cat'].notna()].drop_duplicates(subset=['name'])


def build_source2(elections_path: str, lookup: pd.DataFrame) -> pd.DataFrame:
    """Join election roster with party reference to attach party_final and lrgen_cat."""
    elections = load_csv(elections_path)
    merged = elections.merge(lookup, left_on='party_name_short', right_on='party', how='inner')
    merged = merged.rename(columns={'politician': 'name'})
    merged = merged[['name', 'party', 'party_final', 'lrgen_cat']]
    merged = merged[merged['name'].notna() & (merged['name'].str.strip() != '')]
    return merged[merged['lrgen_cat'].notna()].drop_duplicates(subset=['name'])


def build_reference(cc: str, config: dict) -> pd.DataFrame:
    actors_path = os.path.join(INTERMEDIATE_DIR, config['actors_file'])
    elections_path = os.path.join(RAW_DIR, config['elections_file'])
    parties_path = os.path.join(REFERENCE_DIR, config['parties_file'])

    if not os.path.exists(parties_path):
        print(f"  [{cc}] ERROR: parties reference not found: {parties_path}")
        return pd.DataFrame()

    lookup = party_lookup(load_parties(parties_path))

    frames = []

    if os.path.exists(actors_path):
        src1 = build_source1(actors_path, lookup)
        print(f"  [{cc}] Source 1 (matched actors): {len(src1)} rows")
        frames.append(src1)
    else:
        print(f"  [{cc}] WARNING: actors file not found: {actors_path}")

    if os.path.exists(elections_path):
        src2 = build_source2(elections_path, lookup)
        print(f"  [{cc}] Source 2 (election roster): {len(src2)} rows")
        frames.append(src2)
    else:
        print(f"  [{cc}] WARNING: elections file not found: {elections_path}")

    if not frames:
        print(f"  [{cc}] ERROR: no source data available, skipping")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=['name'], keep='first')
    combined = combined.sort_values('name').reset_index(drop=True)
    return combined[['name', 'party', 'party_final', 'lrgen_cat']]


def main(countries: list):
    for cc in countries:
        if cc not in COUNTRY_CONFIG:
            print(f"Unknown country code: {cc}. Valid: {list(COUNTRY_CONFIG)}")
            continue
        print(f"\nBuilding reference for: {cc.upper()}")
        ref = build_reference(cc, COUNTRY_CONFIG[cc])
        if ref.empty:
            continue
        os.makedirs(REFERENCE_DIR, exist_ok=True)
        out_path = os.path.join(REFERENCE_DIR, f"{cc}_politicians_reference.xlsx")
        ref.to_excel(out_path, index=False, engine='openpyxl')
        print(f"  [{cc}] Saved {len(ref)} entries → {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build per-country politician reference files')
    parser.add_argument(
        '--country',
        nargs='+',
        default=list(COUNTRY_CONFIG.keys()),
        choices=list(COUNTRY_CONFIG.keys()),
        help='Country code(s) to process (default: all)'
    )
    args = parser.parse_args()
    main(args.country)

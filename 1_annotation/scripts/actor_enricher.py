import random

import pandas as pd
import json
import csv
import re
import unicodedata
import os
import argparse
from typing import List, Dict, Optional
import stanza
from tqdm import tqdm


class ActorEnricher:
    """
    Enriches actor annotations from LLM output by:
    1. Parsing and cleaning LLM JSON output
    2. Expanding actors to row-level data
    3. Calculating actor statistics per function category
    4. Extracting core names using NER
    5. Enriching political actors with party affiliations from multiple sources
    """

    # Wikimedia-compliant User-Agent: https://meta.wikimedia.org/wiki/User-Agent_policy
    # Override by setting WIKIDATA_UA in the environment (or a .env file loaded at startup).
    _WIKIDATA_UA = os.environ.get(
        "WIKIDATA_UA",
        "ActorEnricher/1.0 (https://github.com/your-repo; your-email@example.com)",
    )

    # Generic cross-country mapping of non-standard actor_function values → a/b/c/d.
    # Sourced from manual corrections across all 5 immigration notebooks.
    _GENERIC_FUNCTION_MAP: Dict[str, str] = {
        # → a (politicians / government)
        'government': 'a', 'government official': 'a', 'government and politics': 'a',
        'government agency': 'a', 'government agencies': 'a', 'government administration': 'a',
        'government institution': 'a', 'government department': 'a',
        'politician': 'a', 'political figure': 'a', 'political organization': 'a',
        'local politician': 'a', 'national politician': 'a',
        'minister': 'a', 'prime minister': 'a', 'president': 'a', 'former president': 'a',
        'official': 'a', 'candidate': 'a', 'governor': 'a',
        'military organization': 'a', 'prosecutor': 'a', 'police officer': 'a',
        'function a': 'a',
        # → b (experts / professionals / media)
        'journalist': 'b', 'media': 'b', 'news source': 'b', 'interviewer': 'b',
        'professional': 'b', 'professionals and experts': 'b',
        'professional/expert': 'b', 'professionals/experts': 'b',
        'expert': 'b', 'economist': 'b', 'economists': 'b',
        'researcher': 'b', 'professor': 'b', 'academic': 'b', 'academic/professor': 'b',
        'political scientist': 'b', 'professor of political science': 'b',
        'function b': 'b',
        # → c (civil society / NGOs)
        'civil society': 'c', 'civil society organizations': 'c', 'ngo': 'c',
        'function c': 'c',
        # → d (general public)
        'citizen': 'd', 'citizens': 'd', 'activist': 'd', 'student': 'd', 'interviewee': 'd',
        'function d': 'd',
    }

    def __init__(
            self,
            actor_data_path: Optional[str] = None,
            actor_df: Optional[pd.DataFrame] = None,
            id_column: str = 'uri',
            language: str = 'en',
            center_parties: bool = True,
            politicians_data_path: Optional[str] = None,
            party_data_path: Optional[str] = None,
            function_map: Optional[Dict[str, str]] = None):

        """
        Initialize ActorEnricher.

        Args:
            actor_data_path: Path to CSV file with actor annotations, optional
            actor_df: DataFrame with actor annotations (alternative to path if path not provided)
            id_column: Name of the column containing unique article identifiers
            language: Language for NER processing ('en', 'nl', etc.)
            center_parties: Whether the country has center parties in addition to left and right
            politicians_data_path: Path to XLSX with politician reference data (columns: name, party)
            party_data_path: Path to XLSX with party ideology scores (political_parties_{CC}.xlsx).
                             Used for party-name mention matching. Covers all parties including those
                             without individual politicians in the reference data.
        """

        self.actor_data_path = actor_data_path
        self.politicians_data_path = politicians_data_path
        self.id_column = id_column
        self.language = language
        self.center_parties = center_parties
        # Merge generic map with any country-specific overrides (overrides win)
        self._function_map = {**self._GENERIC_FUNCTION_MAP, **(function_map or {})}

        # Load actor data
        self.actor_df = actor_df if actor_df is not None else self._load_actor_data()
        # Initialize NLP pipeline for NER
        print(f"Initializing Stanza NLP pipeline for language: {language}")
        self.nlp = stanza.Pipeline(
            language=self.language,
            processors='tokenize,ner',
            tokenize_no_ssplit=True,
            verbose=False
        )

        # ideology_reference_df: built from party Excel — covers ALL parties for mention matching
        _party_df = self._load_party_data(party_data_path)
        if _party_df is not None:
            self.score_cols = ['lrgen_cat'] if 'lrgen_cat' in _party_df.columns else []
            party_ref_cols = [c for c in ['party', 'party_name', 'party_final', 'lrgen_cat']
                              if c in _party_df.columns]
            self.ideology_reference_df: Optional[List[Dict]] = (
                _party_df[party_ref_cols].drop_duplicates().to_dict('records')
            )
        else:
            self.score_cols = []
            self.ideology_reference_df = None

        # politician_reference_df: built from politicians XLSX — name→party lookup only
        # lrgen_cat and party_final are resolved from the party reference after matching
        _political_df = self._load_politicians_data(politicians_data_path)
        if _political_df is not None:
            self.politician_reference_df: Optional[List[Dict]] = (
                _political_df[['name', 'party']].drop_duplicates().to_dict('records')
            )
        else:
            self.politician_reference_df = None

    def _load_actor_data(self) -> pd.DataFrame:
            """Load actor data from CSV file into a DataFrame"""
            if self.actor_data_path:
                return pd.read_csv(self.actor_data_path, sep=';', quoting=csv.QUOTE_NONNUMERIC)
            else:
                raise ValueError("No actor data path provided and no DataFrame was passed.")

    def _load_politicians_data(self, path: Optional[str]) -> Optional[pd.DataFrame]:
        """Load politician reference XLSX for individual name matching."""
        if not path:
            print("  WARNING: No politicians_data_path provided — skipping politician matching.")
            return None
        if not os.path.exists(path):
            print(f"  WARNING: Politicians reference not found at: {path}")
            return None
        print(f"Loading politician reference data from {path}")
        df = pd.read_excel(path, dtype=str, engine='openpyxl')
        return df.drop_duplicates(subset=['name'], keep='first')

    def _load_party_data(self, path: Optional[str]) -> Optional[pd.DataFrame]:
        """Load party ideology XLSX for party-mention matching."""
        if path and os.path.exists(path):
            print(f"Loading party data from {path}")
            df = pd.read_excel(path, dtype=str, engine='openpyxl')
            if 'party' in df.columns and 'party_name_short' in df.columns:
                df = df.drop(columns=['party'])
            df = df.rename(columns={'party_name_short': 'party'})
            df = df[df['party'].notna() & (df['party'].str.strip() != '')]
            # keep party_name as matching alias if present
            if 'party_name' not in df.columns:
                df['party_name'] = None
            return df
        return None

    @staticmethod
    def _clean_json_string(s: str) -> str:
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"^```(?:json)?|```$", "", s, flags=re.IGNORECASE | re.MULTILINE).strip()
        return re.sub(r"\s+", " ", s)

    def _extract_json_from_text(self, text: str):
        """Extract JSON structure from messy text."""
        obj_match = re.search(r'\{[^{}]*"actors"\s*:\s*\[[^\]]*\][^{}]*\}', text, re.DOTALL | re.IGNORECASE)
        if obj_match:
            try:
                return json.loads(obj_match.group(0))
            except json.JSONDecodeError:
                pass
        array_match = re.search(r'\[\s*\{[^]]+\}\s*(?:,\s*\{[^]]+\}\s*)*\]', text, re.DOTALL)
        if array_match:
            try:
                return json.loads(array_match.group(0))
            except json.JSONDecodeError:
                pass
        return None

    def _get_actors_from_data(self, data):
        """Extract actors list from various data structures."""
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in data:
                if key.lower() == 'actors':
                    return data[key]
        return None

    def _normalize_actor_fields(self, actor: dict) -> Optional[dict]:
        """Normalize actor dict to standard field names, handling LLM variation."""
        name_fields = ['actor_name', 'name', 'actor', 'actorName', 'actor name',
                       'name/description', 'actor_name/description']
        function_fields = ['actor_function', 'function', 'role', 'actor_type', 'actorType',
                           'category', 'type', 'actor_f', 'actor_type (function)']
        party_fields = ['actor_pp', 'party', 'pp', 'actorParty']

        def _first(fields):
            for f in fields:
                if f in actor:
                    v = actor[f]
                    if isinstance(v, list):
                        v = v[0] if v else None
                    if v is not None and str(v).strip():
                        return str(v).strip()
            return ''

        actor_name = _first(name_fields)
        if not actor_name:
            return None
        actor_pp = _first(party_fields)
        if actor_pp == 'null':
            actor_pp = ''
        return {
            'actor_name': actor_name,
            'actor_function': _first(function_fields),
            'actor_pp': actor_pp,
        }

    def _fallback_parse(self, raw: str) -> Optional[str]:
        """Parse malformed LLM output, returning normalized JSON string or None."""
        if not isinstance(raw, str) or not raw.strip():
            return None
        # Guard against catastrophic regex backtracking on garbage/oversized strings.
        # A valid actor JSON list is never more than ~5k chars; anything longer is truncated.
        if len(raw) > 5_000:
            raw = raw[:5_000]
        cleaned = self._clean_json_string(raw)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            data = self._extract_json_from_text(cleaned)
        if not data:
            return None
        actors_list = self._get_actors_from_data(data)
        if not isinstance(actors_list, list):
            return None
        normalized = [self._normalize_actor_fields(a) for a in actors_list if isinstance(a, dict)]
        normalized = [a for a in normalized if a]
        return json.dumps({'actors': normalized}, ensure_ascii=False) if normalized else None

    def _clean_actor_function(self, value: str) -> str:
        """
        Normalise a raw actor_function value to one of a/b/c/d.
        Returns the original value (lowercased) if already valid, looks it up
        in the combined generic + country-specific map, or returns '' if unknown.
        """
        if not isinstance(value, str):
            return ''
        v = value.strip().lower()
        if v in ('a', 'b', 'c', 'd'):
            return v
        return self._function_map.get(v, '')

    def _parse_actors_from_row(self, row: dict):
        """
        Parse actors from a data row dict, trying news_actors first then
        falling back to news_actors_raw via the fallback parser.
        """
        def _extract(json_str):
            if json_str is None or (not isinstance(json_str, str) and pd.isna(json_str)):
                return None
            cleaned = self._clean_json_string(json_str)
            try:
                data = json.loads(cleaned)
                actors = data.get('actors', [])
                if actors:
                    return (
                        [a.get('actor_name', '') for a in actors],
                        [a.get('actor_function', '') for a in actors],
                        [a.get('actor_pp', '') for a in actors],
                    )
            except (json.JSONDecodeError, AttributeError):
                pass
            return None

        result = _extract(row.get('news_actors'))
        if result:
            return result

        # Fallback: try to repair and re-parse from raw LLM output
        raw = row.get('news_actors_raw')
        if raw and isinstance(raw, str) and raw.strip():
            repaired = self._fallback_parse(raw)
            if repaired:
                result = _extract(repaired)
                if result:
                    return result

        return [], [], []

    def expand_actors_to_rows(self, actor_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Transform DataFrame from article-level to actor-level.
        Each actor becomes a separate row with article metadata.
        Articles without actors get a row with empty values in actor columns.

        Args:
            actor_df: Optional DataFrame to expand. If None, uses self.actor_df

        Returns:
            DataFrame with one row per actor
        """
        if actor_df is None:
            actor_df = self.actor_df

        if actor_df.empty:
            return pd.DataFrame()

        # Validate that the ID column exists
        if self.id_column not in actor_df.columns:
            raise ValueError(
                f"ID column '{self.id_column}' not found in DataFrame. "
                f"Available columns: {list(actor_df.columns)}")

        rows = []
        for row in actor_df.to_dict('records'):
            names, functions, parties = self._parse_actors_from_row(row)
            raw_output = row.get('news_actors_raw', '')

            # If no actors found, create one row with empty values
            if len(names) == 0:
                rows.append({
                    self.id_column: row[self.id_column],
                    'actor_name': '',
                    'actor_function': '',
                    'actor_function_raw': '',
                    'actor_pp': '',
                    'news_actors_raw': raw_output,
                })
            else:
                # Each actor as a separate row
                for i in range(len(names)):
                    raw_func = functions[i] if i < len(functions) else ''
                    rows.append({
                        self.id_column: row[self.id_column],
                        'actor_name': names[i] if i < len(names) else '',
                        'actor_function': self._clean_actor_function(raw_func),
                        'actor_function_raw': raw_func,
                        'actor_pp': parties[i] if i < len(parties) else '',
                        'news_actors_raw': raw_output,
                    })

        if not rows:
            return pd.DataFrame(columns=[
                self.id_column, 'actor_name', 'actor_function', 'actor_function_raw', 'actor_pp', 'news_actors_raw'
            ])
        return pd.DataFrame(rows)

    # calculate nr of unique actors per function per article
    def calculate_actors_per_function(self, actor_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate number of unique actors per function for each article.

        Args:
            actor_df: Optional actor-level DataFrame. If None, expands self.actor_df

        Returns:
            DataFrame with columns: id_column, nr_actors_a, nr_actors_b, nr_actors_c,
            nr_actors_d, nr_actors_total
        """
        if actor_df is None:
            actor_df = self.expand_actors_to_rows()

        if actor_df.empty:
            print("Actor DataFrame is empty. Returning empty DataFrame.")
            return pd.DataFrame()

        # Keep only valid functions
        valid_functions = ['a', 'b', 'c', 'd']
        actor_df = actor_df[actor_df['actor_function'].isin(valid_functions)]

        if actor_df.empty:
            print("No actors with valid functions (a, b, c, d) found.")
            return pd.DataFrame()

        # Count unique actor names per (id, function)
        unique_counts = (
            actor_df
            .groupby([self.id_column, 'actor_function'])['actor_name']
            .nunique()
            .reset_index()
            .rename(columns={'actor_name': 'nr_actors'})
        )

        # Pivot to have functions as columns
        functions_df = (
            unique_counts
            .pivot(index=self.id_column, columns='actor_function', values='nr_actors')
            .fillna(0)
            .reset_index()
        )
        functions_df.columns.name = None

        # if one of the function columns is missing, add it with zeros
        for func in valid_functions:
            if func not in functions_df.columns:
                functions_df[func] = 0

        # Rename columns
        rename_map = {f: f'nr_actors_{f}' for f in valid_functions if f in functions_df.columns}
        functions_df = functions_df.rename(columns=rename_map)

        # Calculate total number of unique actors
        actor_cols = [c for c in functions_df.columns if c.startswith('nr_actors_')]
        functions_df['nr_actors_total'] = functions_df[actor_cols].sum(axis=1)

        return functions_df

    def _clean_actor_name(self, name: str) -> str:
        """Remove text in parentheses and extra whitespace."""
        if name is None or (not isinstance(name, str) and pd.isna(name)):
            return ""
        return re.sub(r"\(.*?\)", "", str(name)).strip()

    def extract_core_name(self, full_name: str) -> Optional[str]:
        """
        Extract the core person name using NER.

        Args:
            full_name: Full actor name string

        Returns:
            Extracted person name or None if not a person
        """
        if full_name is None or pd.isna(full_name) or not str(full_name).strip():
            return None

        clean_name = self._clean_actor_name(full_name)
        if not clean_name:
            return None

        doc = self.nlp(clean_name)

        # Get unique person entity names
        person_entities = list({
            ent.text for ent in doc.ents
            if ent.type in ['PER', 'PERSON']
        })

        # Return first person entity found
        if person_entities:
            entity_str = person_entities[0].strip()
            return entity_str.title() if entity_str else None

        return None

    def _drop_raw_outliers(self, actor_df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows whose news_actors_raw length exceeds the 99th percentile of the dataset."""
        col = 'news_actors_raw'
        if col not in actor_df.columns:
            return actor_df
        lengths = actor_df[col].str.len().fillna(0)
        fence = lengths.quantile(0.99)
        mask = lengths > fence
        if mask.any():
            dropped = actor_df[mask]
            print(f"  Dropping {len(dropped)} row(s) with news_actors_raw length "
                  f"above 99th percentile (>{fence:.0f} chars):")
            for _, r in dropped.iterrows():
                print(f"    {self.id_column}={r.get(self.id_column)}  len={int(lengths[r.name])}")
            actor_df = actor_df[~mask].copy()
        return actor_df

    @property
    def _wikidata_session(self):
        """Lazily created requests.Session shared across all Wikidata calls."""
        if not hasattr(self, '_wd_session'):
            import requests as _requests
            self._wd_session = _requests.Session()
            self._wd_session.headers.update({"User-Agent": self._WIKIDATA_UA})
        return self._wd_session

    def _wikidata_get(self, url: str, params: dict, timeout=(5, 30)) -> "requests.Response":
        """GET with shared session, retrying once on 429/503 per Retry-After."""
        import time
        resp = self._wikidata_session.get(url, params=params, timeout=timeout)
        if resp.status_code in (429, 503):
            retry_after = int(resp.headers.get("Retry-After", 60))
            print(f"\n  Rate-limited by Wikidata ({resp.status_code}), waiting {retry_after}s")
            time.sleep(retry_after)
            resp = self._wikidata_session.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp

    def _query_sparql(self, sparql: str) -> Dict:
        """Execute SPARQL query against Wikidata WDQS using requests directly.

        Timeout is (5s connect, 30s read). WDQS hard server limit is 60s,
        so 30s lets us fail fast on slow queries without losing valid ones.
        """
        resp = self._wikidata_get(
            "https://query.wikidata.org/sparql",
            params={"query": sparql, "format": "json"},
        )
        return resp.json()

    def _search_wikidata(self, name: str, language: str) -> Optional[str]:
        """Search for a person on Wikidata and return their QID."""
        resp = self._wikidata_get(
            "https://www.wikidata.org/w/api.php",
            params={
                "action": "wbsearchentities",
                "search": name,
                "language": language,
                "format": "json",
                "limit": 1,
                "maxlag": 5,
            },
        )
        hits = resp.json().get("search", [])
        return hits[0]["id"] if hits else None

    def get_latest_party_from_wikidata(self, name: str, language: str
                                       ) -> Optional[Dict[str, Optional[str]]]:
        """
        Query Wikidata for a person's latest party affiliation.

        Args:
            name: Person's name
            language: Language code for labels

        Returns:
            Dictionary with 'party_name' and 'party_name_short' keys, or None if not found
        """
        qid = self._search_wikidata(name, language=language)
        if not qid:
            return None

        sparql = f"""
        SELECT ?partyLabel ?shortName ?start ?end WHERE {{
            VALUES ?person {{ wd:{qid} }}
            ?person p:P102 ?stmt .
            ?stmt ps:P102 ?party .
            OPTIONAL {{ ?stmt pq:P580 ?start. }}
            OPTIONAL {{ ?stmt pq:P582 ?end. }}
            OPTIONAL {{ ?party wdt:P1813 ?shortName. }}
            SERVICE wikibase:label {{
                bd:serviceParam wikibase:language "{language},en".
            }}
        }}
        """

        try:
            results = self._query_sparql(sparql)
        except Exception as e:
            print(f"SPARQL query failed for {name}: {e}")
            return None

        records = [{
            "party": r["partyLabel"]["value"],
            "short_name": r.get("shortName", {}).get("value"),
            "start": r.get("start", {}).get("value"),
            "end": r.get("end", {}).get("value"),
        } for r in results["results"]["bindings"]]

        if not records:
            return None

        # Sort by start/end descending (ISO date strings sort lexicographically)
        records.sort(key=lambda x: (x['start'] or '', x['end'] or ''), reverse=True)
        latest = records[0]
        return {
            "party_name": latest["party"],
            "party_name_short": latest["short_name"] or None,
        }

    def fetch_party_info(self, name: str, language: str = "en") -> Dict[str, Optional[str]]:
        """
        Wrapper to safely fetch party information from Wikidata.

        Returns dict with party_name, party_name_short, and _error (True if a
        network/API exception occurred, False for a clean "not found" result).
        Callers should only treat _error=True as a signal for backoff.
        """
        try:
            result = self.get_latest_party_from_wikidata(name, language=language)
            if result:
                return {
                    "party_name": result["party_name"],
                    "party_name_short": result["party_name_short"],
                    "_error": False,
                }
            return {"party_name": None, "party_name_short": None, "_error": False}
        except Exception as e:
            print(f"Error fetching party info for {name}: {e}")
            return {"party_name": None, "party_name_short": None, "_error": True}

    def _normalize_string(self, s: str) -> str:
        """Unicode-safe normalization + uppercase."""
        if not isinstance(s, str):
            return ""
        s = unicodedata.normalize("NFKC", s)
        return s.strip().upper()

    def _tokenize_name(self, name: str, stop_tokens: set) -> list[str]:
        """
        Normalize + split name into meaningful tokens.
        Removes titles but keeps surname particles.
        """
        name = self._normalize_string(name)

        # Replace hyphens with spaces for flexible matching
        name = name.replace("-", " ")

        tokens = name.split()

        # Remove stop tokens
        tokens = [t for t in tokens if t not in stop_tokens]

        return tokens

    def _extract_surname(self, tokens: list[str]) -> str:
        """
        Extract surname including particles.
        Example:
        ['MARK', 'RUTTE'] → RUTTE
        ['JAN', 'VAN', 'DIJK'] → VAN DIJK
        """
        if not tokens:
            return ""

        surname_parts = []
        i = len(tokens) - 1

        # Always include last token
        surname_parts.insert(0, tokens[i])
        i -= 1

        # Include preceding particles
        while i >= 0 and tokens[i] in {"VAN", "DER", "DEN", "DE", "VON", "ZU", "TEN", "TER", "LA", "LE", "DI"}:
            surname_parts.insert(0, tokens[i])
            i -= 1

        return " ".join(surname_parts)

    def _surname_match(self, partial_name: str, full_name: str) -> bool:
        """
        Precision-controlled surname matching for political actors.
        """

        if not partial_name or not full_name:
            return False

                # First we make a list of possible function titles that could precede surnames in ENG, NL, DE, TR
        STOP_TOKENS = {
            # NL
            "KAMERLID", "TWEEDE", "EERSTE", "LID", "STAATSSECRETARIS", "MINISTER",
            "MINISTER-PRESIDENT", "VICEPREMIER", "PREMIER", "FRACTIEVOORZITTER",
            "BURGEMEESTER", "WETHOUDER", "RAADSLID", "SENATOR", "VOORZITTER", "DEMISSIONAIR",

            # DE
            "BUNDESKANZLER", "KANZLER", "MINISTERPRÄSIDENT", "ABGEORDNETER",
            "ABGEORDNETE", "BUNDESTAG", "LANDTAG", "BÜRGERMEISTER",
            "VORSITZENDER", "GESCHÄFTSFÜHREND",

            # EN
            "PRIME", "MINISTER", "SECRETARY", "STATE", "MP", "MEP",
            "MAYOR", "GOVERNOR", "PRESIDENT", "CHAIR", "CHAIRMAN",
            "CHAIRWOMAN", "SPEAKER", "MEMBER",

            # IE
            "TAOISEACH", "TÁNAISTE", "TD",

            # TR
            "BAKAN", "BAKANI", "BAŞBAKAN", "CUMHURBAŞKANI",
            "MILLETVEKILI", "BELEDIYE", "BELEDIYE BAŞKANI"
        }

        partial_tokens = self._tokenize_name(partial_name, STOP_TOKENS)
        full_tokens = self._tokenize_name(full_name, STOP_TOKENS)

        if not partial_tokens or not full_tokens:
            return False

        partial_surname = self._extract_surname(partial_tokens)
        full_surname = self._extract_surname(full_tokens)

        # --- Step 1: surnames must match exactly ---
        if partial_surname != full_surname:
            return False

        # --- Step 2: If only surname provided ---
        if len(partial_tokens) == 1:
            # avoid false positives like "LI", "NG"
            if len(partial_surname) < 4:
                return False
            return True

        # --- Step 3: If firstname present, require at least one match ---
        partial_firstnames = set(partial_tokens[:-1])
        full_firstnames = set(full_tokens[:-1])

        if partial_firstnames & full_firstnames:
            return True

        return False


    def enrich_political_actors(self,
                                actor_df: Optional[pd.DataFrame] = None,
                                use_wikidata: bool = True,
                                language: str = "en") -> pd.DataFrame:
        """
        Enrichment pipeline for politicians (function 'a', NER person):
        1. Check for party name mentions in actor names (fast, direct signal)
        2. Extract core names using NER for unmatched rows
        3. Match against party reference data (exact → token → original name)
        4. Query Wikidata for still-missing information (if use_wikidata=True)
        5. Merge with ideology scores

        Args:
            actor_df: Optional actor-level DataFrame. If None, expands self.actor_df
            use_wikidata: Whether to query Wikidata for missing party info
            language: Language code for Wikidata queries
        Returns:
            DataFrame with enriched actor information including party affiliations
        """
        if actor_df is None:
            actor_df = self.expand_actors_to_rows()

        if actor_df.empty:
            print("Actor DataFrame is empty. Returning empty DataFrame.")
            return pd.DataFrame()

        # Filter for political actors only (function 'a'), work as list of dicts
        political_actors: List[Dict] = actor_df[actor_df['actor_function'] == 'a'].to_dict('records')

        if not political_actors:
            print("No political actors (function 'a') found.")
            return pd.DataFrame()

        print(f"Processing {len(political_actors)} political actor records...")

        # Normalize actor names and initialize all extra columns on every row
        for row in political_actors:
            row['actor_name_upper'] = self._normalize_string(row['actor_name'])
            row['party'] = None
            row['party_final'] = None
            row['in_reference'] = False
            row['matched_name'] = None
            row['core_actor_name'] = None
            row['core_actor_name_upper'] = None
            for col in self.score_cols:
                row[col] = None

        if self.politician_reference_df is not None:
            for ref_row in self.politician_reference_df:
                ref_row['name'] = self._normalize_string(ref_row['name'])

        # Step 3.1: Check for party name mentions in actor_name (before NER)
        print("Step 3.1: Checking for party name mentions in actor names...")
        party_mention_count = 0
        extra_rows: List[Dict] = []

        if self.ideology_reference_df is not None:
            for row in political_actors:
                actor_name = re.sub(r"[-–—/]", " ", row['actor_name_upper'])
                actor_name = re.sub(r"\s+", " ", actor_name).strip()

                matched_parties = []

                for ref_row in self.ideology_reference_df:
                    party_short = str(ref_row['party']).strip()
                    # Build list of name variants to check: short name + full name if present
                    variants = [party_short]
                    full_name = ref_row.get('party_name')
                    if full_name and isinstance(full_name, str) and full_name.strip():
                        variants.append(full_name.strip())

                    matched_variant = None
                    for variant in variants:
                        v_upper = self._normalize_string(variant)
                        v_upper = re.sub(r"[-–—/]", " ", v_upper)
                        v_upper = re.sub(r"\s+", " ", v_upper).strip()
                        if re.search(rf"(?<!\w){re.escape(v_upper)}(?!\w)", actor_name):
                            matched_variant = variant
                            break

                    if matched_variant:
                        match_entry = {
                            'party': party_short,
                            'matched_name': matched_variant,
                        }
                        for col in self.score_cols:
                            match_entry[col] = ref_row.get(col)
                        matched_parties.append(match_entry)

                # For each matched party, create or update rows
                if matched_parties:
                    party_mention_count += 1
                    row['party'] = matched_parties[0]['party']
                    row['matched_name'] = matched_parties[0]['matched_name']
                    row['in_reference'] = True
                    for col in self.score_cols:
                        row[col] = matched_parties[0].get(col)
                    for match in matched_parties[1:]:
                        new_row = dict(row)  # in_reference=True already copied
                        new_row['party'] = match['party']
                        new_row['matched_name'] = match['matched_name']
                        for col in self.score_cols:
                            new_row[col] = match.get(col)
                        extra_rows.append(new_row)

            # Append new rows for additional matches
            political_actors.extend(extra_rows)

        print(f". → Matched {party_mention_count} actors with party name mentions")

        unmatched = [row for row in political_actors if row['party'] is None]
        print(f"Step 3.2: Extracting core names using NER for {len(unmatched)} actors with no party mentions...")

        for row in tqdm(unmatched, desc="Extracting names"):
            row['core_actor_name'] = self.extract_core_name(row['actor_name'])

        for row in political_actors:
            cn = row['core_actor_name']
            row['core_actor_name_upper'] = self._normalize_string(cn) if cn is not None else None

        if self.politician_reference_df is not None:
            print("Step 3.3: Exact matching on core_actor_name...")
            step1_matched = sum(1 for row in political_actors if row['party'] is not None)

            # Build lookup dict for O(1) exact matching
            exact_lookup = {ref['name']: ref for ref in self.politician_reference_df}

            for row in political_actors:
                if row['party'] is None and row['core_actor_name_upper']:
                    match = exact_lookup.get(row['core_actor_name_upper'])
                    if match:
                        row['party'] = match['party']
                        row['matched_name'] = match['name']
                        row['in_reference'] = True
                        for col in self.score_cols:
                            row[col] = match.get(col)

            exact_count = sum(1 for row in political_actors if row['party'] is not None) - step1_matched
            print(f"  → Matched {exact_count} actors with exact core_actor_name match")
            seen: set = set()
            unique_matched = []
            for r in political_actors:
                if r['party'] is not None and r['core_actor_name'] not in seen:
                    seen.add(r['core_actor_name'])
                    unique_matched.append({'core_actor_name': r['core_actor_name'], 'matched_name': r['matched_name']})

        # Step 3.4: Token match on core_actor_name (for unmatched rows)
        if self.politician_reference_df is not None:
            print("Step 3.4: Token matching on core_actor_name...")
            token_match_count = 0

            for row in political_actors:
                if row['party'] is None:
                    for ref_row in self.politician_reference_df:
                        if self._surname_match(row['core_actor_name_upper'], ref_row['name']):
                            row['party'] = ref_row['party']
                            row['matched_name'] = ref_row['name']
                            row['in_reference'] = True
                            for col in self.score_cols:
                                row[col] = ref_row.get(col)
                            token_match_count += 1
                            break

            print(f"→ Matched {token_match_count} actors with token match on core_actor_name")
            seen2: set = set()
            unique_matched2 = []
            for r in political_actors:
                if r['party'] is not None and r['core_actor_name'] not in seen2:
                    seen2.add(r['core_actor_name'])
                    core = (r['core_actor_name'] or '').upper()
                    matched = (r['matched_name'] or '').upper()
                    if core != matched:
                        unique_matched2.append({'core_actor_name': r['core_actor_name'], 'matched_name': r['matched_name']})
            if unique_matched2:
                print(f"Non-obvious matches with reference data: {unique_matched2}")

            # Step 3.5: Exact + token match on actor_name (for still unmatched rows)
            print("Step 3.5: Matching on original actor_name...")
            actor_name_count = 0

            for row in political_actors:
                if row['party'] is None:
                    actor_name = row['actor_name_upper']
                    match = exact_lookup.get(actor_name)
                    if match:
                        row['party'] = match['party']
                        row['matched_name'] = match['name']
                        row['in_reference'] = True
                        for col in self.score_cols:
                            row[col] = match.get(col)
                        actor_name_count += 1
                        continue

                    for ref_row in self.politician_reference_df:
                        if self._surname_match(actor_name, ref_row['name']):
                            row['party'] = ref_row['party']
                            row['matched_name'] = ref_row['name']
                            row['in_reference'] = True
                            for col in self.score_cols:
                                row[col] = ref_row.get(col)
                            actor_name_count += 1
                            break

            print(f"→ Matched {actor_name_count} actors with actor_name matching")

            total_matched = sum(1 for row in political_actors if row['party'] is not None)
            print(f"\nTotal matched after name matching: {total_matched} actors")

        total_matched = sum(1 for row in political_actors if row['party'] is not None)
        print(f"\nTotal matched: {total_matched} actors with reference data")

        # Step 3.7: Query Wikidata for missing information
        if use_wikidata:
            missing = [row for row in political_actors if row['party'] is None]

            if missing:
                print(f"Step 3.7: Querying Wikidata for {len(missing)} actors with missing party info...")

                unique_names = list({row['core_actor_name'] for row in missing if row['core_actor_name'] is not None})
                print(f"Querying {len(unique_names)} unique names...")

                wikidata_results = {}
                import time
                consecutive_failures = 0
                max_consecutive_failures = 5

                for name in tqdm(unique_names, desc="Wikidata queries"):
                    if consecutive_failures >= max_consecutive_failures:
                        print(f"\n  Wikidata aborted after {max_consecutive_failures} consecutive failures "
                              f"(likely rate-limited or in outage). {len(wikidata_results)} names queried.")
                        break

                    result = self.fetch_party_info(name, language=language)
                    wikidata_results[name] = result

                    if result.get('_error'):
                        # Real network/API error — back off
                        consecutive_failures += 1
                        backoff = min(60, 3 * (2 ** consecutive_failures)) + random.uniform(0, 2)
                        print(f"\n  API error for '{name}' (failure {consecutive_failures}/{max_consecutive_failures}), backing off {backoff:.1f}s")
                        time.sleep(backoff)
                    elif result.get('party_name') is not None:
                        # Found — be polite but don't over-sleep
                        consecutive_failures = 0
                        time.sleep(random.uniform(1, 2))
                    else:
                        # Clean "not found" — short pause, no backoff
                        consecutive_failures = 0
                        time.sleep(0.5)

                # Apply results to rows
                for row in political_actors:
                    if row['party'] is None:
                        name = row['core_actor_name']
                        if name in wikidata_results:
                            result = wikidata_results[name]
                            if result.get('party_name_short') is not None:
                                row['party'] = result['party_name_short']
                            for col in self.score_cols:
                                if result.get(col) is not None:
                                    row[col] = result[col]


        # Resolve party_final and lrgen_cat from the domestic party reference
        if self.ideology_reference_df is not None:
            party_lookup = {
                self._normalize_string(r['party']): r
                for r in self.ideology_reference_df
            }
            for row in political_actors:
                if row.get('party') is not None:
                    ref = party_lookup.get(self._normalize_string(row['party']))
                    if ref:
                        row['party_final'] = ref.get('party_final')
                        row['lrgen_cat'] = ref.get('lrgen_cat')

        matched = sum(1 for r in political_actors if r.get('party') is not None)
        unmatched = len(political_actors) - matched
        if unmatched:
            print(f"  {unmatched} political actors left unmatched (party=None, kept in output for inspection)")

        return pd.DataFrame(political_actors) if political_actors else pd.DataFrame()

    def calculate_actors_per_partyideology(self, political_actors_df: pd.DataFrame, center_parties: bool) -> pd.DataFrame:
        """
        Calculate number of unique actors per party ideology category for each article.

        Args:
            political_actors_df: DataFrame with enriched political actors (must include id_column and lrgen_cat)

        Returns:
            DataFrame with columns: id_column, nr_actors_left, nr_actors_right, nr_actors_center,
            nr_actors_political
        """

        if political_actors_df.empty:
            print("Political actors DataFrame is empty. Returning empty DataFrame.")
            return pd.DataFrame()

        # Keep only valid ideology categories
        valid_categories = ['left', 'right']
        if center_parties:
            valid_categories.append('center')

        political_actors_df = political_actors_df[political_actors_df['lrgen_cat'].isin(valid_categories)]

        if political_actors_df.empty:
            print("No actors with valid ideology categories found.")
            return pd.DataFrame()

        # Count unique actor names per (id, ideology category)
        unique_counts = (
            political_actors_df
            .groupby([self.id_column, 'lrgen_cat'])['actor_name']
            .nunique()
            .reset_index()
            .rename(columns={'actor_name': 'nr_actors'})
        )

        # Pivot to have ideology categories as columns
        ideology_df = (
            unique_counts
            .pivot(index=self.id_column, columns='lrgen_cat', values='nr_actors')
            .fillna(0)
            .reset_index()
        )
        ideology_df.columns.name = None

        # if one of the ideology columns is missing, add it with zeros
        for category in valid_categories:
            if category not in ideology_df.columns:
                ideology_df[category] = 0

        # Rename columns
        rename_map = {cat: f'nr_actors_{cat}' for cat in valid_categories if cat in ideology_df.columns}
        ideology_df = ideology_df.rename(columns=rename_map)

        # Calculate total number of unique political actors
        actor_cols = [c for c in ideology_df.columns if c.startswith('nr_actors_')]
        ideology_df['nr_actors_political'] = ideology_df[actor_cols].sum(axis=1)

        return ideology_df


    def calculate_actors_per_party(self, political_actors_df: pd.DataFrame) -> pd.DataFrame:
        """
        Per-article count of unique actors per canonical party name (party_final).

        Returns a long-format DataFrame:
            id_column | party_final | lrgen_cat | nr_actors

        Rows where party_final is null are excluded — use enriched_actors_df to
        inspect unmatched actors. This output is intended for analysis-time
        aggregation (e.g. groupby party_final or lrgen_cat per article).
        """
        df = political_actors_df[political_actors_df['party_final'].notna()].copy()
        if df.empty:
            return pd.DataFrame(columns=[self.id_column, 'party_final', 'lrgen_cat', 'nr_actors'])

        result = (
            df.groupby([self.id_column, 'party_final', 'lrgen_cat'], dropna=False)['actor_name']
            .nunique()
            .reset_index()
            .rename(columns={'actor_name': 'nr_actors'})
        )
        return result

    def run_full_enrichment(
        self,
        use_wikidata: bool = True,
        language: Optional[str] = None,
        actor_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Run the complete enrichment pipeline.

        Args:
            use_wikidata: Whether to query Wikidata for party information
            language: Language code for Wikidata queries (defaults to the language set at init)

        Returns:
            DataFrame with columns: id_column, nr_actors_a, nr_actors_b, nr_actors_c,
            nr_actors_d, nr_actors_total, nr_actors_left, nr_actors_right,
            nr_actors_center (if center_parties=True), nr_actors_political
        """
        language = language or self.language
        print("\n" + "="*60)
        print("STARTING FULL ACTOR ENRICHMENT PIPELINE")
        print("="*60 + "\n")

        # Step 0: Deduplicate on article_id before any expansion
        if actor_df is None:
            actor_df = self.actor_df
        before = len(actor_df)
        actor_df = actor_df.drop_duplicates(subset=[self.id_column], keep='last')
        dupes = before - len(actor_df)
        if dupes:
            print(f"Step 0: Removed {dupes} duplicate article rows (kept last occurrence — batch file takes precedence)\n")

        actor_df = self._drop_raw_outliers(actor_df)

        # Step 1: Expand actors to rows
        print("Step 1: Expanding actors to row-level...")
        expanded_df = self.expand_actors_to_rows(actor_df)
        before_actor_dedup = len(expanded_df)
        expanded_df = expanded_df.drop_duplicates(
            subset=[self.id_column, 'actor_name', 'actor_function'], keep='first'
        )
        actor_dupes = before_actor_dedup - len(expanded_df)
        if actor_dupes:
            print(f"  Removed {actor_dupes} duplicate (article, actor, function) rows")
        print(f"Expanded to {len(expanded_df)} actor records\n")

        # Step 2: Calculate actor statistics per function
        print("Step 2: Calculating actor statistics per function...")
        functions_df = self.calculate_actors_per_function(expanded_df)
        print(f"Generated statistics for {len(functions_df)} articles\n")

        # Step 3: Enrich political actors
        print("Step 3: Enriching political actors...")
        enriched_political_df = self.enrich_political_actors(
            expanded_df,
            use_wikidata=use_wikidata,
            language=language
        )
        self.enriched_actors_df = enriched_political_df
        print(f"Enriched {len(enriched_political_df)} political actor records\n")

        # Step 4: Calculate actors per party ideology (only if lrgen_cat was in reference data)
        ideology_df = pd.DataFrame()
        if not enriched_political_df.empty and 'lrgen_cat' in enriched_political_df.columns:
            print("Step 4: Calculating actors per party ideology...")
            ideology_df = self.calculate_actors_per_partyideology(
                enriched_political_df,
                center_parties=self.center_parties
            )
            print(f"Generated ideology statistics for {len(ideology_df)} articles\n")
        else:
            print("Step 4: Skipping ideology counts (no lrgen_cat in reference data)\n")

        # Step 5: Calculate actors per party_final (long format, for analysis-time aggregation)
        self.actors_per_party_df = pd.DataFrame()
        if not enriched_political_df.empty and 'party_final' in enriched_political_df.columns:
            print("Step 5: Calculating actors per party_final...")
            self.actors_per_party_df = self.calculate_actors_per_party(enriched_political_df)
            n_parties = self.actors_per_party_df['party_final'].nunique() if not self.actors_per_party_df.empty else 0
            print(f"Generated party counts for {len(self.actors_per_party_df)} article×party rows ({n_parties} unique parties)\n")
        else:
            print("Step 5: Skipping party_final counts (no party_final column in reference data)\n")

        print("="*60)
        print("ENRICHMENT PIPELINE COMPLETED")
        print("="*60 + "\n")

        if functions_df.empty:
            return ideology_df
        if ideology_df.empty:
            return functions_df
        return functions_df.merge(ideology_df, on=self.id_column, how='left')


def main(args):
    """Main execution function for command-line usage."""

    # Initialize enricher
    enricher = ActorEnricher(
        actor_data_path=args.actor_data_path,
        id_column=args.id_column,
        language=args.language,
        politicians_data_path=args.politicians_data_path,
    )

    # Run full enrichment pipeline
    actor_stats_df = enricher.run_full_enrichment(
        use_wikidata=args.use_wikidata,
        language=args.language
    )

    # Save outputs
    output_dir = os.path.dirname(args.output_prefix) or '.'
    os.makedirs(output_dir, exist_ok=True)

    # Save article-level actor statistics (counts per function / ideology)
    stats_path = f"{args.output_prefix}_actor_stats.csv"
    actor_stats_df.to_csv(stats_path, sep=';', index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"Saved actor statistics to: {stats_path}")

    # Save actor-level enriched data (one row per actor with party + score columns)
    if hasattr(enricher, 'enriched_actors_df') and not enricher.enriched_actors_df.empty:
        actors_path = f"{args.output_prefix}_actors_enriched.csv"
        enricher.enriched_actors_df.to_csv(actors_path, sep=';', index=False, quoting=csv.QUOTE_NONNUMERIC)
        print(f"Saved enriched actor data to: {actors_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enrich actors from LLM-annotated DataFrame"
    )

    # Input/output arguments
    parser.add_argument(
        "--actor_data_path",
        type=str,
        required=True,
        help="Path to input CSV file with LLM actor annotations"
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="actors_output",
        help="Prefix for output CSV files (will create _expanded.csv, _functions.csv, _political.csv, _ideology.csv)"
    )
    parser.add_argument(
        "--id_column",
        type=str,
        default="uri",
        help="Column name for unique article identifier (default: 'uri')"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language for NER processing (default: 'en'). Use 'nl' for Dutch."
    )
    parser.add_argument(
        "--use_wikidata",
        action="store_true",
        help="Query Wikidata for missing party information"
    )
    parser.add_argument(
        "--politicians_data_path",
        type=str,
        help="Path to CSV with party reference data (columns: name, party, lrgen_cat)"
    )

    args = parser.parse_args()
    main(args)
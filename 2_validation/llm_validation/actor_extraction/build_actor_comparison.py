"""
Build human-vs-gpt_oss actor comparison tables, one Excel per LANGUAGE.

Language -> countries (article_id prefix) -> coder
    English  : UK (1x) + IR (2x)   coder: coder_dutch
    Dutch    : NL (3x) + BE (4x)   coder: coder_dutch
    Turkish  : TR (5x)             coder: coder_turkish

Human source files:
    Actors_annotation/uk_actors_annotated_test.csv   -> UK  (already long format)
    Actors_annotation/human_annotations_NLTRBEIR.csv -> IR, NL, BE, TR (Qualtrics wide)
NOTE: human_annotations_ENG.csv is the UK *first-round / reliability* set and does
NOT overlap the GPT test articles, so it is NOT used here. The UK test-set gold is
uk_actors_annotated_test.csv.
GPT source (current run, long format, one row per actor):
    actor_annotations_percountry/<cc>_annotated_test_gptoss20b.csv

For every article annotated by BOTH sides we flag, per actor:
    missedbygpt  -> human actor with no matching gpt actor   (false negative, FN)
    coded extra  -> gpt actor with no matching human actor   (false positive, FP)
    matched on both sides                                    (true positive,  TP)
=> precision = TP/(TP+FP), recall = TP/(TP+FN), F1 = harmonic mean.

Matching is on actor_name within an article (function/pp may differ), the same
basis as the manual NL gold. Because gpt prepends titles/roles
("demissionair premier Mark Rutte") and human names carry descriptors, matching
uses normalised token overlap (subset or Jaccard >= 0.5), greedy one-to-one.
The flags are AUTO-SUGGESTED and meant to be verified by hand; the summary
precision/recall/F1 are therefore PRELIMINARY until that review is done.
"""
import os
import re
import csv
import unicodedata
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
GPT_DIR = f"{BASE}/actor_annotations_percountry"
HUMAN_UK = f"{BASE}/Actors_annotation/uk_actors_annotated_test.csv"
HUMAN_NLTR = f"{BASE}/Actors_annotation/human_annotations_NLTRBEIR.csv"
OUT_DIR = f"{BASE}/Manual_Checks"

FUNCTION_MAP = {
    'Government and politics (includes international, national, regional politicians, government/state officials, members of law enforcement)': 'a',
    'Professionals and experts (expets outside government/state organizations, media and business representatives, researchers, academics, celebrities)': 'b',
    'Civil society organizations (non-governmental organizations, non-profit organizations, trade unions, social movements, interest groups)': 'c',
    'Citizens/members of the public (ordinary citizens, residents, people speaking from personal experience rather than professional expertise such as residents affected by policies, immigrants sharing personal stories, eyewitnesses to events, community members expressing individual opinions; this category also includes collective public opinion if quoted/paraphrased)': 'd',
}

_ARTICLES = {'de', 'het', 'een', 'the', 'a', 'an', 'le', 'la', 'les', 'der', 'die', 'das'}
_CODE_FOR = {'1': 'uk', '2': 'ir', '3': 'nl', '4': 'be', '5': 'tr'}

# Each source is either an already-long CSV ('long') or a Qualtrics wide export to
# melt ('qualtrics', needs a coder name). English pools UK + Ireland.
LANGUAGES = {
    'english': [
        {'kind': 'long', 'path': HUMAN_UK, 'prefixes': ['11', '12', '13', '14']},
        {'kind': 'qualtrics', 'path': HUMAN_NLTR, 'coder': 'coder_dutch',
         'prefixes': ['21', '22', '23', '24']},
    ],
    'dutch': [
        {'kind': 'qualtrics', 'path': HUMAN_NLTR, 'coder': 'coder_dutch',
         'prefixes': ['31', '32', '33', '34', '41', '42', '43', '44']},
    ],
    'turkish': [
        {'kind': 'qualtrics', 'path': HUMAN_NLTR, 'coder': 'coder_turkish',
         'prefixes': ['51', '52', '53', '54']},
    ],
}


# --------------------------------------------------------------------------- #
# normalisation + matching
# --------------------------------------------------------------------------- #
def _tokens(name):
    if pd.isna(name):
        return []
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.casefold()
    s = re.sub(r"\(.*?\)", " ", s)        # drop parentheticals e.g. "(Asiel, VVD)"
    s = re.sub(r"[^\w\s]", " ", s)
    return [t for t in s.split() if t and t not in _ARTICLES]


def _is_match(h_tokens, g_tokens):
    hs, gs = set(h_tokens), set(g_tokens)
    if not hs or not gs:
        return False
    if hs <= gs or gs <= hs:               # one name contained in the other
        return True
    return len(hs & gs) / len(hs | gs) >= 0.5


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def _read_qualtrics(path):
    return pd.read_csv(path, quoting=csv.QUOTE_ALL, dtype=str)


def melt_human(path, coder_name, prefixes):
    """One Qualtrics wide export -> long actor rows for one coder + country set."""
    df = _read_qualtrics(path)
    df = df[df['M2. ArticleID'].astype(str).str.match(r'^\d{8}$', na=False)]
    df = df[df['M1. Coder'] == coder_name]
    df = df[df['M2. ArticleID'].astype(str).str[:2].isin(prefixes)]
    if df.empty:
        return pd.DataFrame(columns=['article_id', 'actor_name', 'actor_function', 'actor_pp']), []
    df = df.rename(columns={'M1. Coder': 'coder', 'M2. ArticleID': 'article_id',
                            'M3. ArticleTitle': 'article_title', 'Filter1': 'has_actor'})
    no_actor = df.loc[df.get('has_actor') == 'No', 'article_id'].astype(int).unique().tolist()
    keep = ['coder', 'article_id', 'article_title', 'has_actor']
    keep = [c for c in keep if c in df.columns]
    actor_cols = [c for c in df.columns if re.match(r'^\d+_V[345]\.', c)]
    long = df[keep + actor_cols].melt(id_vars=keep, var_name='variable', value_name='value')
    long['actor_number'] = long['variable'].str.extract(r'(\d+)_V')[0].astype(int)
    long['actor_field'] = long['variable'].str.extract(r'V\d+\.(.*)')[0]
    wide = long.pivot_table(index=keep + ['actor_number'], columns='actor_field',
                            values='value', aggfunc='first').reset_index()
    wide.columns = [c.strip().replace(' ', '_').lower() for c in wide.columns]
    wide = wide.rename(columns={'actorname': 'actor_name', 'actorfunction': 'actor_function',
                                'actorpp': 'actor_pp'})
    wide['actor_function'] = wide['actor_function'].str.strip().map(FUNCTION_MAP)
    wide['article_id'] = wide['article_id'].astype(int)
    wide = wide[wide['actor_name'].notna() & (wide['actor_name'].str.strip() != '')]
    return wide[['article_id', 'actor_name', 'actor_function', 'actor_pp']], no_actor


def load_long(path, prefixes):
    """Read an already-long human file (article_id, actor_name, actor_function, actor_pp)."""
    try:
        df = pd.read_csv(path, sep=';', encoding='utf-8-sig')
        if df.shape[1] == 1:
            raise ValueError
    except Exception:
        df = pd.read_csv(path, encoding='utf-8-sig')
    df['article_id'] = df['article_id'].astype(int)
    df = df[df['article_id'].astype(str).str[:2].isin(prefixes)]
    df = df[df['actor_name'].notna() & (df['actor_name'].astype(str).str.strip() != '')]
    return df[['article_id', 'actor_name', 'actor_function', 'actor_pp']], []


def load_gpt(prefixes):
    countries = sorted({_CODE_FOR[p[0]] for p in prefixes})
    frames = []
    for c in countries:
        p = f"{GPT_DIR}/{c}_annotated_test_gptoss20b.csv"
        try:
            g = pd.read_csv(p, sep=';', quoting=csv.QUOTE_NONNUMERIC, encoding='utf-8-sig')
        except Exception:
            g = pd.read_csv(p, encoding='utf-8-sig')
        frames.append(g[['article_id', 'actor_name', 'actor_function', 'actor_pp']])
    gpt = pd.concat(frames, ignore_index=True)
    gpt['article_id'] = gpt['article_id'].astype(int)
    no_actor = gpt.loc[gpt['actor_name'].isna(), 'article_id'].unique().tolist()
    gpt = gpt[gpt['actor_name'].notna() & (gpt['actor_name'].astype(str).str.strip() != '')]
    return gpt, no_actor


# --------------------------------------------------------------------------- #
# comparison
# --------------------------------------------------------------------------- #
def compare(human, gpt, human_no_actor=(), gpt_no_actor=()):
    """Interleaved human/gpt rows for shared articles, with auto match flags.

    The shared universe is every article BOTH sides processed, including articles
    where one side recorded no actors: those become all-missed (FN) or all-extra
    (FP), so precision/recall are not biased by dropping the hardest cases.
    """
    human_articles = set(human['article_id']) | set(human_no_actor)
    gpt_articles = set(gpt['article_id']) | set(gpt_no_actor)
    shared = sorted(human_articles & gpt_articles)
    rows = []
    for aid in shared:
        h = human[human['article_id'] == aid].reset_index(drop=True)
        g = gpt[gpt['article_id'] == aid].reset_index(drop=True)
        htok = [_tokens(x) for x in h['actor_name']]
        gtok = [_tokens(x) for x in g['actor_name']]
        g_match_for = {}      # human idx -> gpt idx
        h_match_for = {}      # gpt idx -> human idx
        used_g = set()
        for i in range(len(h)):
            for j in range(len(g)):
                if j in used_g:
                    continue
                if _is_match(htok[i], gtok[j]):
                    g_match_for[i] = j
                    h_match_for[j] = i
                    used_g.add(j)
                    break
        for i, r in h.iterrows():
            j = g_match_for.get(i)
            rows.append({'coder': 'human', 'article_id': aid,
                         'actor_function': r['actor_function'], 'actor_name': r['actor_name'],
                         'actor_pp': r['actor_pp'],
                         'missedbygpt': 0 if j is not None else 1, 'coded extra': '',
                         'matched_with': g.loc[j, 'actor_name'] if j is not None else ''})
        for j, r in g.iterrows():
            i = h_match_for.get(j)
            rows.append({'coder': 'gpt_oss', 'article_id': aid,
                         'actor_function': r['actor_function'], 'actor_name': r['actor_name'],
                         'actor_pp': r['actor_pp'],
                         'missedbygpt': '', 'coded extra': 0 if i is not None else 1,
                         'matched_with': h.loc[i, 'actor_name'] if i is not None else ''})
    comp = pd.DataFrame(rows)
    # human rows before gpt rows within each article (matches the NL gold layout)
    comp['_order'] = comp['coder'].map({'human': 0, 'gpt_oss': 1})
    comp = comp.sort_values(['article_id', '_order']).drop(columns='_order').reset_index(drop=True)
    return comp, shared


def summarize(comp, n_articles):
    h = comp[comp['coder'] == 'human']
    g = comp[comp['coder'] == 'gpt_oss']
    tp = int((h['missedbygpt'] == 0).sum())
    fn = int((h['missedbygpt'] == 1).sum())
    fp = int((g['coded extra'] == 1).sum())
    prec = tp / (tp + fp) if (tp + fp) else float('nan')
    rec = tp / (tp + fn) if (tp + fn) else float('nan')
    f1 = 2 * prec * rec / (prec + rec) if prec and rec and (prec + rec) else float('nan')
    return pd.DataFrame([
        ('shared articles (coded by both)', n_articles),
        ('human actors total (TP+FN)', tp + fn),
        ('gpt actors total (TP+FP)', tp + fp),
        ('matched / true positives (TP)', tp),
        ('missed by gpt / false negatives (FN)', fn),
        ('coded extra by gpt / false positives (FP)', fp),
        ('precision = TP/(TP+FP)  [PRELIMINARY]', round(prec, 3)),
        ('recall = TP/(TP+FN)  [PRELIMINARY]', round(rec, 3)),
        ('F1  [PRELIMINARY]', round(f1, 3)),
    ], columns=['metric', 'value'])


def build_language(lang):
    h_frames, h_no_actor, prefixes = [], [], []
    for src in LANGUAGES[lang]:
        if src['kind'] == 'long':
            hf, hna = load_long(src['path'], src['prefixes'])
        else:
            hf, hna = melt_human(src['path'], src['coder'], src['prefixes'])
        h_frames.append(hf)
        h_no_actor += hna
        prefixes += src['prefixes']
    human = pd.concat(h_frames, ignore_index=True)
    gpt, g_no_actor = load_gpt(prefixes)
    comp, shared = compare(human, gpt, h_no_actor, g_no_actor)
    summary = summarize(comp, len(shared))
    return comp, summary


def _write_if_absent(path, comp, summary):
    """Write the comparison Excel ONLY if it does not already exist.

    Once manual checking has started these files are hand-edited, so this never
    overwrites them. Delete a file by hand if you want it regenerated from scratch.
    """
    name = path.split('/')[-1]
    if os.path.exists(path):
        print(f"SKIP (exists, manual edits protected): {name}")
        return
    with pd.ExcelWriter(path, engine='openpyxl') as xl:
        comp.to_excel(xl, sheet_name='comparison', index=False)
        summary.to_excel(xl, sheet_name='summary', index=False)
    print(f"wrote {name}")


def main():
    for lang in LANGUAGES:
        comp, summary = build_language(lang)
        if lang == 'english':
            # split into per-country files (UK + Ireland), checked separately
            for cc, pfx in [('UK', '1'), ('IE', '2')]:
                sub = comp[comp['article_id'].astype(str).str[0] == pfx].reset_index(drop=True)
                _write_if_absent(f"{OUT_DIR}/human_gpt_comparison_actors_english_{cc}.xlsx",
                                 sub, summarize(sub, sub['article_id'].nunique()))
        else:
            _write_if_absent(f"{OUT_DIR}/human_gpt_comparison_actors_{lang}.xlsx", comp, summary)


if __name__ == '__main__':
    main()

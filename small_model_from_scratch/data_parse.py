import pandas as pd
import os
import re

# --- CONFIGURATION ----------------------------------------------------------
base_path  = "/Users/aweise/Documents/OMSCS/2025_Spring/DL_7643/group_assignment/data/data/all_data_en"
output_dir = os.path.join(base_path, "overlap_output")
os.makedirs(output_dir, exist_ok=True)

splits = ['train', 'dev', 'test']
read_opts = dict(
    sep='\t',
    quoting=3,
    engine='python',
    na_values=['NA'],
    keep_default_na=True,
    dtype={'lang_conf': 'float64'}
)

def path_for(task, split):
    return os.path.join(
        base_path,
        f"crisis_consolidated_{task}_filtered_lang_en_{split}.tsv"
    )

# --- PATTERNS & FUNCTIONS FOR TEXT UNIFICATION ------------------------------
mojibake_pattern = re.compile(r'â[€œ€™–]')

def count_mojibake(s: str) -> int:
    return len(mojibake_pattern.findall(s)) if isinstance(s, str) else 0

def unify_text(row) -> str:
    h = row['humanitarian_text']
    i = row['informativeness_text']
    # strip whitespace & wrapping quotes
    h_c = h.strip().strip('"').strip("'")
    i_c = i.strip().strip('"').strip("'")
    if h_c == i_c:
        return h_c
    # pick the version with fewer mojibake artifacts
    if count_mojibake(h) < count_mojibake(i):
        chosen = h
    elif count_mojibake(i) < count_mojibake(h):
        chosen = i
    else:
        chosen = h
    return chosen.strip().strip('"').strip("'")

# --- 1) Load & concatenate all splits for each task -------------------------

# Humanitarian dataset
h_dfs = []
for sp in splits:
    df = pd.read_csv(path_for('humanitarian', sp), **read_opts)
    df = df.rename(columns={
        'event':        'humanitarian_event',
        'source':       'humanitarian_source',
        'text':         'humanitarian_text',
        'lang':         'humanitarian_lang',
        'lang_conf':    'humanitarian_lang_conf',
        'class_label':  'humanitarian_label'
    })
    df['orig_split'] = sp
    h_dfs.append(df)
h_all = pd.concat(h_dfs, ignore_index=True).drop_duplicates(subset='id')

# Informativeness dataset
i_dfs = []
for sp in splits:
    df = pd.read_csv(path_for('informativeness', sp), **read_opts)
    df = df.rename(columns={
        'event':        'informativeness_event',
        'source':       'informativeness_source',
        'text':         'informativeness_text',
        'lang':         'informativeness_lang',
        'lang_conf':    'informativeness_lang_conf',
        'class_label':  'informativeness_label'
    })
    df['orig_split'] = sp
    i_dfs.append(df)
i_all = pd.concat(i_dfs, ignore_index=True).drop_duplicates(subset='id')

# --- 2) Compute intersection of IDs -----------------------------------------
h_ids = set(h_all['id'])
i_ids = set(i_all['id'])
common_ids = h_ids & i_ids
print(f"Found {len(common_ids):,} IDs in both datasets.")

# --- 3) Subset each DataFrame to the common IDs -----------------------------
h_sub = h_all[h_all['id'].isin(common_ids)].copy()
i_sub = i_all[i_all['id'].isin(common_ids)].copy()

# --- 4) Merge to bring everything together ---------------------------------
merged = pd.merge(
    h_sub,
    i_sub.drop(columns=['orig_split']),
    on='id',
    how='inner'
)

# --- 5) Identify label mismatches and drop them -----------------------------
bad_not_inf = (
    (merged['informativeness_label'] == 'not_informative') &
    (merged['humanitarian_label'] != 'not_humanitarian')
)
bad_inf = (
    (merged['informativeness_label'] == 'informative') &
    (merged['humanitarian_label'] == 'not_humanitarian')
)
mismatch = merged[bad_not_inf | bad_inf]
if not mismatch.empty:
    print(f"⚠️  Dropping {len(mismatch)} rows with label mismatches:")
    print("Sample IDs:", mismatch['id'].tolist()[:10])
    merged = merged[~(bad_not_inf | bad_inf)].reset_index(drop=True)
else:
    print("✅ All labels align correctly.")

# --- 6) Unify tweet text and drop split‐specific columns --------------------
merged['text'] = merged.apply(unify_text, axis=1)

# Now drop the old text columns and orig_split
merged = merged.drop(columns=[
    'humanitarian_text',
    'informativeness_text',
    'orig_split'
])

# --- 7) Reorder columns ------------------------------------------------------
# Always: id, text, then all remaining columns except the two labels,
# and finally informativeness_label and humanitarian_label
labels = ['informativeness_label', 'humanitarian_label']
others = [c for c in merged.columns if c not in ['id','text'] + labels]
desired_order = ['id','text'] + others + labels
merged = merged[desired_order]

# --- 8) Write out full combined dataset ------------------------------------
combined_path = os.path.join(output_dir, "combined.tsv")
merged.to_csv(combined_path, sep='\t', index=False)
print(f"Wrote full combined dataset to: {combined_path}")

# --- 9) Split by humanitarian’s original split and write -------------------
# We need orig_split back for splitting—reload it:
orig_splits = h_sub[['id','orig_split']].drop_duplicates(subset='id')
merged = pd.merge(orig_splits, merged, on='id', how='right')

for sp in splits:
    df_sp = merged[merged['orig_split'] == sp].drop(columns=['orig_split'])
    out_sp = os.path.join(output_dir, f"combined_{sp}.tsv")
    df_sp.to_csv(out_sp, sep='\t', index=False)
    print(f"Wrote {sp} split ({len(df_sp):,} rows): {out_sp}")

# --- 10) Print final label distributions ------------------------------------
def print_dist(name, series):
    vc = series.value_counts(dropna=False)
    tot = vc.sum()
    print(f"\n{name} (total {tot:,}):")
    for lab, ct in vc.items():
        print(f" - {lab:30s}: {ct:6,} ({100*ct/tot:5.2f}%)")

print_dist("▶ Final humanitarian labels", merged['humanitarian_label'])
print_dist("▶ Final informativeness labels", merged['informativeness_label'])
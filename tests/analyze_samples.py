import sys
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv('results/sample_translations.csv')

# Find matching columns case-insensitively
cols_lower = {c.lower(): c for c in df.columns}
en_col = next((c for c in df.columns if 'en' in c.lower()), df.columns[0])
ref_col = next((c for c in df.columns if 'ref' in c.lower()), df.columns[1])
pred_col = next((c for c in df.columns if 'pred' in c.lower()), df.columns[2])

print(f"Total evaluated pairs: {len(df)}")
print("=" * 60)
for i in range(min(30, len(df))):
    row = df.iloc[i]
    print(f"[{i+1}] EN: {row[en_col]}")
    print(f"    REF: {row[ref_col]}")
    print(f"    PRED: {row[pred_col]}")
    print("-" * 40)

import sys
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
df = pd.read_csv('results/sample_translations.csv')

print(f"Total evaluated pairs: {len(df)}")
print("=" * 60)
for i in range(30):
    row = df.iloc[i]
    print(f"[{i+1}] EN: {row['english']}")
    print(f"    REF: {row['reference']}")
    print(f"    PRED: {row['prediction']}")
    print("-" * 40)

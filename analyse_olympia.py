from CSFSLoader import CSFSLoader
from infoformulas_listcomp import IG

path = "datasets/olympia/Olympia_2_update.csv"
df = CSFSLoader().load_dataset(path)
target = "medals"

print(df.describe())
df = df[:20]
import pandas as pd
ig_data = {f:IG(df[target], df[f]) for f in df}

ordered = sorted(ig_data, key=ig_data.__getitem__, reverse=True)
for f in ordered:
    print(f, ig_data[f])

ordered.remove(target)
ordered.remove('id')
print('best 10:')
print(ordered[:10])
print('worst 10:')
print(ordered[-10:])
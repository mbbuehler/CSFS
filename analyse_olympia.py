from CSFSLoader import CSFSLoader

path = "datasets/olympia/Olympia_2_update.csv"
df = CSFSLoader().load_dataset(path)
print(df.describe())
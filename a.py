import pandas as pd
df = pd.read_parquet("warehouse/parquet/clusters_merged.parquet")
print(df.head())
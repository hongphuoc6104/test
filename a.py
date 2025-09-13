import pandas as pd

df = pd.read_parquet("warehouse/parquet/clusters_merged.parquet")

# Ví dụ: xem tất cả cụm gốc merge thành Character 2
print(df[df["final_character_id"] == 0][["movie", "cluster_id", "frame"]].head(20))

# Hoặc: tìm cụm gốc 78 thuộc về Character nào
print(df[df["cluster_id"] == "0_78"]["final_character_id"].unique())

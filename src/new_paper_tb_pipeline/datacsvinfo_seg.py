import pandas as pd

csv_path = r"G:\Ai-CDD\data\segmentation\TBX11K_segmentation dataset\tbx11k-simplified\data.csv"

df = pd.read_csv(csv_path)

print(df.head())
print(df.columns)
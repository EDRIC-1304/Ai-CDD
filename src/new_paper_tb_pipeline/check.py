import pandas as pd
import ast

csv_path = (
    r"G:\Ai-CDD\data\segmentation"
    r"\TBX11K_segmentation dataset"
    r"\tbx11k-simplified\data.csv"
)

df = pd.read_csv(csv_path)

tb_rows = df[df["bbox"] != "none"]

bbox = ast.literal_eval(tb_rows.iloc[0]["bbox"])

print(bbox)
print(type(bbox))
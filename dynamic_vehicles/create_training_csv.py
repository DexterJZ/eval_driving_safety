import pandas as pd
import os
import torch


device = ("cuda" if torch.cuda.is_available() else "cpu")

training_df = pd.DataFrame(columns=["img_name", "label"])
training_df["img_name"] = sorted(os.listdir("data/training_image_2/"))

for idx, i in enumerate(sorted(os.listdir("data/training_image_2/"))):
    if "s" in i:
        training_df["label"][idx] = 0
    if "d" in i:
        training_df["label"][idx] = 1

training_df.to_csv(r'training_csv.csv', index=False, header=True)

import pandas as pd
import os


def txt2png(names):
    return [(name[:-4]+'.png') for name in names if name.endswith('.txt')]


training_df = pd.DataFrame(columns=["img_name", "label"])
dynamics = txt2png(os.listdir("data/dynamic/"))
road_dynamics = txt2png(os.listdir("data/road_dynamic/"))
statics = txt2png(os.listdir("data/static/"))

# 0: highway
# 1: road
dyna_label = [1] * len(dynamics)
road_dyna_label = [0] * len(road_dynamics)
static_label = [1] * len(statics)

training_df["img_name"] = dynamics + road_dynamics + statics
training_df["label"] = dyna_label + road_dyna_label + static_label

training_df = training_df[training_df.img_name != ".DS_Store"]
training_df = training_df.sort_values(by='img_name', ignore_index=True)
training_df.to_csv(r'training_csv.csv', index=False, header=True)



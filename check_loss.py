import json
import glob
import numpy as np
import pandas as pd

# file_paths = glob.glob("models_loss/*.json")

# for file in file_paths:
#     with open(file, "r") as f:
#         loss = json.load(f)
#     print(file[12:-5])
#     print((loss["loss"][-1]))
#     print((loss["val_loss"][-1]))

with open("mse_per.json", "r") as f:
    mse = json.load(f)

df = pd.DataFrame({"name": mse.keys(), "mse": mse.values()})
print(df.loc[df["mse"].idxmin()])

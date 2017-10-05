import pandas as pd

data = pd.read_csv("data.csv", header=0, sep=";")

to_drop = [0]
to_binarize = [1, 5, 7, 18, 20, 21, 25, 26]
classFeature = 29

# save to dict features
# add to X columns for new features
# remove from X columns with old features
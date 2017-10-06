import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score

data = pd.read_csv("data.csv", header=0, sep=";")

to_drop = [0]
to_full_binarize = [1, 5, 7, 18, 21, 26]
binarized = {1:0, 5:0, 7:0, 18:0, 21:0, 26:0}
to_number = [20, 25]
target_feature = 29
rows, columns = data.shape

# save to dict features
bin_vals = dict ()
for ii in to_full_binarize:
    bin_vals[ii] = []
    for i in range(0, rows):
        smth = data.iat[i, ii]
        if (smth not in bin_vals[ii]):
            bin_vals[ii].append(smth)
            binarized[ii] += 1

for j in to_number:
    num = 0
    cat_vals = dict()
    for i in range(0, rows):
        smth = data.iloc[i][j]
        if (smth not in cat_vals):
            cat_vals[smth] = num
            num += 1
        data.iat[i, j] = cat_vals[smth]

new_col = 0
for i in range(0, len(to_full_binarize)):
    new_col += binarized[to_full_binarize[i]]

#print(new_col)

# count new X size
old_columns = columns - len(to_drop) - len(to_full_binarize) - 1
new_col_num = old_columns + new_col
X = np.array([[0.0] * new_col_num] * rows)
y = np.array([0] * rows)

for i in range(0, rows):
    y[i] = data.iat[i, target_feature]
    jj = 0
    for j in range(1, columns): # drop 0-th column
        if (j != target_feature and j not in to_full_binarize): # remove from X columns with old features
            smth = float(str(data.iat[i, j]).replace(',', '.'))
            if (math.isnan(smth)):
                X[i][jj] = -1.0
            else:
                X[i][jj] = smth
            jj += 1
    offset = old_columns
    for j in to_full_binarize: # add columns with binirized features
        smth = data.iloc[i][j]
        X[i][offset + bin_vals[j].index(smth)] = 1.0
        offset += binarized[j]

#print(X[0])
#print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(X_train, y_train)
predictions = gbm.predict(X_test)
print(accuracy_score(y_test, predictions))
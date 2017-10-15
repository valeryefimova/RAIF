import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA

from collections import OrderedDict

data = pd.read_csv("data.csv", header=0, sep=";")

to_drop = [0]
to_full_binarize = [1, 5, 7, 18, 21, 26]
binarized = {1:0, 5:0, 7:0, 18:0, 21:0, 26:0}
car_number = 20
region = [25]
region_to_group = 8
car_group = 9
target_feature = 29
rows, columns = data.shape

#for i in range(0, rows):
#    if (data.iloc[i][target_feature] == 1):
#        print(data.iloc[i])

# save to dict features
bin_vals = dict ()
for ii in to_full_binarize:
    bin_vals[ii] = []
    for i in range(0, rows):
        smth = data.iat[i, ii]
        if (smth not in bin_vals[ii]):
            bin_vals[ii].append(smth)
            binarized[ii] += 1

new_col = 0
for i in range(0, len(to_full_binarize)):
    new_col += binarized[to_full_binarize[i]]

#print(new_col)

# count new X size
old_columns = columns - len(to_drop) - len(to_full_binarize) - 1
new_col_num = old_columns + new_col + region_to_group + car_group
X = np.array([[0.0] * new_col_num] * rows)
y = np.array([0] * rows)

unclass = 0
unclass_dict = dict()
cat_vals = dict()
num = 0
for i in range(0, rows):
    y[i] = data.iat[i, target_feature]
    jj = 0
    for j in range(1, columns): # drop 0-th column
        if (j != target_feature and j != car_number and j not in to_full_binarize and j not in region): # remove from X columns with old features
            smth = float(str(data.iat[i, j]).replace(',', '.'))
            if (math.isnan(smth)):
                X[i][jj] = -1.0
            else:
                X[i][jj] = smth
            jj += 1
        elif (j == car_number):
            smth = data.iloc[i][j]
            # TODO: select class for each
            # HYUNDAI ACCENT, VOLKSWAGEN TOUAREG
            # LADA GRANTA, MITSUBISHI OUTLANDER, MERCEDES-BENZ C 180
            # KIA CEED, MAZDA 3, SUBARU LEGACY, MITSUBISHI LANCER
            # HYUNDAI SOLARIS, MAZDA 6, BMW X5, TOYOTA LAND CRUISER
            # HONDA CIVIC, CHEVROLET CRUZE, DAEWOO NEXIA, TOYOTA MARKII
            # LADA KALINA, MERCEDES-BENZ E 200, LADA PRIORA
            # HONDA CR-V, AUDI A4, NISSAN QASHQAI
            # VOLKSWAGEN PASSAT, TOYOTA LAND CRUISER PRADO
            # NISSAN ALMERA

            if ('SUBARU FORESTER' in smth or 'TOYOTA RAV4' in smth):
                X[i][new_col_num - 9] = 1.0
            elif ('ДРУГОЕ ТС ДРУГОЕ ТС ЛЕГКОВОЙ' in smth):
                X[i][new_col_num - 8] = 1.0
            elif ('ДРУГОЕ ТС ДРУГОЕ ТС ГРУЗОВОЙ' in smth or 'КАМАЗ' in smth or 'ГАЗ' in smth
                  or 'ДРУГОЕ ТС ДРУГОЕ ТС ТРАКТОРА' in smth or 'микрогрузовик' in smth
                  or 'ДРУГОЕ ТС ДРУГОЕ ТС АВТОБУСЫ' in smth or 'ПАЗ' in smth):
                X[i][new_col_num - 7] = 1.0
            elif ('CITROEN C3' in smth or 'FORD FUSION' in smth or 'FORD FIESTA' in smth
                or 'MITSUBISHI COLT' in smth or 'KIA RIO' in smth or 'HONDA JAZZ' in smth
                or 'NISSAN MICRA' in smth or 'NISSAN NOTE' in smth or 'HONDA FIT' in smth
                or 'FIAT ALBEA' in smth or 'MERCEDES-BENZ A-KLASSE' in smth or 'VOLKSWAGEN POLO' in smth
                or 'MAZDA DEMIO' in smth or 'SKODA FABIA' in smth or 'TOYOTA IST' in smth
                or 'OPEL CORSA' in smth or 'ВАЗ' in smth): # B - 16
                X[i][new_col_num - 6] = 1.0
            elif ('CHEVROLET LACETTI' in smth or 'CITROEN C4' in smth or 'FORD FOCUS' in smth
                  or 'KIA CERATO' in smth or 'OPEL ASTRA' in smth or 'PEUGEOT 307' in smth or 'HYUNDAI ELANTRA' in smth
                  or 'HYUNDAI SONATA' in smth or 'HYUNDAI MATRIX' in smth or 'SUZUKI LIANA' in smth
                  or 'SKODA OCTAVIA' in smth or 'TOYOTA COROLLA' in smth): #C - 12
                X[i][new_col_num - 5] = 1.0
            elif ('CHEVROLET EPICA' in smth or 'CHRYSLER SEBRING' in smth or 'CITROEN C5' in smth
                  or 'KIA MAGENTIS' in smth or 'HONDA ACCORD' in smth or 'OPEL VECTRA'in smth
                  or 'SKODA SUPERB' in smth or 'TOYOTA AVENSIS' in smth or 'VOLVO S40' in smth
                  or 'VOLVO S60' in smth): #D - 10
                X[i][new_col_num - 4] = 1.0
            elif ('AUDI A6' in smth or 'BMW 5 SERIES' in smth or 'LEXUS GS' in smth
                  or 'MITSUBISHI GALANT' in smth or 'HONDA LEGEND' in smth or 'JAGUAR S-TYPE' in smth
                  or 'VOLVO S80' in smth or 'NISSAN TEANA' in smth or 'HYUNDAI GRANDEUR' in smth
                  or 'MERCEDES-BENZ E-KLASSE' in smth or 'INFINITI M' in smth): #E - 11
                X[i][new_col_num - 3] = 1.0
            elif ('PORSCHE' in smth or 'CADILLAC' in smth or 'DODGE' in smth or 'JAGUAR' in smth
                  or 'MERCEDES-BENZ S-KLASSE' in smth or 'AUDI S8' in smth or 'HYUNDAI EQUUS' in smth
                  or 'LEXUS LS' in smth): #F - 8
                X[i][new_col_num - 2] = 1.0
            else:
                X[i][new_col_num - 1] = 1.0
                unclass += 1
                if (smth not in unclass_dict):
                    unclass_dict[smth] = 1
                else:
                    unclass_dict[smth] += 1
            if (smth not in cat_vals):
                cat_vals[smth] = num
                num += 1
            data.iat[i, j] = cat_vals[smth]

    offset = old_columns
    for j in to_full_binarize: # add columns with binirized features
        smth = data.iloc[i][j]
        X[i][offset + bin_vals[j].index(smth)] = 1.0
        offset += binarized[j]

    # region to group from 1 to 8
    smth = str(data.iloc[i][25])
    if ('Москва' in smth):
        X[i][offset] = 1.0
    elif ('Санкт-Петербург' in smth):
        X[i][offset + 1] = 1.0
    elif ('Московская' in smth):
        X[i][offset + 2] = 1.0
    elif ('Ленинградская' in smth or 'Татарстан' in smth or 'автономный округ' in smth):
        X[i][offset + 3] = 1.0
    elif ('Краснодарский' in smth or 'Свердловская' in smth or 'Новосибирская' in smth or 'Самарская' in smth
          or 'Ростовская' in smth or 'Рязанская' in smth or 'Липецкая' in smth or 'Сахалинская' in smth
          or 'Нижегородская' in smth or 'Белгородская' in smth or 'Воронежская' in smth or 'Тюменская' in smth
          or 'Калининградская' in smth or 'Курская' in smth): # middle
        X[i][offset + 4] = 1.0
    elif ('Республика' in smth):
        X[i][offset + 5] = 1.0
    elif ('Новосибирская' in smth or 'Кемеровская' in smth or 'Ставропольский' in smth or 'Алтайский' in smth
          or 'Смоленская' in smth): # poor
        X[i][offset + 6] = 1.0
    else:
        X[i][offset + 7] = 1.0

print(unclass)
s = [(k, unclass_dict[k]) for k in sorted(unclass_dict, key=unclass_dict.get, reverse=True)]
print(s)

for i in range(0, 10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=i)
    gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(X_train, y_train)
    predictions = gbm.predict(X_test)
    rg = xgb.XGBRegressor().fit(X_train, y_train)
    predictions_reg = rg.predict(X_test)
    print('Classification score = ' + str(accuracy_score(y_test, predictions))
          + ' Regression score = ' + str(mean_squared_error(y_test, predictions_reg)))

    X_test = PCA(n_components=2).fit_transform(X_test)

    plt.figure(figsize=(8, 6))
    zero_class = np.where(predictions_reg < 0.6)
    one_class = np.where(predictions_reg > 0.9)
    true_zero_class = np.where(y_test < 1)
    true_one_class = np.where(y_test > 0)
    plt.scatter(X_test[true_zero_class, 0], X_test[true_zero_class, 1], s=40, c='b', edgecolors=(0, 0, 0), label='true class 0')
    plt.scatter(X_test[true_one_class, 0], X_test[true_one_class, 1], s=40, c='orange', edgecolors=(0, 0, 0), label='true class 1')
    plt.scatter(X_test[zero_class, 0], X_test[zero_class, 1], s=160, edgecolors='b',
                facecolors='none', linewidths=2, label='class 0')
    plt.scatter(X_test[one_class, 0], X_test[one_class, 1], s=80, edgecolors='orange',
                facecolors='none', linewidths=2, label='class 1')
    plt.show()

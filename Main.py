import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, OrthogonalMatchingPursuit, BayesianRidge, Perceptron
from sklearn.svm import SVR, SVC
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import LeaveOneOut, GridSearchCV

data = pd.read_csv("data.csv", header=0, sep=";")

to_drop = [0, 20]
to_full_binarize = [1, 5, 7, 18, 21, 26]
binarized = {1:0, 5:0, 7:0, 18:0, 21:0, 26:0}
car_number = 20
region = [25]
region_to_group = 8
car_group = 38
target_feature = 29
rows, columns = data.shape

car_models_bad = dict()
for i in range(0, rows):
    if (data.iloc[i][target_feature] == 1):
        tmp = data.iloc[i][car_number]
        if (tmp not in car_models_bad):
            car_models_bad[tmp] = 0
        else:
            car_models_bad[tmp] += 1

s = [(k, car_models_bad[k]) for k in sorted(car_models_bad, key=car_models_bad.get, reverse=True)]
print(s)

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
    for j in range(1, columns): # drop 0-th and 20-th column
        if (j != target_feature and j != car_number and j not in to_full_binarize and j not in region): # remove from X columns with old features
            smth = float(str(data.iat[i, j]).replace(',', '.'))
            if (math.isnan(smth)):
                X[i][jj] = -100.0
            else:
                X[i][jj] = smth
            jj += 1
        elif (j == car_number):
            smth = data.iloc[i][j]

            if ('ВАЗ' in smth):
                X[i][new_col_num - 38] = 1.0
            elif ('SUBARU FORESTER' in smth or 'TOYOTA RAV4' in smth or 'VOLKSWAGEN TOUAREG' in smth
                or 'TOYOTA LAND CRUISER' in smth or 'MITSUBISHI OUTLANDER' in smth or 'BMW X5' in smth
                or 'HONDA CR-V' in smth or 'KIA SPORTAGE' in smth or 'NISSAN QASHQAI' in smth
                or 'NISSAN X-TRAIL' in smth or 'LEXUS RX' in smth):
                X[i][new_col_num - 37] = 1.0
            elif ('ДРУГОЕ ТС ДРУГОЕ ТС ЛЕГКОВОЙ' in smth):
                X[i][new_col_num - 36] = 1.0
            elif ('ДРУГОЕ ТС ДРУГОЕ ТС ГРУЗОВОЙ' in smth or 'КАМАЗ' in smth or 'ГАЗ' in smth
                  or 'ДРУГОЕ ТС ДРУГОЕ ТС ТРАКТОРА' in smth or 'микрогрузовик' in smth
                  or 'ДРУГОЕ ТС ДРУГОЕ ТС АВТОБУСЫ' in smth or 'ПАЗ' in smth or 'УАЗ' in smth
                  or 'ДРУГОЕ ТС ДРУГОЕ ТС МОТОЦИКЛЫ' in smth or 'БОГДАН' in smth or 'ЛИАЗ' in smth
                  or 'МАЗ' in smth or 'МОСКВИЧ' in smth or 'ЗИЛ' in smth or 'ИЖ' in smth
                  or 'ДРУГОЕ ТС ДРУГОЕ ТС ТРОЛЛЕЙБУС' in smth or 'УРАЛ' in smth):
                X[i][new_col_num - 35] = 1.0
            elif ('CITROEN C3' in smth or 'FORD FUSION' in smth or 'FORD FIESTA' in smth
                or 'MITSUBISHI COLT' in smth or 'KIA RIO' in smth or 'HONDA JAZZ' in smth
                or 'NISSAN MICRA' in smth or 'NISSAN NOTE' in smth or 'HONDA FIT' in smth
                or 'FIAT ALBEA' in smth or 'MERCEDES-BENZ A-KLASSE' in smth or 'VOLKSWAGEN POLO' in smth
                or 'MAZDA DEMIO' in smth or 'SKODA FABIA' in smth or 'TOYOTA IST' in smth
                or 'OPEL CORSA' in smth or 'RENAULT LOGAN' in smth or 'HYUNDAI ACCENT' in smth
                or 'LADA GRANTA' in smth or 'LADA KALINA' in smth): # B - 16
                X[i][new_col_num - 34] = 1.0
            elif ('CHEVROLET LACETTI' in smth or 'CITROEN C4' in smth or 'FORD FOCUS' in smth
                  or 'KIA CERATO' in smth or 'OPEL ASTRA' in smth or 'PEUGEOT 307' in smth or 'HYUNDAI ELANTRA' in smth
                  or 'HYUNDAI SONATA' in smth or 'HYUNDAI MATRIX' in smth or 'SUZUKI LIANA' in smth
                  or 'SKODA OCTAVIA' in smth or 'TOYOTA COROLLA' in smth or 'HYUNDAI SOLARIS' in smth
                  or 'LADA PRIORA' in smth or 'MITSUBISHI LANCER' in smth or 'DAEWOO NEXIA' in smth
                  or 'HONDA CIVIC' in smth or 'MAZDA 3' in smth or 'MERCEDES-BENZ C 180' in smth
                  or 'CHEVROLET CRUZE' in smth or 'NISSAN ALMERA' in smth or 'SUBARU LEGACY' in smth
                  or 'KIA CEED' in smth or 'SUBARU IMPREZA' in smth or 'VOLKSWAGEN JETTA' in smth): #C - 12
                X[i][new_col_num - 33] = 1.0
            elif ('CHEVROLET EPICA' in smth or 'CHRYSLER SEBRING' in smth or 'CITROEN C5' in smth
                  or 'KIA MAGENTIS' in smth or 'HONDA ACCORD' in smth or 'OPEL VECTRA'in smth
                  or 'SKODA SUPERB' in smth or 'TOYOTA AVENSIS' in smth or 'VOLVO S40' in smth
                  or 'VOLVO S60' in smth or 'MAZDA 6' in smth or 'AUDI A4' in smth): #D - 10
                X[i][new_col_num - 32] = 1.0
            elif ('AUDI A6' in smth or 'BMW 5 SERIES' in smth or 'LEXUS GS' in smth
                  or 'MITSUBISHI GALANT' in smth or 'HONDA LEGEND' in smth or 'JAGUAR S-TYPE' in smth
                  or 'VOLVO S80' in smth or 'NISSAN TEANA' in smth or 'HYUNDAI GRANDEUR' in smth
                  or 'MERCEDES-BENZ E-KLASSE' in smth or 'INFINITI M' in smth
                  or 'TOYOTA CAMRY' in smth or 'MERCEDES-BENZ E 200' in smth or 'VOLKSWAGEN PASSAT' in smth
                  or 'TOYOTA MARKII' in smth or 'BMW 520' in smth): #E - 11
                X[i][new_col_num - 31] = 1.0
            elif ('PORSCHE' in smth or 'CADILLAC' in smth or 'DODGE' in smth or 'JAGUAR' in smth
                  or 'MERCEDES-BENZ S-KLASSE' in smth or 'AUDI S8' in smth or 'HYUNDAI EQUUS' in smth
                  or 'LEXUS LS' in smth or 'BENTLEY' in smth): #F - 8
                X[i][new_col_num - 30] = 1.0
            elif ('CITROEN' in smth):
                X[i][new_col_num - 29] = 1.0
            elif ('ISUZU' in smth):
                X[i][new_col_num - 28] = 1.0
            elif ('OPEL' in smth):
                X[i][new_col_num - 27] = 1.0
            elif ('VOLVO' in smth):
                X[i][new_col_num - 26] = 1.0
            elif ('FIAT' in smth):
                X[i][new_col_num - 25] = 1.0
            elif ('INFINITI' in smth):
                X[i][new_col_num - 24] = 1.0
            elif ('SUBARU' in smth):
                X[i][new_col_num - 23] = 1.0
            elif ('LEXUS' in smth):
                X[i][new_col_num - 22] = 1.0
            elif ('MAZDA' in smth):
                X[i][new_col_num - 21] = 1.0
            elif ('SKODA' in smth):
                X[i][new_col_num - 20] = 1.0
            elif ('HONDA' in smth):
                X[i][new_col_num - 19] = 1.0
            elif ('KIA' in smth):
                X[i][new_col_num - 18] = 1.0
            elif ('AUDI' in smth):
                X[i][new_col_num - 17] = 1.0
            elif ('LAND ROVER' in smth):
                X[i][new_col_num - 16] = 1.0
            elif ('DAEWOO' in smth):
                X[i][new_col_num - 15] = 1.0
            elif ('CHEVROLET' in smth):
                X[i][new_col_num - 14] = 1.0
            elif ('PEUGEOT' in smth):
                X[i][new_col_num - 13] = 1.0
            elif ('HYUNDAI' in smth):
                X[i][new_col_num - 12] = 1.0
            elif ('SUZUKI' in smth):
                X[i][new_col_num - 11] = 1.0
            elif('LADA' in smth):
                X[i][new_col_num - 10] = 1.0
            elif('FORD' in smth):
                X[i][new_col_num - 9] = 1.0
            elif ('BMW' in smth):
                X[i][new_col_num - 8] = 1.0
            elif ('VOLKSWAGEN' in smth):
                X[i][new_col_num - 7] = 1.0
            elif ('TOYOTA' in smth):
                X[i][new_col_num - 6] = 1.0
            elif ('MITSUBISHI' in smth):
                X[i][new_col_num - 5] = 1.0
            elif ('RENAULT' in smth):
                X[i][new_col_num - 4] = 1.0
            elif ('NISSAN' in smth):
                X[i][new_col_num - 3] = 1.0
            elif ('MERCEDES-BENZ' in smth):
                X[i][new_col_num - 2] = 1.0
            else:
                #X[i][new_col_num - 1] = 1.0
                unclass += 1
                if (smth not in unclass_dict):
                    unclass_dict[smth] = 1
                else:
                    unclass_dict[smth] += 1
            #if (smth not in cat_vals):
            #    cat_vals[smth] = num
            #    num += 1
            #data.iat[i, j] = cat_vals[smth]

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

print('unclassified cars = ' + str(unclass))
s = [(k, unclass_dict[k]) for k in sorted(unclass_dict, key=unclass_dict.get, reverse=True)]
print(s)
print('number of all new features = ' + str(new_col_num))

X_new = X

#X_new = SelectKBest(chi2, k=50).fit_transform(X, y)

# model = LinearRegression()
# model.fit(X, y)
# select = 80
# rfe = RFE(model, select)
# rfe = rfe.fit(X, y)
# print('Ranking of features, 1 == important')
# print(rfe.ranking_)
# selected_features = []
# for i in range(0, len(rfe.ranking_)):
#     if (rfe.ranking_[i] == 1):
#         selected_features.append(i)
#
# X_new = np.array([[0.0] * select] * rows)
# for i in range(0, rows):
#     jj = 0
#     for j in range(0, new_col_num):
#         if (j in selected_features):
#             X_new[i][jj] = X[i][j]
#             jj += 1

# loo = LeaveOneOut()
# sum_reg = 0.0
# sum_clf = 0.0
# sum_f1  = 0.0
# ind = 0
# num_llo = 50
# for train_index, test_index in loo.split(X):
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#     rg = SVR(C=50.0, epsilon=0.2)
#     rg.fit(X_train, y_train)
#     y_pred = rg.predict(X_test)
#     sum_reg += mean_squared_error(y_test, y_pred)
#
#     svc = SVC(C=50.0)
#     params = {'C':[1, 100]}
#     clf = GridSearchCV(svc, params)
#     clf.fit(X_train, y_train)
#     y_pred = clf.predict(X_test)
#     sum_clf += accuracy_score(y_test, y_pred)
#     sum_f1 += f1_score(y_test, y_pred)
#
#     print(ind)
#     ind += 1
#     if (ind == num_llo):
#         break
#
# sum_reg /= float(num_llo)
# sum_clf /= float(num_llo)
# sum_f1  /= float(num_llo)
# print('regression error = ' + str(sum_reg))
# print('classification accurcy = ' + str(sum_clf) + ' f1 = ' + str(sum_f1))

params = {'C':[1, 100]}
for i in range(0, 10):
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.01, random_state=i)
    #gbm = xgb.XGBClassifier().fit(X_train, y_train)
    gbm = SVC(C=50)
    #gbm = GridSearchCV(svc, params)
    gbm.fit(X_train, y_train)
    predictions = gbm.predict(X_test)
    #rg = xgb.XGBRegressor().fit(X_train, y_train)
    #rg = LinearRegression().fit(X_train, y_train)
    #rg = Lasso(alpha=0.1).fit(X_train, y_train) bad
    #rg = ElasticNet(alpha=0.1, l1_ratio=0.7).fit(X_train, y_train) bad
    #rg = OrthogonalMatchingPursuit(n_nonzero_coefs=17).fit(X_train, y_train) many mistaces
    #rg = BayesianRidge().fit(X_train, y_train)
    #rg = Perceptron() # fuuuuuuuuu
    rg = SVR(C=50.0, epsilon=0.2) # good, tune hyperparameters
    #rg = GridSearchCV(svr, params)
    rg.fit(X_train, y_train)
    predictions_reg = rg.predict(X_test)
    print('Classification score = ' + str(accuracy_score(y_test, predictions))
          + ' Regression score = ' + str(mean_squared_error(y_test, predictions_reg)))

    X_test = PCA(n_components=2).fit_transform(X_test)

    plt.figure(figsize=(8, 6))
    #print(predictions_reg)
    zero_class = np.where(predictions_reg < 0.6)
    one_class = np.where(predictions_reg > 0.6)
    true_zero_class = np.where(y_test < 1) # 0 class
    true_one_class = np.where(y_test > 0) # 1 class
    plt.scatter(X_test[true_zero_class, 0], X_test[true_zero_class, 1], s=40, c='b', edgecolors=(0, 0, 0), label='true class 0')
    plt.scatter(X_test[true_one_class, 0], X_test[true_one_class, 1], s=40, c='orange', edgecolors=(0, 0, 0), label='true class 1')
    plt.scatter(X_test[zero_class, 0], X_test[zero_class, 1], s=160, edgecolors='b',
                facecolors='none', linewidths=2, label='predicted class 0')
    plt.scatter(X_test[one_class, 0], X_test[one_class, 1], s=80, edgecolors='orange',
                facecolors='none', linewidths=2, label='predicted class 1')
    plt.legend()
    plt.show()

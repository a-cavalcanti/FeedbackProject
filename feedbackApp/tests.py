import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from views import extract_liwc, cross_validation, additionals


classifiers = []
for i in range(11):
    classifiers.append(RandomForestClassifier(n_estimators=200, max_features=37, warm_start=True, oob_score=True))


def read_classes2(path):
    data = pd.read_csv(path)
    id_name = data.keys()[0]
    print(id_name)
    del data[id_name]  # Remove o id
    return data.values.tolist()


def read_classes(path):
    data = pd.read_csv(path)
    id_name = data.keys()[0]
    vector = []
    print(id_name)
    # Remove id col
    del data[id_name]
    # 11 -> classes number
    for i in range(11):
        vector.append(data[data.keys()[i]].values.tolist())
    return vector


def read_data(csv):
    data = pd.read_csv(csv)

    # id col name
    id_name = data.keys()[0]
    new_data = data.copy()

    # Remove id col
    del new_data[id_name]
    return data, new_data.values.tolist()


def load_classifier():
    print(os.getcwd())
    csv_classes = "classes.csv"
    classes = read_classes(csv_classes)

    csv_features = "features.csv"
    data_train, features = read_data(csv_features)
    X1 = None
    y1 = None
    metricas = ["acurácia", "kappa", "OOB_erro"]

    for j in range(len(classes)):
        print("classe " + str(j))
        resultados = {}
        resultados.update({'ntree': []})
        resultados.update({'mtry': []})

        y_train = classes[j]
        resultados.update({"erro_classe_" + str(0): []})
        resultados.update({"erro_classe_" + str(1): []})
        for metrica in metricas:
            resultados.update({metrica: []})

        global classifiers
        resultados, classifiers[j] = cross_validation(features, y_train, X1, y1, k=10, ntree=200, mtry=37,
                                                      metricas=metricas, resultados=resultados)
        print(resultados)
        data = {}
        data['dados'] = "pronto"


def predict_classes():
    for j in range(11):
        texto = "Excelente aluno! Parabéns. Você fez uma ótima atividade!"

        liwc = extract_liwc(texto)
        adds = additionals(texto)
        cohmetrix = [50.0, 0.0, 500.0, 86.405, 450.0, 2.0, 2.5, 10.0, 150.0, 1.0, 2.0, 20.0, 100.0, 300.0, 50.0, 0.0, 0.0,
                     0.0, 50.0, 76562.5, 3441.5, 1.0, 0.0, 0.0, 0.95, 0.25, 250.0, 0.0, 150.0, 0.0, 50.0, 0.0, 100.0, 0.0,
                     0.0, 0.0, 0.0, 0.0, 0.0, 1.66666666666667, 10.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        features = []
        for x in liwc:
            features.append(x)
        for y in cohmetrix:
            features.append(y)
        features.append(adds[0])
        features.append(adds[1])
        features.append(0)
        features.append(0)

        newfeatures = []
        newfeatures.append(features)
        global classifiers
        y_pred = classifiers[j].predict(newfeatures)
        print("classe predita ", y_pred)


# classes = read_classes('classes.csv')
# data, features = read_data('features.csv')
# for line in features:
#     print(line)
#     print(len(line))
load_classifier()
predict_classes()

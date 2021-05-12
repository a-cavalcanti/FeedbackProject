import cgi

from django.shortcuts import render

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedKFold, cross_validate, RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score  # , mean_squared_error
import os
import xgboost as xgb

from nltk.tokenize import word_tokenize
import string
import nltk

nltk.download('punkt')
# import spacy
# nlp = spacy.load('pt')

# Create your views here.

classifiers = []
for i in range(11):
    # classifiers.append(RandomForestClassifier(n_estimators=200, max_features=37, warm_start=True, oob_score=True))
    classifiers.append(xgb.XGBClassifier(n_estimators=500, use_label_encoder=False))


def home(request):
    data = {}
    # var1 = "texto em python"
    # data['dados'] = var1
    return render(request, 'feedbackApp/home.html')


def dados(request):
    data = {}
    var1 = "texto var1"
    var2 = " texto var2"
    data['dados'] = [var1, var2]
    return render(request, 'feedbackApp/dados.html', data)


def process_text(request):
    formulario = cgi.FieldStorage()
    texto = request.POST['mensagem']
    print(texto)
    data = {}
    for j in range(11):
        liwc = extract_liwc(texto)
        adds = additionals(texto)
        cohmetrix = [50.0, 0.0, 500.0, 86.405, 450.0, 2.0, 2.5, 10.0, 150.0, 1.0, 2.0, 20.0, 100.0, 300.0, 50.0, 0.0,
                     0.0,
                     0.0, 50.0, 76562.5, 3441.5, 1.0, 0.0, 0.0, 0.95, 0.25, 250.0, 0.0, 150.0, 0.0, 50.0, 0.0, 100.0,
                     0.0,
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
        np_features = np.asarray(newfeatures)

        global classifiers
        y_pred = classifiers[j].predict(np_features)
        print("classe predita ", y_pred)
        data['classe'+str(j)] = y_pred[0]
    return render(request, 'feedbackApp/home.html', data)


def load_classifier(request):
    print(os.getcwd())
    csv_classes = "feedbackApp/classes.csv"
    classes = read_classes(csv_classes)

    csv_features = "feedbackApp/features.csv"
    data_train, features = read_data(csv_features)

    data = {}
    for j in range(len(classes)):

        resultados = {}
        resultados.update({'ntree': []})
        resultados.update({'mtry': []})
        resultados.update({'acurácia': []})
        resultados.update({'kappa': []})
        resultados.update({'accuracy': []})
        resultados.update({'erro': []})
        resultados.update({"erro_classe_" + str(0): []})
        resultados.update({"erro_classe_" + str(1): []})

        y_train = classes[j]

        global classifiers
        resultados, classifiers[j] = cross_validation(features, y_train, k=10, ntree=500, mtry=37,
                                                      resultados=resultados)
        print(resultados)
        data['dados'] = "pronto"

    return render(request, 'feedbackApp/home.html', data)


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


def search(lista, valor):
    return [lista.index(x) for x in lista if valor in x]


def extract_liwc(text):
    # reading liwc
    wn = open('feedbackApp/LIWC2007_Portugues_win.dic.txt', 'r', encoding='utf-8', errors='ignore').read().split('\n')
    word_set_liwc = []
    for line in wn:
        words = line.split('\t')
        if words != []:
            word_set_liwc.append(words)

    # indexes of liwc
    indices = open('feedbackApp/indices.txt', 'r', encoding='utf-8', errors='ignore').read().split('\n')

    words_line = []
    for word in word_tokenize(text):
        if word not in string.punctuation + "\..." and word != '``' and word != '"':
            words_line.append(word.lower())

    # initializing liwc with zero
    liwc = [0] * len(indices)

    print("writing liwc ")

    for word in words_line:
        position = search(word_set_liwc, word)
        if position != []:
            tam = len(word_set_liwc[position[0]])
            for i in range(tam):
                if word_set_liwc[position[0]][i] in indices:
                    position_indices = search(indices, word_set_liwc[position[0]][i])
                    liwc[position_indices[0]] = liwc[position_indices[0]] + 1

    return liwc


# ------------------------------------ADITIONAL FEATURES---------------------------------------------------------------#
# additional features
def additionals(post):
    original_post = post.lower()
    # post = nlp(post)

    greeting = sum([word_tokenize(original_post).count(word) for word in
                    ['olá', 'oi', 'como vai', 'tudo bem', 'como está', 'como esta', 'bom dia', 'boa tarde',
                     'boa noite']])
    compliment = sum([word_tokenize(original_post).count(word) for word in
                      ['parabéns', 'parabens', 'excelente', 'fantástico', 'fantastico', 'bom', 'bem', 'muito bom',
                       'muito bem', 'ótimo', 'otimo', 'incrivel', 'incrível', 'maravilhoso', 'sensacional',
                       'irrepreensível', 'irrepreensivel', 'perfeito']])
    # ners = len(post.ents)

    return [greeting, compliment]


def cross_validation(X, y, k, ntree, mtry, resultados):

    # SAVE RESULTS FOR EACH ROUND OF CROSS-VALIDATION
    resultados_parciais = {}
    resultados_parciais.update({'acurácia': []})
    resultados_parciais.update({'kappa': []})

    # cross-validation
    rkf = RepeatedStratifiedKFold(n_splits=k, n_repeats=1, random_state=54321)
    matriz_confusao = np.zeros((2, 2))

    for train_index, test_index in rkf.split(X, y):

        X_train, X_test = [X[i] for i in train_index], [X[j] for j in test_index]
        y_train, y_test = [y[i] for i in train_index], [y[j] for j in test_index]

        X_train_np = np.asarray(X_train)
        X_test_np = np.asarray(X_test)
        y_train_np = np.asarray(y_train)
        y_test_np = np.asarray(y_test)

        classificador = xgb.XGBClassifier(n_estimators=ntree, use_label_encoder=False)
        classificador.fit(X_train_np, y_train_np)
        y_pred = classificador.predict(X_test_np)
        y_pred_np = np.asarray(y_pred)

        resultados_parciais["acurácia"].append(accuracy_score(y_pred_np, y_test_np))
        resultados_parciais["kappa"].append(cohen_kappa_score(y_pred_np, y_test_np))

        # THE FINAL CONFUSION MATRIX WILL BE THE SUM OF CONFUSION MATRICES FOR EACH KFOLD ROUND
        matriz_confusao = matriz_confusao + confusion_matrix(y_pred=y_pred_np, y_true=y_test_np)

    # SAVING PARAMETERS AND EXPERIMENT RESULTS
    resultados['ntree'].append(classificador.n_estimators)
    erro_por_classe(matriz_confusao, resultados)

    media = np.mean(resultados_parciais["acurácia"])
    std = np.std(resultados_parciais["acurácia"])
    resultados["acurácia"].append(str(round(media, 4))+"("+str(round(std, 4))+")")

    resultados["accuracy"].append(round(media, 4))
    resultados["erro"].append(round(1 - media, 4))

    media = np.mean(resultados_parciais["kappa"])
    std = np.std(resultados_parciais["kappa"])
    resultados["kappa"].append(str(round(media, 4)) + "(" + str(round(std, 4)) + ")")

    return resultados, classificador


def erro_por_classe(matriz_confusao, resultados):
    tam = matriz_confusao.shape[0]

    for i in range(tam):
        acerto = matriz_confusao[i][i]
        total = sum(matriz_confusao[i])

        taxa_erro = round(1 - (acerto / total), 4)
        print(taxa_erro)

        resultados["erro_classe_" + str(i)].append(taxa_erro)

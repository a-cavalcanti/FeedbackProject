import cgi

from django.shortcuts import render

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedKFold, cross_validate, RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score  # , mean_squared_error
import os

from nltk.tokenize import word_tokenize
import string
import nltk

nltk.download('punkt')
# import spacy
# nlp = spacy.load('pt')

# Create your views here.

classifiers = []
for i in range(11):
    classifiers.append(RandomForestClassifier(n_estimators=200, max_features=37, warm_start=True, oob_score=True))


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


def processa(request):
    formulario = cgi.FieldStorage()
    texto = request.POST['mensagem']
    print(texto)
    data = {}
    for j in range(11):
        liwc = extractLiwc(texto)
        adds = aditionals(texto)
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
        global classifiers
        y_pred = classifiers[j].predict(newfeatures)
        print("classe predita ", y_pred)
        data['classe'+str(j)] = y_pred[0]
    return render(request, 'feedbackApp/home.html', data)


def load_classifier(request):
    print(os.getcwd())
    csv_classes = "feedbackApp/classes.csv"
    classes = read_classes(csv_classes)

    csv_features = "feedbackApp/features.csv"
    data_train, features = read_data(csv_features)
    X1 = None
    y1 = None
    metricas = ["acurácia", "kappa", "OOB_erro"]
    data = {}
    for j in range(len(classes)):
        resultados = {}
        resultados.update({'ntree': []})
        resultados.update({'mtry': []})

        y_train = classes[j]
        resultados.update({"erro_classe_" + str(0): []})
        resultados.update({"erro_classe_" + str(1): []})
        for metrica in metricas:
            resultados.update({metrica: []})

        global classifiers
        resultados, classifiers[j] = validacao_cruzada(features, y_train, X1, y1, k=10, ntree=200, mtry=37,
                                                       metricas=metricas, resultados=resultados)
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


def extractLiwc(text):
    # reading liwc
    wn = open('feedbackApp/LIWC2007_Portugues_win.dic.txt', 'r', encoding='utf-8', errors='ignore').read().split('\n')
    wordSetLiwc = []
    for line in wn:
        words = line.split('\t')
        if (words != []):
            wordSetLiwc.append(words)

    # indexes of liwc
    indices = open('feedbackApp/indices.txt', 'r', encoding='utf-8', errors='ignore').read().split('\n')

    # dataset tokenization
    wordsDataSet = []

    wordsLine = []
    for word in word_tokenize(text):
        if word not in string.punctuation + "\..." and word != '``' and word != '"':
            wordsLine.append(word.lower())

    # initializing liwc with zero
    liwc = [0] * len(indices)
    # liwc.append([0] * len(indices))

    # performing couting

    print("writing liwc ")
    print(liwc)

    for word in wordsLine:
        position = search(wordSetLiwc, word)
        if position != []:
            tam = len(wordSetLiwc[position[0]])
            for i in range(tam):
                if wordSetLiwc[position[0]][i] in indices:
                    positionIndices = search(indices, wordSetLiwc[position[0]][i])
                    liwc[positionIndices[0]] = liwc[positionIndices[0]] + 1

    return liwc


# ------------------------------------ADITIONAL FEATURES---------------------------------------------------------------#
# aditional features
def aditionals(post):
    postOriginal = post.lower()
    # post = nlp(post)

    greeting = sum([word_tokenize(postOriginal).count(word) for word in
                    ['olá', 'oi', 'como vai', 'tudo bem', 'como está', 'como esta', 'bom dia', 'boa tarde',
                     'boa noite']])
    compliment = sum([word_tokenize(postOriginal).count(word) for word in
                      ['parabéns', 'parabens', 'excelente', 'fantástico', 'fantastico', 'bom', 'bem', 'muito bom',
                       'muito bem', 'ótimo', 'otimo', 'incrivel', 'incrível', 'maravilhoso', 'sensacional',
                       'irrepreensível', 'irrepreensivel', 'perfeito']])
    # ners = len(post.ents)

    return [greeting, compliment]


def avaliacao(y_pred, y_test, metricas, classificador, resultados_parciais):
    for metrica in metricas:
        if metrica == "acurácia":
            accuracy = accuracy_score(y_pred, y_test)
            resultados_parciais[metrica].append(accuracy)
        elif metrica == "kappa":
            resultados_parciais[metrica].append(cohen_kappa_score(y_pred, y_test))
        elif metrica == "OOB_erro":
            resultados_parciais[metrica].append(1 - classificador.oob_score_)

    return accuracy


def validacao_cruzada(X, y, X1, y1, k, ntree, mtry, metricas, resultados):
    resultados_parciais = {}  # SALVAR RESULTADOS DE CADA RODADA DA VALIDAÇÃO CRUZADA

    for metrica in metricas:
        resultados_parciais.update({metrica: []})

    ## VALIDAÇÃO CRUZADA

    rkf = RepeatedStratifiedKFold(n_splits=k, n_repeats=1,
                                  random_state=54321)  # DIVIDI OS DADOS NOS CONJUNTOS QUE SERÃO DE      TREINO E TESTE EM CADA RODADA DA VALIDAÇÃO CRUZZADA

    matriz_confusao = np.zeros((2, 2))

    global classificador

    for train_index, test_index in rkf.split(X, y):

        if X1 == None:
            X_train, X_test = [X[i] for i in train_index], [X[j] for j in test_index]
            y_train, y_test = [y[i] for i in train_index], [y[j] for j in test_index]
        else:

            X_train, X_test = [X[i] for i in train_index], [X1[j] for j in test_index]
            y_train, y_test = [y[i] for i in train_index], [y1[j] for j in test_index]

        classificador = RandomForestClassifier(n_estimators=ntree, max_features=mtry, warm_start=True, oob_score=True)
        classificador.fit(X_train, y_train)
        y_pred = classificador.predict(X_test)

        avaliacao(y_pred, y_test, metricas, classificador, resultados_parciais)

        matriz_confusao = matriz_confusao + confusion_matrix(y_pred=y_pred,
                                                             y_true=y_test)  ##A MATRIZ DE CONFUSÃO FINAL SERÁ A SOMA DAS MATRIZES DE CONFUSÃO DE CADA RODADA DO KFOLD

    ## SALVANDO OS PARÊMTROS E RESULTADOS DO EXPERIMENTO

    print(matriz_confusao)
    resultados['ntree'].append(classificador.n_estimators)
    resultados['mtry'].append(classificador.max_features)
    erro_por_classe(matriz_confusao, resultados)

    for metrica in metricas:
        media = np.mean(resultados_parciais[metrica])
        std = np.std(resultados_parciais[metrica])

        resultados[metrica].append(str(round(media, 4)) + "(" + str(round(std, 4)) + ")")

    return resultados, classificador


def erro_por_classe(matriz_confusao, resultados):
    tam = matriz_confusao.shape[0]

    for i in range(tam):
        acerto = matriz_confusao[i][i]
        total = sum(matriz_confusao[i])

        taxa_erro = round(1 - (acerto / total), 4)
        print(taxa_erro)

        resultados["erro_classe_" + str(i)].append(taxa_erro)

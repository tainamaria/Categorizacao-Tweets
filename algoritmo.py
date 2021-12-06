import pandas as pd
from sklearn import svm
from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords 
import nltk

def pre_processamento(opcao): #Esta função utiliza a função de leitura para ler dois arquivos, um que possui dados para o treinamento e outro para os testes.

dftrain=0
dfpred=0
arq=0

if opcao==1:

dftrain, dfpred = lerArquivos("pre_candidatura_treino.csv","pre_candidatura_teste.csv")
arq="_pré_candidatura.csv"

elif opcao==2:

dftrain, dfpred = lerArquivos("durante_candidatura_treino.csv","durante_candidatura_teste.csv")
arq="_durante_candidatura.csv"
    
elif opcao==3:

dftrain, dfpred = lerArquivos("pos_candidatura_treino.csv","pos_candidatura_teste.csv")
arq="_pós_candidatura.csv"

dftrain.loc[dftrain['sentimento']=='neutro', 'sentimento'] = 0
dftrain.loc[dftrain['sentimento']=='ofensivo', 'sentimento'] = 1
dftrain.loc[dftrain['sentimento']=='positivo', 'sentimento'] = 2
dftrain.loc[dftrain['sentimento']=='antidemocrático', 'sentimento'] = 3

x_train, x_test, y_train, y_test = train_test_split(dftrain.texto, dftrain.sentimento) 

return x_train, x_test, y_train, y_test,dfpred, arq


def lerArquivos(arq1, arq2): #Função para ler os arquivos e ainda tira os stop words de manera recursiva, pela função tirarStopWords
dftrain = pd.read_csv(arq1)
dftrain.head()
dftrain.texto= tirarStopWords(dftrain.texto)
print("Para esse treinamento foram utilizados",len(dftrain.sentimento), "tweets:")
print(dftrain.sentimento.value_counts())
dfpred = pd.read_csv(arq2)
dfpred.head()
dfpred.texto= tirarStopWords(dfpred.texto)

return dftrain, dfpred


def tirarStopWords(lista): #É necessário o download do módulo stop words para exercutar o processo

aux=[]
for i in range(len(lista)):
words = nltk.word_tokenize(lista[i])
newwords = ' '.join([word for word in words if word not in stopwords.words('portuguese')])
aux.append(newwords) 

return aux


def naive_bayes(x_train, x_test, y_train, y_test, dfpred, arq):
clf = Pipeline([('vectorizer', CountVectorizer()), ('nb', MultinomialNB())])
dataset_classificado(dfpred, clf, x_train, y_train, "previsão_naive_bayes"+arq)

return clf.score(x_test, y_test.astype('int'))


def func_svm(x_train, x_test, y_train, y_test, dfpred, arq):
vec = TfidfVectorizer(min_df=5, max_df=0.95, sublinear_tf = True,use_idf = True,ngram_range=(1, 2))
svm_clf =svm.LinearSVC(C=0.1)
vec_clf = Pipeline([('vectorizer', vec), ('pac', svm_clf)])
dataset_classificado(dfpred, vec_clf, x_train, y_train, "previsão_svm"+arq)

return vec_clf.score(x_test, y_test.astype('int'))


def arvore_decisao(x_train, x_test, y_train, y_test, dfpred, arq):
dt = tree.DecisionTreeClassifier(criterion='entropy',random_state = 100,max_depth=3,min_samples_leaf=5)
dt_clf = Pipeline([('vectorizer', CountVectorizer()), ('tree', dt)])
dataset_classificado(dfpred, dt_clf, x_train, y_train, "previsão_árvore_decisão"+arq)

return dt_clf.score(x_test, y_test.astype('int'))


def dataset_classificado(dfpred, clf, x_train, y_train, arq): #Processamento em comum entre os algoritmos analisados, a fim de ter a mesma comparação nos resultados
clf.fit(x_train.values, y_train.astype('int'))
pred = dfpred.texto
clf.predict(pred)
dfpred.sentimento = clf.predict(pred)
dfpred.loc[dfpred['sentimento']==0, 'sentimento'] = 'neutro'
dfpred.loc[dfpred['sentimento']==1, 'sentimento'] = 'ofensivo'
dfpred.loc[dfpred['sentimento']==2, 'sentimento'] = 'positivo'
dfpred.loc[dfpred['sentimento']==3, 'sentimento'] = 'antidemocrático'

print("Esse algoritmo preveu, dentre", len(dfpred.sentimento),"tweets, a quantidade a seguir:")
print(dfpred.sentimento.value_counts())
print()
dfpred.to_csv(arq) #criação do arquivo com o resultado da previsão


def main(): 
print("-----------------------------------------------------------------------")
print("Qual período dos tweets de Jair Bolsonaro você deseja analisar? \n [1] Pré candidatura \n [2] Durante candidatura \n [3] Pós candidatura")
opcao = int(input("Digite apenas uma opção: "))
print("-----------------------------------------------------------------------")

if opcao >= 1 and opcao <= 3:
   
x_train, x_test, y_train, y_test, dfpred, arq = pre_processamento(opcao)

print("-----------------------------------------------------------------------")
print("                             PREDIÇÃO")
print("-----------------------------------------------------------------------")
print("Naive Bayes")
print()
print("Acurácia: ", naive_bayes(x_train, x_test, y_train, y_test, dfpred, arq))
print("-----------------------------------------------------------------------")
print("SVM")
print()
print("Acurácia: ", func_svm(x_train, x_test, y_train, y_test, dfpred, arq))
print("-----------------------------------------------------------------------")
print("Árvore de decisão")
print()
print("Acurácia: ", arvore_decisao(x_train, x_test, y_train, y_test, dfpred, arq))
print("-----------------------------------------------------------------------")
else:
print("Opção inválida") 

main()



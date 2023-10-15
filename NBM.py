import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pickle
import unidecode
import re
import nltk.corpus
from tqdm import tqdm

#stopwords_br = nltk.corpus.stopwords.words('portuguese')

def preProcessing():
    dataFrame = pd.read_csv('geNews.csv', engine = 'python', on_bad_lines = 'skip')
    dataFrame.drop(columns=['date', 'time', 'link'], inplace = True)
    dataFrame['data'] = dataFrame['title'] + ' ' + dataFrame['text']
    
    nmbNews = len(dataFrame['text'])
    print(f'{nmbNews} noticias')
    
    le = LabelEncoder()
    dataFrame['label'] = le.fit_transform(dataFrame['club'])

    clubToLabel = pd.DataFrame({'club': dataFrame['club'].unique(), 'label': dataFrame['label'].unique()})
    distributionDataset(dataFrame)

    dataFrame.drop(columns=['title', 'text', 'club'], inplace = True)

    return dataFrame, clubToLabel

def distributionDataset(dataFrame : pd.read_csv):
    qtdCategories = dataFrame['club'].value_counts()
    plt.figure(figsize = (15,10))
    qtdCategories.plot(kind = 'bar')
    plt.xlabel('Times')
    plt.ylabel('Número de notícias')
    plt.title('Distribuição das notícias por times')
    plt.show()

def houldout(dataFrame):
    df_news_labels = dataFrame[['data', 'label']]

    news = df_news_labels.data
    labels = df_news_labels.label

    """
    for i, d in news.items():
        text = " ".join(word for word in d.split())
        text = unidecode.unidecode(text)
        text = re.sub(r'[^\w$\s]', '', text)
        news.at[i] = text
    """
    
    x_train, x_test, y_train, y_test = train_test_split(news, labels, test_size=0.20)

    return x_train, x_test, y_train, y_test

def BOW(x_train, x_test):
    bow_model = CountVectorizer() #stop_words= stopwords_br)
    x_train = bow_model.fit_transform(x_train)
    x_test = bow_model.transform(x_test)

    return x_train, x_test

def NaiveBayes(x_train, y_train, x_test, y_test):
    model = MultinomialNB()
    model.fit(x_train, y_train)

    train_accuracy = model.score(x_train, y_train)
    test_accuracy = model.score(x_test, y_test)
    
    with open('naive_bayes_model.pkl', 'wb') as file:
        pickle.dump(model, file)

    return model, train_accuracy, test_accuracy

def confusion(model, x_test, y_test, clubToLabel):
    y_pred = model.predict(x_test)

    confusion = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(15, 10))
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=clubToLabel['club'], yticklabels=clubToLabel['club'])
    plt.xlabel('Previsão')
    plt.ylabel('Verdadeiro')
    plt.title('Matriz de Confusão')
    plt.show()

if __name__ == "__main__":
    dataFrame, clubToLabel = preProcessing()
    x_train, x_test, y_train, y_test = houldout(dataFrame)
    x_train, x_test = BOW(x_train, x_test)
    
    model, train_accuracy, test_accuracy = NaiveBayes(x_train, y_train, x_test, y_test)
    
    print("Acuracia do treino:", train_accuracy)
    print("Acuracia do teste:", test_accuracy)

    confusion(model, x_test, y_test, clubToLabel)
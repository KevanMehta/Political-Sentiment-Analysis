# Databricks notebook source
# MAGIC %pip install nltk
# MAGIC %pip install newsapi-python
# MAGIC %pip install kafka-python

# COMMAND ----------

import nltk
nltk.download('punkt')
nltk.download('stopwords')
import numpy as np
import pickle as pkl
from scipy.spatial import distance
import itertools
import re
from nltk.tokenize import *
from nltk.corpus import stopwords
from pyspark.sql import SparkSession
from pyspark.ml.feature import *
import pyspark.sql.functions as pyFunc
from sklearn.metrics import *
import os
from newsapi import NewsApiClient
import requests
import json
import urllib.parse
from kafka import KafkaProducer

# COMMAND ----------

def preProcessing(tweetLabel):
    tweet = tweetLabel[0]
    if tweet == None or len(tweet.split(" ")) == 0:
        return (" ", tweetLabel[1])
    
    # removing username from input sentence
    processed_string = ' '.join(re.sub("(@[a-zA-Z0-9]+)"," ",tweet).split())

    # removing hashtags from input sentence
    processed_string = ' '.join(re.sub("([^a-z0-9A-Z \t])"," ",processed_string).split())

    # removing special characters from input sentence
    processed_string = ' '.join(re.sub("(\w+:\/\/\S+)"," ",processed_string).split()).lower()

    return (processed_string,tweetLabel[1])
 
#create embedings for train data
def getEmbedings(processedData):
    embeding = np.zeros(len(uniqueWords))
    for eachword in word_tokenize(processedData[0]):
        index = uniqueWords.index(eachword) if eachword in uniqueWords else 0
        embeding[index] = embeding[index]+1
    return (embeding, processedData[1])

# creating embedings for test darta
def getTestEmbedings(processedData):
    embedings, uniqueWords, y = loadModelFiles()
    embeding = np.zeros(len(uniqueWords))
    for eachword in word_tokenize(processedData[0]):
        index = uniqueWords.index(eachword) if eachword in uniqueWords else 0
        embeding[index] = embeding[index]+1
    return (embeding, processedData[1])

# function to save the model to use during prediction
def saveModelFiles(embedings, uniqueWords, labels):
    with open("knnEmbeding", "wb") as filePointer:
        pkl.dump(embedings, filePointer)
    with open("knnUniqueWords", "wb") as filePointer:
        pkl.dump(uniqueWords, filePointer)
    with open("labels", "wb") as filePointer:
        pkl.dump(labels, filePointer)

# function to load the model files saved during model creation
def loadModelFiles():
    with open("knnEmbeding", "rb") as filePointer:
        embeding = pkl.load(filePointer)
    with open("knnUniqueWords", "rb") as filePointer:
        knnUniqueWords = pkl.load(filePointer)
    with open("labels", "rb") as filePointer:
        labels = pkl.load(filePointer)
    
    return embeding, knnUniqueWords, labels

# function to compute the metrics
def metrics(groundTruth, predictions):
    precisionScore = precision_score(groundTruth, predictions, average='macro')
    recallScore = recall_score(groundTruth, predictions, average='macro')
    f1Score = f1_score(groundTruth, predictions, average='macro')
    confusionMatrix = confusion_matrix(groundTruth, predictions)
    accuracyVal = accuracy_score(groundTruth, predictions)
    print("Precision " + str(precisionScore))
    print("Recall " + str(recallScore))
    print("F1-Score " + str(f1Score))
    print("Accuracy"+ str(accuracyVal))
    return confusionMatrix

# function to display the confusion matrix
def display_confusion_matrix(confusionMatrix):
    ConfusionMatrixDisplay(confusionMatrix).plot()


# COMMAND ----------

import os
os.environ["news_api_ai_key"] = "76875077-d9ee-4bdb-8f53-da936f099ce9"
os.environ["news_api_app_id"] = "53988099"
os.environ["news_api_app_key"] = "74f5989a6f8bc71efc3970ef6a289f6f"
os.environ["news_api_org"] = "f401be411dec4ea69aa09a0d60abee14"

news_api_id = os.getenv("news_api_app_id")
news_api_key = os.getenv("news_api_app_key")
news_api_url = "https://api.aylien.com/news/stories?text="
news_api_header = {
    "X-AYLIEN-NewsAPI-Application-ID": news_api_id,
    "X-AYLIEN-NewsAPI-Application-Key": news_api_key
}


news_api_ai_key = os.getenv("news_api_ai_key")
news_api_ai_key_url = "http://eventregistry.org/api/v1/article/getArticles"
news_api_ai_key_body = {
    "action": "getArticles",
    "keyword": "leader_name",
    "articlesPage": 1,
    "articlesCount": 10,
    "articlesSortBy": "date",
    "articlesSortByAsc": False,
    "articlesArticleBodyLen": -1,
    "resultType": "articles",
    "dataType": [
        "news",
        "pr"
    ],
    "apiKey": news_api_ai_key,
    "forceMaxDataTimeWindow": 31
}

news_api_org = os.getenv("news_api_org")

news_api_set = set()
news_api_ai_set = set()
news_api_org_set = set()

political_leaders = ["narendra modi", "joe biden", "donald trump", "emmanuel macron",
                     "kamala harris", "xi jinping", "kim jong un", "rishi sunak", "ron desantis"]

newsapi = NewsApiClient(api_key=news_api_org)

producer = KafkaProducer(
    bootstrap_servers=['ec2-3-83-29-213.compute-1.amazonaws.com:9092'],
    value_serializer=lambda m: str(m).encode('utf-8'))

for leader in political_leaders:
    news = []

    top_headlines = newsapi.get_everything(q=leader,
                                           sources='bbc-news,the-verge',
                                           language='en')
    for i in range(len(top_headlines)):
        if not top_headlines['articles'][i]['title'] in news_api_org_set:
            news_api_org_set.add(top_headlines['articles'][i]['title'])
            news.append(top_headlines['articles'][i]['title']+" "+top_headlines['articles'][i]['description'])
    news_api_ai_key_body["keyword"] = leader
    response =  requests.post(news_api_ai_key_url, json=news_api_ai_key_body)
    newsobj =  json.loads(response.text)

    for i in range(len(newsobj)):
        if not newsobj['articles']['results'][i]['uri'] in news_api_ai_set:
            news_api_ai_set.add(newsobj['articles']['results'][i]['uri'])
            news.append(newsobj['articles']['results'][i]['title'])

    url_name = urllib.parse.quote(leader)
    news_api_url_leader = news_api_url + url_name +"&language=en"

    response = requests.get(news_api_url_leader, headers= news_api_header)
    news_api_obj = json.loads(response.text)
    for i in range(len(news_api_obj)):
        if not news_api_obj['stories'][i]['id'] in news_api_set:
            news_api_set.add(news_api_obj['stories'][i]['id'])
            news.append(news_api_obj['stories'][i]['title'])
    
    newRdd = sc.parallelize(zip(news, np.zeros(len(news))))
    sentencesRDD_test = newsRdd
    processedSentences_test = sentencesRDD_test.map(lambda xValues:preProcessing(xValues))
    embedingVectorTest = processedSentences_test.map(lambda xValues:getTestEmbedings(xValues))
    embedingVectorTemp = embedingVectorTest.collect()
    test_embedings = []
    test_y = []

    for embeding in embedingVectorTemp:
        test_embedings.append(embeding[0])
        test_y.append(embeding[1])

    embedings, uniqueWords, y = loadModelFiles()
    distances = distance.cdist(test_embedings, embedings, 'euclidean')
    
    neighborsLabels = []

    for ls in distances:
        sorted_distances_indices = np.argsort(ls)
        knn_indices = []
        knn_indices = list(itertools.islice(sorted_distances_indices,10))
        label = []
        for index in knn_indices:
            label.append(y[index])
        neighborsLabels.append(label)

    predictedLabels = []
    for neighborLabel in neighborsLabels:
        predictedLabels.append(max(set(neighborLabel), key = neighborLabel.count))

    for i in range(len(predictedLabels)):
        item_dict = {"news": news[i], "leader":leader, "knnpolarity" : predictedLabels[i]}
        json_string = json.dumps(item_dict)
        producer.send('my-topic', json_string)


# Databricks notebook source
# MAGIC %pip install nltk
# MAGIC %pip install newsapi-python
# MAGIC %pip install kafka-python

# COMMAND ----------

import re
import nltk
import pickle as pkl
from nltk.tokenize import *
from nltk.corpus import stopwords
from pyspark.sql import SparkSession
from pyspark.ml.feature import *
import pyspark.sql.functions as pyFunc
from sklearn.metrics import *
from pyspark.ml.feature import StopWordsRemover, RegexTokenizer
from sklearn.metrics import *
import os
from newsapi import NewsApiClient
import requests
import json
import urllib.parse
from kafka import KafkaProducer
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')

# COMMAND ----------

class Model:
    def __init__(self, classes, priors, likelihoods):
        self.classes = classes
        self.priors = priors
        self.likelihoods = likelihoods
        
#Implemenatation of Naive Bayes algorithm
class NaiveBayes:
        
    # Transform function to predict the labels
    def transform(self, model, test_df):
        results=[]
        testDF = sc.parallelize(test_df)
        Xrdd = testDF.map(lambda xValues: xValues.features.toArray())
        Y = testDF.map(lambda xValues: xValues.label).collect() 
        
        #map function that computes the predictions
        def map_fun(x):
            local_posterior= {}
            for y in model.classes:
                likelihood = 1
                for i in range(len(x)):
                    if x[i] in model.likelihoods[y][i]:
                        likelihood *= model.likelihoods[y][i][x[i]]
                    else:
                        likelihood = 0
                local_posterior[y]=model.priors[y] * likelihood
            return float(max(local_posterior, key=local_posterior.get))
        
        #for each of the feature labels compute the prediction in a map function
        results = Xrdd.map(lambda xValues: map_fun(xValues)).collect()
        return spark.createDataFrame(list(zip(Y, results)), ["label", "prediction"])
            
    def load_model(self):
        with open("navieBayesModel", "rb") as fp:   # Unpickling
            model = pkl.load(fp)
        return model

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
    
        dummyLabel = [0 for i in range(len(news))]
        newsRdd = sc.parallelize(zip(news, dummyLabel))
        processedSentences_test = newsRdd.map(lambda xValues:preProcessing(xValues))

        tweets_data = spark.createDataFrame(processedSentences_test, schema = ["text", "label"])
        #Tokenize the text to words
        regexTokenizer = RegexTokenizer(inputCol="text", outputCol="words", pattern="\\W")
        regexTokenized_tweets_data = regexTokenizer.transform(tweets_data)

        # COMMAND ----------

        #Remove the stopwords
        from pyspark.ml.feature import StopWordsRemover
        remover = StopWordsRemover(inputCol="words", outputCol="filtered")
        tweets_data_sw_r = remover.transform(regexTokenized_tweets_data)

        # COMMAND ----------

        #Here I use countvectorizer to create the features
        cv = CountVectorizer(inputCol="filtered", outputCol="features")
        tweets_data_cv = cv.fit(tweets_data_sw_r).transform(tweets_data_sw_r)

        nbs = NaiveBayes()
        modelFromFile = nbs.load_model()
        b = nbs.load_model()
        result = nbs.transform(model=b, test_df=tweets_data_cv.collect())

        predictedLabels = result.select("prediction").collect()
        
        for i in range(len(predictedLabels)):
            item_dict = {"news": news[i], "leader":leader, "naviesBayesPolarity" : predictedLabels[i]}
            json_string = json.dumps(item_dict)
            producer.send('my-topic', json_string)
            print(item_dict)

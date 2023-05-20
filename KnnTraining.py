# Databricks notebook source
# MAGIC %pip install nltk

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
 
# create embedings for train data
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

# MAGIC %sh curl https://bigdataproject-s23.s3.amazonaws.com/collectivetweets.csv --output /tmp/tweets.csv

# COMMAND ----------

# loading the data from file and converting polarity to positive and negative sentiments
words = []
processedSentences = []

dbutils.fs.mv("file:/tmp/tweets.csv", "dbfs:/tmp/tweets.csv")
dataFromFile = spark.read.option("header",True).csv("/tmp/tweets.csv")
rawData = dataFromFile.select("Tweet","Polarity")
rawData = rawData.withColumn("label", pyFunc.when(rawData.Polarity < 0.0, 0).otherwise(1))
rawData.display()

# COMMAND ----------

# splitting data into train and test 
train_sentences, test_sentences = rawData.select('Tweet','label').randomSplit([0.7,0.3],seed=100)
train_sentences = rawData.select("Tweet", "label").rdd.map(lambda xValues: xValues).collect()


# preparing the training data and creating the unique words list to create embeding
train_rdd = sc.parallelize(train_sentences)
train_rdd.collect()
cleaned_train_data = train_rdd.map(lambda xValues:preProcessing(xValues))
uniqueWords = cleaned_train_data.flatMap(lambda xValues:xValues[0].split(" ")).distinct()
uniqueWords = uniqueWords.collect()

# COMMAND ----------

# creating word embeding for train data and saving these embedings

embedingVector = cleaned_train_data.map(lambda xValues:getEmbedings(xValues))
embedingVector.collect()

embedingVectorTemp = embedingVector.collect()
embedings = []
y = []

for embeding in embedingVectorTemp:
    embedings.append(embeding[0])
    y.append(embeding[1])

saveModelFiles(embedings, uniqueWords, y)

# COMMAND ----------

embedings, uniqueWords, y = loadModelFiles()

distances = distance.cdist(embedings, embedings, 'euclidean')

# COMMAND ----------

# predicting the labels based on the number of neighbours specified
neighborsLabels = []

for ls in distances:
    sorted_distances_indices = np.argsort(ls)
    knn_indices = []
    knn_indices = list(itertools.islice(sorted_distances_indices,8))
    label = []
    for index in knn_indices:
        label.append(y[index])
    neighborsLabels.append(label)

predictedLabels = []
for neighborLabel in neighborsLabels:
    predictedLabels.append(max(set(neighborLabel), key = neighborLabel.count))

# COMMAND ----------

# test data
test_sentences = test_sentences.select("Tweet", "label").rdd.map(lambda xValues: xValues).collect()
sentencesRDD_test = sc.parallelize(test_sentences)
sentencesRDD_test.collect()
processedSentences_test = sentencesRDD_test.map(lambda xValues:preProcessing(xValues))

embedingVectorTest = processedSentences_test.map(lambda xValues:getEmbedings(xValues))
# embedingVectorTest.collect()

embedingVectorTemp = embedingVectorTest.collect()
test_embedings = []
test_y = []

for embeding in embedingVectorTemp:
    test_embedings.append(embeding[0])
    test_y.append(embeding[1])

embedings, uniqueWords, y = loadModelFiles()
distances = distance.cdist(test_embedings, embedings, 'euclidean')


# COMMAND ----------

# predicting the labels based on the number of neighbours specified
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

# COMMAND ----------

confusionMatrix = metrics(test_y, predictedLabels)

# COMMAND ----------

display_confusion_matrix(confusionMatrix)

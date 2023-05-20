# Databricks notebook source
# MAGIC %pip install nltk

# COMMAND ----------

import re
import nltk
import pickle as pkl
from nltk.tokenize import *
from nltk.corpus import stopwords
from pyspark.sql import SparkSession
from pyspark.ml.feature import *
from sklearn.metrics import *
import pyspark.sql.functions as pyFunc
from pyspark.ml.evaluation import BinaryClassificationEvaluator

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
    
    #Fit function to train the model
    def fit(self, train_df):
        trainDf = sc.parallelize(train_df)
        
        # Extract feature and label column
        X = trainDf.map(lambda xValue: xValue.features.toArray()).collect()
        Y = trainDf.map(lambda xValue: xValue.label).collect() 
        
        # Find the count of the labels and create a rdd
        mapped_rdd = trainDf.toDF().select("label").rdd.map(lambda a: (a, 1)).reduceByKey(lambda a, b: a+b).map(lambda x:  (x[0]['label'], x[1]))
        # Find all the classes, works with both binary and non binary classes  
        classes = mapped_rdd.collectAsMap()
        # Total of label
        totals = mapped_rdd.map(lambda xValue: xValue[1]).reduce(lambda a, b: a+b)
        #Count priors which is labelcount/totalcount
        fractions = mapped_rdd.map(lambda xValues:(xValues[0], xValues[1]/totals))
        priors = fractions.collectAsMap()
        likelihoods ={}
        
        #Compute the likelihood of the classes
        for y in classes:
            likelihoods[y] = {}
            for i in range(len(X[0])):
                likelihoods[y][i] = {}
                values = set([X[j][i] for j in range(len(X))])
                for v in values:
                    count = 0
                    for j in range(len(X)):
                        if X[j][i] == v and Y[j] == y:
                            count += 1
                    likelihoods[y][i][v] = float(count) / classes[y]
        #return the model that has class, priors and likelihood
        return Model(classes=classes, priors=priors, likelihoods=likelihoods)
        
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
    
    def save_model(self, model):
        with open("navieBayesModel", "wb") as filePointer:
            pkl.dump(model, filePointer)
            
    def load_model(self):
        with open("navieBayesModel", "rb") as filePointer:
            model = pkl.load(filePointer)
        return model
    
    def metrics(self, result):
        groundTruth = result.select("label").collect()
        predictions = result.select("prediction").collect()
        precisionScore = precision_score(groundTruth, predictions, average='macro')
        recallScore = recall_score(groundTruth, predictions, average='macro')
        f1Score = f1_score(groundTruth, predictions, average='macro')
        confusionMatrix = confusion_matrix(groundTruth, predictions)
        print("Precision " + str(precisionScore))
        print("Recall " + str(recallScore))
        print("F1-Score " + str(f1Score))
        print("Accuracy " + str(accuracy))
        
        return confusionMatrix
    
    def display_confusion_matrix(self, confusionMatrix):
        ConfusionMatrixDisplay(confusionMatrix).plot()
        
        


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

# LODAING THE DATA
tweets_data = spark.read.option("header", "true").option("inferSchema", "true").csv("/FileStore/tables/twitter_tweets.csv")
display(tweets_data)

# COMMAND ----------

# PREPROCESING THE DATA
# COMMAND ----------

# #Setting the negatve comment to 1 and positive comment to 0
tweets_data_2 = tweets_data.withColumnRenamed("Tweet","text").withColumnRenamed("Polarity","label")
tweets_data_2 = tweets_data_2.withColumn("label", pyFunc.when(tweets_data_2.label < 0.0, 0).otherwise(1))

tweets_data_rdd = tweets_data_2.rdd.map(lambda xValues: (xValues["text"],xValues["label"]))
tweets_data_rdd_cleaned = tweets_data_rdd.map(lambda xValues:preProcessing(xValues))

tweets_data_2 = spark.createDataFrame(tweets_data_rdd_cleaned, schema = ["text", "label"])

# COMMAND ----------

#Tokenize the text to words
regexTokenizer = RegexTokenizer(inputCol="text", outputCol="words", pattern="\\W")
regexTokenized_tweets_data = regexTokenizer.transform(tweets_data_2)

# COMMAND ----------

#Remove the stopwords
# from pyspark.ml.feature import StopWordsRemover
remover = StopWordsRemover(inputCol="words", outputCol="filtered")
tweets_data_sw_r = remover.transform(regexTokenized_tweets_data)

# COMMAND ----------

#Here I use countvectorizer to create the features
cv = CountVectorizer(inputCol="words", outputCol="features")
tweets_data_cv = cv.fit(tweets_data_sw_r).transform(tweets_data_sw_r)

# COMMAND ----------

display(tweets_data_cv)

# COMMAND ----------

#Just taking the relevant columns from the dataframe and create test and train dataset
train_df, test_df = tweets_data_cv.select('features','label').randomSplit([0.7,0.3],seed=100)

# COMMAND ----------

display(train_df)

# COMMAND ----------

#Initialzie the instance of Naive Bayes and train teh model using fit function
nbs = NaiveBayes()
model = nbs.fit(train_df.collect())

# saving our model
nbs.save_model(model)

# COMMAND ----------

#Obtain the result
modelFromFile = nbs.load_model()
b = nbs.load_model()
result = nbs.transform(model=b, test_df=test_df.collect())

evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction")
accuracy = evaluator.evaluate(result)


# COMMAND ----------

confusionMatrix = nbs.metrics(result)

# COMMAND ----------

nbs.display_confusion_matrix(confusionMatrix)

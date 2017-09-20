#!/usr/bin/python
#-*- coding: utf-8 -*-
from sklearn import linear_model
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime
from TagsTrain import TagsTrain
import csv
import sys

def bayesianARDRegression(X_train, Y_train, X_validation):

    bayesianARDRegression = linear_model.ARDRegression()
    bayesianARDRegression.fit(X_train, Y_train)
    predictions = bayesianARDRegression().predict(X_validation)

    bayesianARDRegressionOutputFile = open('bayesianARDRegresionOutput.txt', 'w')
    bayesianARDRegressionOutputFile.write('tag_id, click_count' + '\n')
    index = 0

    for prediction in predictions:
        bayesianARDRegressionOutputFile.write(str(tags[index].tag_id) + ', ' +  str(prediction) + '\n')
        index += 1

def linearRegression(X_train, Y_train, X_validation):

    lmlinearRegression = linear_model.LinearRegression()
    lmlinearRegression.fit(X_train, Y_train)
    predictions = lmlinearRegression.predict(X_validation)

    lmlinearRegressionOutputFile = open('lmlinearRegresionOutput.txt', 'w')
    lmlinearRegressionOutputFile.write('tag_id, click_count' + '\n')
    index = 0

    for prediction in predictions:
        lmlinearRegressionOutputFile.write(str(tags[index].tag_id) + ', ' +  str(prediction) + '\n')
        index += 1

def logisticRegression(X_train, Y_train, X_validation):
    lmlr = linear_model.LogisticRegression()
    lmlr.fit(X_train, Y_train)
    predictions = lmlr.predict(X_validation)

    lmlrOutputFile = open('lmlrOutput.txt', 'w')
    lmlrOutputFile.write('tag_id, click_count' + '\n')
    index = 0

    for prediction in predictions:
        lmlrOutputFile.write(str(tags[index].tag_id) + ', ' +  str(prediction) + '\n')
        index += 1

def kNeighbors(X_train, Y_train, X_validation):

    knr = KNeighborsClassifier()
    knr.fit(X_train, Y_train)
    predictions = knr.predict(X_validation)

    knrOutputFile = open('knrOutput.txt', 'w')
    knrOutputFile.write('tag_id, click_count' + '\n')
    index = 0

    for prediction in predictions:
        knrOutputFile.write(str(tags[index].tag_id) + ', ' +  str(prediction) + '\n')
        index += 1

reload(sys)
sys.setdefaultencoding('UTF8')

tags_file = open("tags_train.csv", "r")
tags_file.readline()
tags_reader = csv.reader(tags_file, delimiter=';', quotechar='"')
tags = []

for tag_row in tags_reader:
    tags.append(TagsTrain(tag_row[0], tag_row[4], tag_row[5], tag_row[6]))

X = []
Y = []

for tag in tags:
    datetime = datetime.strptime(tag.date, '%Y-%m-%d')
    dateNumber = int(10000 * datetime.year + 100 * datetime.month + datetime.day)
    vectorColumn = [dateNumber, int(tag.color)]
    X.append(vectorColumn)
    Y.append(tag.clicks)

validation_size = 0
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

print "Processing..."

tags_file = open("tags_test.csv", "r")
tags_file.readline()
tags_reader = csv.reader(tags_file, delimiter=';', quotechar='"')
tags = []


for tag_row in tags_reader:
    tags.append(TagsTrain(tag_row[1], tag_row[4], tag_row[5], 0))

for tag in tags:
    datetime = datetime.strptime(tag.date, '%Y-%m-%d')
    dateNumber = int(10000 * datetime.year + 100 * datetime.month + datetime.day)
    vectorColumn = [dateNumber, int(tag.color)]
    X_validation.append(vectorColumn)

#linearRegression(X_train, Y_train, X_validation)
bayesianARDRegression(X_train, Y_train, X_validation)








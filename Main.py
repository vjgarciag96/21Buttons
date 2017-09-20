#!/usr/bin/python
#-*- coding: utf-8 -*-
from sklearn import linear_model
from sklearn import gaussian_process
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime
from TagsTrain import TagsTrain
import csv
import sys

def parseUsers():
    users = dict()
    users_file = open("users.csv", "r")
    users_file.readline()
    users_reader = csv.reader(users_file, delimiter=';', quotechar='"')
    for user_row in users_reader:
        user = dict()
        user["id"] = user_row[0]
        user["date"] = user_row[1]
        user["country"] = user_row[2]
        users[user_row[0]] = user
    print "...users parsed..."
    return users

#--------------gaussian models---------------------------#
def gaussianProcessClassifier(X_train, Y_train, X_validation):
    gaussianProcessClassifier = gaussian_process.GaussianProcessClassifier()
    gaussianProcessClassifier.fit(X_train, Y_train)
    predictions = gaussianProcessClassifier.predict(X_validation)

    gaussianProcessClassifierOutputFile = open('gaussianProcessClassifierOutput.txt', 'w')
    gaussianProcessClassifierOutputFile.write('tag_id, click_count' + '\n')
    index = 0

    for prediction in predictions:
        gaussianProcessClassifierOutputFile.write(str(tags[index].tag_id) + ', ' +  str(prediction) + '\n')
        index += 1


#---------------end of gaussian models------------------#

#---------------linear models---------------------------
def ortogonalMP(X_train, Y_train, X_validation):
    ortogonalMP = linear_model.OrthogonalMatchingPursuit()
    ortogonalMP.fit(X_train, Y_train)
    predictions = ortogonalMP.predict(X_validation)

    ortogonalMPOutputFile = open('ortogonalMPOutput.txt', 'w')
    ortogonalMPOutputFile.write('tag_id, click_count' + '\n')
    index = 0

    for prediction in predictions:
        ortogonalMPOutputFile.write(str(tags[index].tag_id) + ', ' +  str(prediction) + '\n')
        index += 1

def passiveAggressiveClassifier(X_train, Y_train, X_validation):
    passiveAggressiveClassifier = linear_model.PassiveAggressiveClassifier()
    passiveAggressiveClassifier.fit(X_train, Y_train)
    predictions = passiveAggressiveClassifier.predict(X_validation)

    passiveAggressiveClassifierOutputFile = open('passiveAggressiveClassifierOutput.txt', 'w')
    passiveAggressiveClassifierOutputFile.write('tag_id, click_count' + '\n')
    index = 0

    for prediction in predictions:
        passiveAggressiveClassifierOutputFile.write(str(tags[index].tag_id) + ', ' +  str(prediction) + '\n')
        index += 1

#WARNING: this method throws Memory Error
def bayesianRidgeRegression(X_train, Y_train, X_validation):

    bayesianRidgeRegression = linear_model.ARDRegression()
    bayesianRidgeRegression.fit(X_train, Y_train)
    predictions = bayesianRidgeRegression.predict(X_validation)

    bayesianRidgeRegressionOutputFile = open('bayesianRidgeRegresionOutput.txt', 'w')
    bayesianRidgeRegressionOutputFile.write('tag_id, click_count' + '\n')
    index = 0

    for prediction in predictions:
        bayesianRidgeRegressionOutputFile.write(str(tags[index].tag_id) + ', ' +  str(prediction) + '\n')
        index += 1

#WARNING: this method throws Memory Error
def bayesianARDRegression(X_train, Y_train, X_validation):

    bayesianARDRegression = linear_model.ARDRegression()
    bayesianARDRegression.fit(X_train, Y_train)
    predictions = bayesianARDRegression.predict(X_validation)

    bayesianARDRegressionOutputFile = open('bayesianARDRegresionOutput.txt', 'w')
    bayesianARDRegressionOutputFile.write('tag_id, click_count' + '\n')
    index = 0

    for prediction in predictions:
        bayesianARDRegressionOutputFile.write(str(tags[index].tag_id) + ', ' +  str(prediction) + '\n')
        index += 1

def linearRegression(X_train, Y_train, X_validation, Y_validation):

    lmlinearRegression = linear_model.LinearRegression()
    lmlinearRegression.fit(X_train, Y_train)
    predictions = lmlinearRegression.predict(X_validation)

    lmlinearRegressionOutputFile = open('lmlinearRegresionOutput.txt', 'w')
    lmlinearRegressionOutputFile.write('tag_id, click_count' + '\n')
    index = 0

    for prediction in predictions:
        lmlinearRegressionOutputFile.write(str(tags[index].tag_id) + ', ' +  str(int(prediction)) + '\n')
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
#--------------End of linear models----------------------------#

reload(sys)
sys.setdefaultencoding('UTF8')

users = parseUsers()

countries = dict()
countries["ES"] = 0
countries["IT"] = 1
countries["GB"] = 2

tags_file = open("tags_train.csv", "r")
tags_file.readline()
tags_reader = csv.reader(tags_file, delimiter=';', quotechar='"')
tags = []

for tag_row in tags_reader:
    tag = TagsTrain(tag_row[3], tag_row[0], tag_row[4], tag_row[5], tag_row[6])

    if(users[tag_row[3]] != None):
        country = users[tag_row[3]]["country"]
        tag.setCountries(countries[country])

    tags.append(tag)

X = []
Y = []

for tag in tags:
    datetime = datetime.strptime(tag.date, '%Y-%m-%d')
    dateNumber = int(10000 * datetime.year + 100 * datetime.month + datetime.day)
    vectorColumn = [dateNumber, int(tag.color), tag.isIT, tag.isSP, tag.isGB]
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
    tag = TagsTrain(tag_row[0], tag_row[1], tag_row[4], tag_row[5], 0)

    if (users[tag_row[0]] != None):
        country = users[tag_row[0]]["country"]
        tag.setCountries(countries[country])

    tags.append(tag)

for tag in tags:
    datetime = datetime.strptime(tag.date, '%Y-%m-%d')
    dateNumber = int(10000 * datetime.year + 100 * datetime.month + datetime.day)
    vectorColumn = [dateNumber, int(tag.color), tag.isIT, tag.isSP, tag.isGB]
    X_validation.append(vectorColumn)

#passiveAggressiveClassifier(X_train, Y_train, X_validation)
#linearRegression(X_train, Y_train, X_validation)
#bayesianARDRegression(X_train, Y_train, X_validation)
#bayesianRidgeRegression(X_train, Y_train, X_validation)
#ortogonalMP(X_train, Y_train, X_validation)
gaussianProcessClassifier(X_train, Y_train, X_validation)







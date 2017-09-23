#!/usr/bin/python
#-*- coding: utf-8 -*-
from sklearn import linear_model
from sklearn import gaussian_process
from sklearn import model_selection
from sklearn import naive_bayes
from sklearn import neural_network
from sklearn import tree
from sklearn.pipeline import Pipeline
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

def decisionTree(X_train, Y_train, X_validation):
    decisionTree = tree.DecisionTreeClassifier(random_state=0)
    decisionTree.fit(X_train, Y_train)
    predictions = decisionTree.predict(X_validation)

    decisionTreeOutputFile = open('decisionTreeOutput.txt', 'w')
    decisionTreeOutputFile.write('tag_id, click_count' + '\n')
    index = 0

    for prediction in predictions:
        decisionTreeOutputFile.write(str(tags[index].tag_id) + ', ' +  str(prediction) + '\n')
        index += 1

def decisionTreeRegressor(X_train, Y_train, X_validation):
    decisionTree = tree.DecisionTreeRegressor(random_state=0)
    decisionTree.fit(X_train, Y_train)
    predictions = decisionTree.predict(X_validation)

    decisionTreeOutputFile = open('decisionTreeRegressorOutput.txt', 'w')
    decisionTreeOutputFile.write('tag_id, click_count' + '\n')
    index = 0

    for prediction in predictions:
        decisionTreeOutputFile.write(str(tags[index].tag_id) + ', ' +  str(int(prediction)) + '\n')
        index += 1

def bernouilliRBM(X_train, Y_train, X_validation):

    logistic = linear_model.LogisticRegression()

    bernouilliRBM = neural_network.BernoulliRBM(random_state=0, verbose=True)

    classifier = Pipeline(steps=[('rbm', bernouilliRBM), ('logistic', logistic)])

    bernouilliRBM.learning_rate = 0.06
    bernouilliRBM.n_iter = 20

    bernouilliRBM.n_components = 100
    logistic.C = 6000.0

    classifier.fit(X_train, Y_train)

    logistic_classifier = linear_model.LogisticRegression(C=100.0)
    logistic_classifier.fit(X_train, Y_train)
    predictions = classifier.predict(X_validation)

    bernouilliRBMOutputFile = open('bernouilliRBMOutput.txt', 'w')
    bernouilliRBMOutputFile.write('tag_id, click_count' + '\n')
    index = 0

    for prediction in predictions:
        bernouilliRBMOutputFile.write(str(tags[index].tag_id) + ', ' + str(prediction) + '\n')
        index += 1

def gaussianNB(X_train, Y_train, X_validation):
    gaussianNB = naive_bayes.GaussianNB()
    gaussianNB.fit(X_train, Y_train)
    predictions = gaussianNB.predict(X_validation)

    gaussianNBOutputFile = open('gaussianNBOutput.txt', 'w')
    gaussianNBOutputFile.write('tag_id, click_count' + '\n')
    index = 0

    for prediction in predictions:
        gaussianNBOutputFile.write(str(tags[index].tag_id) + ', ' +  str(prediction) + '\n')
        index += 1

#--------------gaussian models---------------------------#

#WARNING. Throws memory error
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

def linearRegression(X_train, Y_train, X_validation):

    lmlinearRegression = linear_model.LinearRegression()
    lmlinearRegression.fit(X_train, Y_train)
    predictions = lmlinearRegression.predict(X_validation)

    lmlinearRegressionOutputFile = open('lmlinearRegresionOutput.txt', 'w')
    lmlinearRegressionOutputFile.write('tag_id, click_count' + '\n')
    index = 0

    for prediction in predictions:
        if(int(prediction) < 0):
            prediction = 0
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

def datetimeToNumber(datetime):
    month2day = [31,28,31,30,31,30,31,31,30,31,30]
    total_days = datetime.year * 365 + (datetime.year / 4)
    total_days_in_months = 0
    if datetime.month > 1:
        for x in range(0, datetime.month-1):
            total_days_in_months += month2day[x]
            total_days += total_days_in_months
    return total_days + datetime.day

def dateTimeToMonth(datetime):
    return int(datetime.month)

def parseProducts():
    products = dict()
    products_file = open("products.csv", "r")
    products_file.readline()
    products_reader = csv.reader(products_file, delimiter=';', quotechar='"')
    for product_row in products_reader:
        product = dict()
        product["id"] = product_row[0]
        product["info"] = str(product_row[1]).replace('\'', '´').replace("\n", "").upper().strip()
        product["description"] = product_row[2]
        product["brand"] = str(product_row[3]).replace('\'', '´').replace("\n", "").upper().strip()
        products[product_row[0]] = product
    print "...products parsed..."
    return products

def createUserClicksDict(tagsTrain):
    userClicksDict = dict()

    for tag in tagsTrain:
        userId = tag.user_id
        if userId not in userClicksDict:
            userClicksDict[userId] = []
        userClicksDict[userId].append(int(tag.clicks))
    return userClicksDict

def userClicksMean(userClicksDict, users):

    userClicksMeanDict = dict()

    for key, clicks in userClicksDict.items():
        mean = reduce(lambda x, y: x + y, clicks) / len(clicks)
        userClicksMeanDict[key] = int(mean)

    for key, user in users.items():
        if key not in userClicksMeanDict:
            userClicksMeanDict[key] = 0

    return userClicksMeanDict


def createColorClicksDict(tagsTrain):
    colorClicksDict = dict()

    for tag in tagsTrain:
        colorId = tag.color
        if colorId not in colorClicksDict:
            colorClicksDict[colorId] = []
        colorClicksDict[colorId].append(int(tag.clicks))

    return colorClicksDict

def colorClicksMean(colorClicksDict):
    colorClicksMeanDictionary = dict()

    for colorId, colorClicks in colorClicksDict.items():
        clicksMean = reduce(lambda x,y: x+y, colorClicks) / len(colorClicks)
        colorClicksMeanDictionary[colorId] = int(clicksMean)

    return colorClicksMeanDictionary

def createCountryClicksDict(tagsTrain):
    countryClicksDict = dict()

    for tag in tagsTrain:
        countryId = tag.country
        if countryId not in countryClicksDict:
            countryClicksDict[countryId] = []
        countryClicksDict[countryId].append(int(tag.clicks))

    return countryClicksDict

def countryClicksMean(countryClicksDict):
    countryClicksMeanDictionary = dict()

    for countryId, countryClicks in countryClicksDict.items():
        clicksMean = reduce(lambda x, y: x + y, countryClicks) / len(countryClicks)
        countryClicksMeanDictionary[countryId] = int(clicksMean)
    print "---CountryClicksMean---"
    print countryClicksMeanDictionary

    return countryClicksMeanDictionary

def brandClicksMean(brandsClickFrequency, products):

    brandClicksDictionary = dict()

    for brandId, brandClicks in brandsClickFrequency.items():
        clicksMean = reduce(lambda x,y: x+y, brandClicks) / len(brandClicks)
        brandClicksDictionary[brandId] = int(clicksMean)

    for key, product in products.items():
        if product["brand"] not in brandClicksDictionary:
            brandClicksDictionary[product["brand"]] = 0

    return brandClicksDictionary

def productInfoClicksMean(productsClickFrequency, products):

    productInfoClicksDictionary = dict()

    for productId, productClicks in productsClickFrequency.items():
        clicksMean = reduce(lambda x,y: x+y, productClicks) / len(productClicks)
        productInfoClicksDictionary[productId] = int(clicksMean)

    for key, product in products.items():
        if product["brand"] not in productInfoClicksDictionary:
            productInfoClicksDictionary[product["info"]] = 0

    return productInfoClicksDictionary

def productClicksMean(productsClickFrequency, products):

    productClicksDictionary = dict()

    for productId, productClicks in productsClickFrequency.items():
        clicksMean = reduce(lambda x,y: x+y, productClicks) / len(productClicks)
        productClicksDictionary[productId] = int(clicksMean)

    for key, product in products.items():
        if product["brand"] not in productClicksDictionary:
            productClicksDictionary[product["brand"]] = 0

    return productClicksDictionary

'''def getBrandMeanClicks(brandsClickFrequency, products):
    brandClicksMeanDict = dict()

    for key, clicks in userClicksDict.items():
        mean = reduce(lambda x, y: x + y, clicks) / len(clicks)
        userClicksMeanDict[key] = int(mean)

    for key, user in users.items():
        if key not in userClicksMeanDict:
            userClicksMeanDict[key] = 0

    return userClicksMeanDict'''


def getBrandTotalClicks(brandsClickFrequency, products):

    brandTotalClicks = dict()

    for key, values in brandsClickFrequency.items():
        total = 0
        for value in values:
            total += value

        brandTotalClicks[key] = total

    for key, product in products.items():
        if product["brand"] not in brandTotalClicks:
            brandTotalClicks[product["brand"]] = 0

    return brandTotalClicks


def getProductsTotalClicks(tags, products):
    productClicks = dict()
    for tag in tags:
        if tag.product_id not in productClicks:
            productClicks[tag.product_id] = tag.clicks
        else:
            productClicks[tag.product_id] += tag.clicks

    for key, product in products.items():
        if product["brand"] not in productClicks:
            productClicks[product["brand"]] = 0

    return productClicks

def getProductsMedianClicks(tags, products):

    productClicks = dict()
    medianClicks = dict()

    for tag in tags:
        if tag.product_id not in productClicks:
            productClicks[tag.product_id] = []
        productClicks[tag.product_id].append(int(tag.clicks))

    for key, value in productClicks.items():
        orderedClicks = sorted(value)
        if len(orderedClicks) % 2 == 0:
            median = int((orderedClicks[int(len(orderedClicks) / 2 - 1)] + orderedClicks[int(len(orderedClicks) / 2)]) / 2)
        else:
            median = orderedClicks[int(len(orderedClicks) / 2 - 1)]
        medianClicks[key] = median

    for key, product in products.items():
        if product["brand"] not in medianClicks:
            medianClicks[product["brand"]] = 0

    return medianClicks

def getProductsMeanClicks(tags, products):

    productClicks = dict()
    meanClicks = dict()

    for tag in tags:
        if tag.product_id not in productClicks:
            productClicks[tag.product_id] = []
        productClicks[tag.product_id].append(int(tag.clicks))

    for key, values in productClicks.items():
        total = 0
        for value in values:
            total += value
        mean = total / len(values)
        meanClicks[key] = mean

    for key, product in products.items():
        if product["brand"] not in meanClicks:
            meanClicks[product["brand"]] = 0

    return meanClicks

def getUserMeanClicks(tags, products):

    productClicks = dict()
    meanClicks = dict()

    for tag in tags:
        if tag.product_id not in productClicks:
            productClicks[tag.product_id] = []
        productClicks[tag.product_id].append(int(tag.clicks))

    for key, values in productClicks.items():
        total = 0
        for value in values:
            total += value
        mean = total / len(values)
        meanClicks[key] = mean

    for key, product in products.items():
        if product["brand"] not in meanClicks:
            meanClicks[product["brand"]] = 0

    return meanClicks

def shortBrandClassification(products):
    brands_times = dict()
    for tag_key, tag_value in products.items():
            brands_times[tag_value["id"]] = tag_value["brand"]
    return brands_times

def shortInfoClassification(products):
    info_times = dict()
    for tag_key, tag_value in products.items():
            info_times[tag_value["id"]] = tag_value["info"]
    return info_times

def createARFFHeader(arffFile):
    arffFile.write("@RELATION 21buttons" + "\n")
    arffFile.write("\n")

    #arffFile.write("@ATTRIBUTE datenumber NUMERIC" + "\n")


    '''arffFile.write("@ATTRIBUTE italian NUMERIC" + "\n")
    arffFile.write("@ATTRIBUTE spanish NUMERIC" + "\n")
    arffFile.write("@ATTRIBUTE britain NUMERIC" + "\n")
    arffFile.write("@ATTRIBUTE userdate NUMERIC" + "\n")

    arffFile.write("@ATTRIBUTE color0 NUMERIC" + "\n")
    arffFile.write("@ATTRIBUTE color1 NUMERIC" + "\n")
    arffFile.write("@ATTRIBUTE color2 NUMERIC" + "\n")
    arffFile.write("@ATTRIBUTE color3 NUMERIC" + "\n")
    arffFile.write("@ATTRIBUTE color4 NUMERIC" + "\n")
    arffFile.write("@ATTRIBUTE color5 NUMERIC" + "\n")
    arffFile.write("@ATTRIBUTE colorclickmean NUMERIC" + "\n")'''

    arffFile.write("@ATTRIBUTE productclickmean NUMERIC" + "\n")
    arffFile.write("@ATTRIBUTE userclickmean NUMERIC" + "\n")
    arffFile.write("@ATTRIBUTE brandclickmean NUMERIC" + "\n")
    arffFile.write("@ATTRIBUTE countryclickmean NUMERIC" + "\n")
    arffFile.write("@ATTRIBUTE infoclickmean NUMERIC" + "\n")
    arffFile.write("@ATTRIBUTE months NUMERIC" + "\n")
    arffFile.write("@ATTRIBUTE clicks NUMERIC" + "\n")

def createARFFData(arffFile, X, Y):
    index = 0

    for item in X:
        for value in item:
            arffFile.write(str(value) + ", ")

        arffFile.write(str(Y[index]) + "\n")
        index += 1

def createARFFFile(X, Y):
    print "creating ARFF File"
    arffFile = open('wekaInput.arff', 'w')
    createARFFHeader(arffFile)
    arffFile.write("@DATA" + "\n")
    createARFFData(arffFile, X, Y)
    print "ARFF created"




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
tagsTrain = []

for tag_row in tags_reader:
    country = users[tag_row[3]]["country"]
    userDate = users[tag_row[3]]["date"]
    datetimeFormated = datetime.strptime(userDate, '%Y-%m-%d')
    dateNumber = datetimeToNumber(datetimeFormated)

    tag = TagsTrain(tag_row[3], tag_row[0], tag_row[2], tag_row[4], dateNumber,tag_row[5], tag_row[6])

    tag.setCountries(countries[country])

    tagsTrain.append(tag)



X = []
Y = []

products = parseProducts()
productsToBrands =  shortBrandClassification(products)
productsToInfo = shortInfoClassification(products)

brandsClickFrequency = dict()
productClickFrequency = dict()

for tag in tagsTrain:
    productId = tag.product_id
    brand = productsToBrands[productId]
    info = productsToInfo[productId]
    if brand not in brandsClickFrequency:
        brandsClickFrequency[brand] = []
    if info not in productClickFrequency:
        productClickFrequency[info] = []
    brandsClickFrequency[brand].append(int(tag.clicks))

brand2median = dict()

for key, value in brandsClickFrequency.items():
    orderedClicks = sorted(value)
    if len(orderedClicks) % 2 == 0:
        median = int((orderedClicks[int(len(orderedClicks)/2 - 1)] + orderedClicks[int(len(orderedClicks)/2)])/2)
    else:
        median = orderedClicks[int(len(orderedClicks)/2 - 1)]
    brand2median[key] = median

productsTotalClicks = getProductsTotalClicks(tagsTrain, products)
productsMedianClicks = getProductsMedianClicks(tagsTrain, products)
productsMeanClicks = getProductsMeanClicks(tagsTrain, products)
brandTotalClicks = getBrandTotalClicks(brandsClickFrequency, products)


userClicksDictionary = createUserClicksDict(tagsTrain)
userClicksMeanDictionary = userClicksMean(userClicksDictionary, users)

colorClicksDictionary = createColorClicksDict(tagsTrain)
colorClicksMeanDictionary = colorClicksMean(colorClicksDictionary)

countryClicksDictionary = createCountryClicksDict(tagsTrain)
countryClicksMeanDictionary = countryClicksMean(countryClicksDictionary)

productClicksMeanDictionary = productClicksMean(productClickFrequency)
brandClicksMeanDictionary = brandClicksMean(brandsClickFrequency, products)

for tag in tagsTrain:
    datetime_formated = datetime.strptime(tag.date, '%Y-%m-%d')
    month = dateTimeToMonth(datetime_formated)
    vectorColumn = [productsMeanClicks[tag.product_id],
                    userClicksMeanDictionary[tag.user_id],
                    brandClicksMeanDictionary[productsToBrands[tag.product_id]],
                    countryClicksMeanDictionary[tag.country],
                    productClicksMeanDictionary[productsToInfo[tag.product_id]],
                    month]
    X.append(vectorColumn)
    Y.append(tag.clicks)

createARFFFile(X, Y)

validation_size = 0
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

print "Processing..."

tags_file = open("tags_test.csv", "r")
tags_file.readline()
tags_reader = csv.reader(tags_file, delimiter=';', quotechar='"')
tags = []


for tag_row in tags_reader:
    country = users[tag_row[0]]["country"]
    userDate = users[tag_row[0]]["date"]
    datetimeFormated = datetime.strptime(userDate, '%Y-%m-%d')
    dateNumber = datetimeToNumber(datetimeFormated)

    tag = TagsTrain(tag_row[0], tag_row[1], tag_row[3], tag_row[4], dateNumber, tag_row[5], 0)
    tag.setCountries(countries[country])

    tags.append(tag)

for tag in tags:
    datetime_formated = datetime.strptime(tag.date, '%Y-%m-%d')
    dateNumber = datetimeToNumber(datetime_formated)
    month = dateTimeToMonth(datetime_formated)

    if tag.product_id in productsMeanClicks:
        vectorColumn = [productsMeanClicks[tag.product_id],
                        userClicksMeanDictionary[tag.user_id],
                        brandClicksMeanDictionary[productsToBrands[tag.product_id]],
                        countryClicksMeanDictionary[tag.country],
                        productClicksMeanDictionary[productsToInfo[tag.product_id]],
                        month]
    else:
        vectorColumn = [0,
                        userClicksMeanDictionary[tag.user_id],
                        brandClicksMeanDictionary[productsToBrands[tag.product_id]],
                        countryClicksMeanDictionary[tag.country],
                        productClicksMeanDictionary[productsToInfo[tag.product_id]],
                        month]
    X_validation.append(vectorColumn)

#passiveAggressiveClassifier(X_train, Y_train, X_validation)
linearRegression(X_train, Y_train, X_validation)
#bayesianARDRegression(X_train, Y_train, X_validation)
#bayesianRidgeRegression(X_train, Y_train, X_validation)
#ortogonalMP(X_train, Y_train, X_validation)
#gaussianProcessClassifier(X_train, Y_train, X_validation)
#gaussianNB(X_train, Y_train, X_validation)
#bernouilliRBM(X_train, Y_train, X_validation)
#decisionTreeRegressor(X_train, Y_train, X_validation)
#logisticRegression(X_train, Y_train, X_validation)







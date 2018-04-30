import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm, preprocessing
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score
from sklearn.decomposition import TruncatedSVD
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from collections import Counter
from knn import KNN


def SVDgraph(trainData, vectorizer, freqVecTrain, trainText, categoryIds, le):
    clf = RandomForestClassifier()
    xAxes, yAxes = [], []
    splits = 10

    # this value denotes the accuracy without truncation
    scores = cross_val_score(clf, freqVecTrain, categoryIds, cv=splits, scoring='accuracy')
    xAxes.append(0)
    yAxes.append(scores.mean())

    # Try truncation with multiple components
    for comp in range(10, 1000, 100):
        print comp

        svd = TruncatedSVD(n_components=comp)
        svdFreqVecTrain = svd.fit_transform(freqVecTrain)

        scores = cross_val_score(clf, svdFreqVecTrain, categoryIds, cv=splits, scoring='accuracy')
        '''
        kf = KFold(n_splits=splits)
        accur = 0

        for trainIndex, testIndex in kf.split(trainData):
            kfFreqVecTrain = vectorizer.transform(np.array(trainText)[trainIndex])
            kfFreqVecTest = vectorizer.transform(np.array(trainText)[testIndex])

            # Truncation
            svd = TruncatedSVD(n_components=comp)
            kfFreqVecTrain = svd.fit_transform(kfFreqVecTrain)
            kfFreqVecTest = svd.fit_transform(kfFreqVecTest)

            kfTestCategIds = le.transform(trainData['Category'][testIndex])

            clf.fit(kfFreqVecTrain, categoryIds[trainIndex])
            kfTestPredIds = clf.predict(kfFreqVecTest)

            accur += accuracy_score(kfTestCategIds, kfTestPredIds)
            print accur

        # Mean accuracy of all folds
        accur /= float(splits)
        '''

        xAxes.append(comp)
        yAxes.append(scores.mean())

    plt.plot(xAxes, yAxes)
    plt.savefig('static/' + 'SVDgraph_RandomForest.png')


def crossValidation(info, clf, name, trainData, vectorizer, freqVecTrain, trainText, categoryIds, le):
    splits = 10
    scores = cross_val_score(clf, freqVecTrain, categoryIds, cv=splits, scoring='accuracy')
    print 'Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2)
    
    kf = KFold(n_splits=splits)
    prec, rec, f1, accur = 0, 0, 0, 0
 
    # Splits data to train and test data
    for trainIndex, testIndex in kf.split(trainData):
        kfFreqVecTrain = vectorizer.transform(np.array(trainText)[trainIndex])
        kfFreqVecTest = vectorizer.transform(np.array(trainText)[testIndex])

        kfTestCategIds = le.transform(trainData['Category'][testIndex])

        clf.fit(kfFreqVecTrain, categoryIds[trainIndex])
        kfTestPredIds = clf.predict(kfFreqVecTest)

        ret = precision_score(kfTestCategIds, kfTestPredIds, average=None)
        prec += float(sum(ret)) / float(len(ret))

        ret = recall_score(kfTestCategIds, kfTestPredIds, average=None)
        rec += float(sum(ret)) / float(len(ret))

        ret = f1_score(kfTestCategIds, kfTestPredIds, average=None)
        f1 += float(sum(ret)) / float(len(ret))

        accur += accuracy_score(kfTestCategIds, kfTestPredIds)

    prec, rec, f1, accur = prec / float(splits), rec / float(splits), f1 / float(splits), accur / float(splits)
    info[name].extend([accur, prec, rec, f1])

def getPos(word):
    wSynsets = wordnet.synsets(word)

    posCounts = Counter()
    posCounts['n'] = len([item for item in wSynsets if item.pos() == 'n'])
    posCounts['v'] = len([item for item in wSynsets if item.pos() == 'v'])
    posCounts['a'] = len([item for item in wSynsets if item.pos() == 'a']  )
    posCounts['r'] = len([item for item in wSynsets if item.pos() == 'r']  )
    
    mostCommonPosList = posCounts.most_common(1)
    return mostCommonPosList[0][0]


def processText(text):
    stopWords = stopwords.words('english')
    stopWords.extend(['saying', 'said', 'say', 'yes', 'instead', 'meanwhile', 'right', 'really', 'finally', 'now', 
                       'one', 'suggested', 'says', 'added', 'think', 'know', 'though', 'let', 'going', 'back',
                       'well', 'example', 'us', 'yet', 'perhaps', 'actually', 'oh', 'year', 'lastyear',
                       'last', 'old', 'first', 'good', 'maybe', 'ask', '.', ',', ':', 'take', 'made', 'n\'t', 'go', 
                       'make', 'two', 'got', 'took', 'want', 'much', 'may', 'never', 'second', 'could', 'still', 'get', 
                       '?', 'would', '(', '\'', ')', '``', '/', "''", '%', '#', '!', 'next', "'s", ';', '[', ']', '...',
                       'might', "'m", "'d", 'also', 'something', 'even', 'new', 'lot', 'a', 'thing', 'time', 'way',
                       'always', 'whose', 'need', 'people', 'come', 'become', 'another', 'many', 'must', 'too', 'as', 'well'])
    
    tokens = word_tokenize(text)
    
    # remove stopWords
    tokens = [wrd for wrd in tokens if (wrd not in stopWords) and 
                                    (not any(ltr.isdigit() for ltr in wrd))]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t, getPos(t)) for t in tokens]
    
    # remove stopWords again after lemmatization
    tokens = [wrd for wrd in tokens if wrd not in stopWords]
    
    stemmer = SnowballStemmer('english')
    tokens = [stemmer.stem(t) for t in tokens]

    return tokens


def classifiers(trainData, testData, clfNames):
    #trainData = trainData[:1000]
    #testData = testData[:2]
    
    # Labels for categories
    le = preprocessing.LabelEncoder()
    categoryIds = le.fit_transform(trainData['Category'])

    # Include in text content and title as well
    trainText, testText = [], []

    # pos=3 is content, pos=2 is title
    for elem in np.array(trainData):
        text = elem[3] + elem[2] * (1 + int(len(elem[3]) / 200))
        text = ''.join([i if ord(i) < 128 else ' ' for i in text])
        trainText.append(text.lower())

    for elem in np.array(testData):
        text = elem[3] + elem[2] * (1 + int(len(elem[3]) / 200))
        text = ''.join([i if ord(i) < 128 else ' ' for i in text])
        testText.append(text.lower())
    
    # Vectorization
    vectorizer = TfidfVectorizer(tokenizer=processText, stop_words='english', max_df=0.9, min_df=3).fit(trainText)
    freqVecTrain = vectorizer.transform(trainText)
    freqVecTest = vectorizer.transform(testText)
    clfs = []
    
    # Truncation
    #SVDgraph(trainData, vectorizer, freqVecTrain, trainText, categoryIds, le)
    #svd = TruncatedSVD(n_components=20)
    #freqVecTrain = svd.fit_transform(freqVecTrain)
    #freqVecTest = svd.fit_transform(freqVecTest)

    # Add classifiers that belong to 'clfNames' list
    if 'SVC' in clfNames:
        parameters = [
            {'C': [0.1, 10, 100], 'kernel': ['linear']},
            {'C': [0.1, 10, 100], 'gamma': [0.001, 0.1, 10], 'kernel': ['rbf']},
        ]

        clfSvc = svm.SVC()
        clf = GridSearchCV(clfSvc, parameters)
        clfs.append((clf, 'SVM'))

    if 'RandomForest' in clfNames:
        clf = RandomForestClassifier()
        clfs.append((clf, 'Random Forest'))

    if 'MultinomialNB' in clfNames:
        clf = MultinomialNB(alpha=0.001)
        clfs.append((clf, 'Naive Bayes'))

    if 'KNN' in clfNames:
        clf = KNN(2)
        clfs.append((clf, 'KNN'))

    if 'My Method' in clfNames:
        clf = svm.SVC(kernel='linear')
        clfs.append((clf, 'My Method'))

    # Information for the csv file
    info = {
        'Statistic Measure': ['Accuracy', 'Precision', 'Recall', 'F-Measure'],
        'Naive Bayes': [],
        'Random Forest': [],
        'SVM': [],
        'KNN': [],
        'My Method': []
    }

    # Run classifiers
    for clf, name in clfs:
        crossValidation(info, clf, name, trainData, vectorizer, freqVecTrain, trainText, categoryIds, le)

        # Train with real trainSet and predict for real testSet
        clf.fit(freqVecTrain, categoryIds)

        predIds = clf.predict(freqVecTest)
        predCategs = le.inverse_transform(predIds)

        # Print predicted output to csv file
        if len(clfs) == 1:
            predInfo = {
                'Id': [],
                'Category': []
            }

            testDataArr = np.array(testData)

            # pos=1 is id
            for i in range(len(testDataArr)):
                predInfo['Id'].append(testDataArr[i][1])
                predInfo['Category'].append(predCategs[i])

            df = pd.DataFrame(predInfo, columns=['ID', 'Predicted_Category'])
            df.to_csv('testSet_categories.csv', sep='\t', index=False)

    # Print output to csv file
    if len(clfs) == 5:
        df = pd.DataFrame(info, columns=['Statistic Measure', 'Naive Bayes', 'Random Forest', 'SVM', 'KNN', 'My Method'])
        df.to_csv('EvaluationMetric_10fold.csv', sep='\t', index=False)

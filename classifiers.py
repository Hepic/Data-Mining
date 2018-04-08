import numpy as np
import pandas as pd
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
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from collections import Counter


def crossValidation(info, clf, name, trainData, vectorizer, freqVecTrain, trainText, categoryIds, le):
    scores = cross_val_score(clf, freqVecTrain, categoryIds, cv=10)
    print 'Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2)
'''
    kf = KFold(n_splits=10)
    fold = 0
    prec, rec, f1, accur = 0, 0, 0, 0

    for trainIndex, testIndex in kf.split(trainData):
        kfFreqVecTrain = vectorizer.transform(np.array(trainText)[trainIndex])
        kfFreqVecTest = vectorizer.transform(np.array(trainText)[testIndex])

        kfTestCategIds = le.transform(trainData['Category'][testIndex])

        clf.fit(kfFreqVecTrain, categoryIds[trainIndex])
        kfTestPredIds = clf.predict(kfFreqVecTest)
        fold += 1

        #print "Fold " + str(fold)
        #print classification_report(kfTestCategIds, kfTestPredIds, target_names=list(le.classes_))

        ret = precision_score(kfTestCategIds, kfTestPredIds, average=None)
        prec += float(sum(ret)) / float(len(ret))

        ret = recall_score(kfTestCategIds, kfTestPredIds, average=None)
        rec += float(sum(ret)) / float(len(ret))

        ret = f1_score(kfTestCategIds, kfTestPredIds, average=None)
        f1 += float(sum(ret)) / float(len(ret))

        accur += accuracy_score(kfTestCategIds, kfTestPredIds)

    prec, rec, f1, accur = prec / 10.0, rec / 10.0, f1 / 10.0, accur / 10.0
    info[name].extend([accur, prec, rec, f1])
'''

def getPos(word):
    wSynsets = wordnet.synsets(word)

    pos_counts = Counter()
    pos_counts['n'] = len([item for item in wSynsets if item.pos() == 'n'])
    pos_counts['v'] = len([item for item in wSynsets if item.pos() == 'v'])
    pos_counts['a'] = len([item for item in wSynsets if item.pos() == 'a']  )
    pos_counts['r'] = len([item for item in wSynsets if item.pos() == 'r']  )

    mostCommonPosList = pos_counts.most_common(3)
    return mostCommonPosList[0][0]


def processText(text):
    stopWords = stopwords.words('english')
    tokens = word_tokenize(text)

    # remove stop-words
    tokens = [wrd for wrd in tokens if wrd not in stopWords]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t, getPos(t)) for t in tokens]

    stemmer = PorterStemmer()
    tokens = [stemmer.stem(t) for t in tokens]

    return tokens


def classifiers(trainData, testData, clfNames):
    trainData = trainData[:500]
    testData = testData[:10]

    # Labels for categories
    le = preprocessing.LabelEncoder()
    categoryIds = le.fit_transform(trainData['Category'])

    # Include in text content and title as well
    trainText, testText = [], []

    # pos=3 is content, pos=2 is title
    for elem in np.array(trainData):
        trainText.append(elem[3] + elem[2])
        trainText[-1] += elem[2] * int(len(elem[3]) / 200)

    for elem in np.array(testData):
        testText.append(elem[3] + elem[2])
        testText[-1] += elem[2] * int(len(elem[3]) / 200)

    # Vectorization
    vectorizer = TfidfVectorizer(tokenizer=processText, ngram_range=(1, 2), stop_words='english').fit(trainText)
    freqVecTrain = vectorizer.transform(trainText)
    freqVecTest = vectorizer.transform(testText)
    clfs = []

    # Truncation
    svd = TruncatedSVD(n_components=10)
    freqVecTrain = svd.fit_transform(freqVecTrain)
    freqVecTest = svd.fit_transform(freqVecTest)

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
        clf = MultinomialNB()
        clfs.append((clf, 'Naive Bayes'))

    if 'KNN' in clfNames:
        clf = MultinomialNB()
        clfs.append((clf, 'KNN'))

    if 'My Method' in clfNames:
        clf = MultinomialNB()
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

        print predCategs

    # Print output to csv file
    if len(clfs) == 5:
        df = pd.DataFrame(info, columns=['Statistic Measure', 'Naive Bayes', 'Random Forest', 'SVM', 'KNN', 'My Method'])
        df.to_csv('EvaluationMetric_10fold.csv', sep='\t', index=False)

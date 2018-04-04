import numpy as np
from sklearn import svm, preprocessing
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.metrics import classification_report, precision_score
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


def processText(text):
    stopWords = stopwords.words('english')
    tokens = word_tokenize(text)

    tokens = [wrd for wrd in tokens if wrd not in stopWords]

    stemmer = PorterStemmer()
    tokens = [stemmer.stem(t) for t in tokens]

    return tokens


def classifiers(trainData, testData, clfNames):
    trainData = trainData[:500]
    testData = testData[:2]

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

    # Add classifiers that belong to 'clfNames' list
    if 'SVC' in clfNames:
        parameters = [
            {'C': [0.1, 10, 100], 'kernel': ['linear']},
            {'C': [0.1, 10, 100], 'gamma': [0.001, 0.1, 10], 'kernel': ['rbf']},
        ]

        clfSvc = svm.SVC()
        clf = GridSearchCV(clfSvc, parameters)
        clfs.append(clf)

    if 'RandomForest' in clfNames:
        clf = RandomForestClassifier()
        clfs.append(clf)

    if 'MultinomialNB':
        clf = MultinomialNB()
        clfs.append(clf)

    # Run classifiers
    for clf in clfs:
        # Cross Validation
        scores = cross_val_score(clf, freqVecTrain, categoryIds, cv=10)
        print 'Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2)

        kf = KFold(n_splits=10)
        fold = 0

        for train_index, test_index in kf.split(trainData):
            kfFreqVecTrain = vectorizer.transform(np.array(trainText)[train_index])
            kfFreqVecTest = vectorizer.transform(np.array(trainText)[test_index])

            clf.fit(kfFreqVecTrain, categoryIds[train_index])
            yPred = clf.predict(kfFreqVecTest)
            fold += 1

            print "Fold " + str(fold)
            print classification_report(le.transform(trainData['Category'][test_index]), yPred, target_names=list(le.classes_))

        # Train with real trainSet and predict for real testSet
        clf.fit(freqVecTrain, categoryIds)

        predIds = clf.predict(freqVecTest)
        predCategs = le.inverse_transform(predIds)

        print predCategs



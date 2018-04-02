from sklearn import svm, preprocessing
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV


def classifiers(train_data, test_data, clfNames):
    train_data = train_data[:2000]
    test_data = test_data[:2]

    le = preprocessing.LabelEncoder()
    categoryIds = le.fit_transform(train_data['Category'])

    vectorizer = TfidfVectorizer(ngram_range=(1, 2)).fit(train_data['Content'])
    freqVecTrain = vectorizer.transform(train_data['Content'])
    clfs = []

    if 'SVC' in clfNames: 
        parameters = [
            {'C': [0.1, 10, 100], 'kernel': ['linear']},
            {'C': [0.1, 10, 100], 'gamma': [0.001, 0.1, 10], 'kernel': ['rbf']},
        ]

        clfSvc = svm.SVC()
        clf = GridSearchCV(clfSvc, parameters)
        clfs.append(clf)

    if 'RandomForest' in clfNames:
        clf = RandomForestClassifier(n_estimators=200)
        clfs.append(clf)
    
    if 'MultinomialNB':
        clf = MultinomialNB()
        clfs.append(clf)
    
    for clf in clfs:
        clf.fit(freqVecTrain, categoryIds)

        freqVecTest = vectorizer.transform(test_data['Content'])
        predIds = clf.predict(freqVecTest)
        predCategs = le.inverse_transform(predIds)

        print predCategs

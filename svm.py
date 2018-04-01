from sklearn import svm, preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn import svm


def SVC(train_data, test_data):
    train_data = train_data[:400]
    test_data = test_data[:2]

    le = preprocessing.LabelEncoder()
    categoryIds = le.fit_transform(train_data['Category'])

    vectorizer = TfidfVectorizer(ngram_range=(1, 2)).fit(train_data['Content'])
    freqVec = vectorizer.transform(train_data['Content'])
    
    parametrs = [
        {'C': [0.1, 10, 100], 'kernel': ['linear']},
        {'C': [0.1, 10, 100], 'gamma': [0.001, 0.1, 10], 'kernel': ['rbf']},
    ]

    clfSvc = svm.SVC()
    clf = GridSearchCV(clfSvc, parametrs)
    clf.fit(freqVec, categoryIds)

    freqVec = vectorizer.transform(test_data['Content'])
    predIds = clf.predict(freqVec)
    predCategs = le.inverse_transform(predIds)

    print predCategs

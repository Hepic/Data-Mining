from sklearn import svm, preprocessing
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
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


def classifiers(train_data, test_data, clfNames):
    test_data = train_data[2001:3001] # TOCHANGED: THAT SHOULD BE RETRIEVED FROM TEST DATA
    train_data = train_data[:2000]

    le = preprocessing.LabelEncoder()
    categoryIds = le.fit_transform(train_data['Category'])

    vectorizer = TfidfVectorizer(tokenizer=processText, ngram_range=(1, 2), stop_words='english').fit(train_data['Content'])
    freqVecTrain = vectorizer.transform(train_data['Content'])
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
        clf.fit(freqVecTrain, categoryIds)

        freqVecTest = vectorizer.transform(test_data['Content'])
        predIds = clf.predict(freqVecTest)
        predCategs = le.inverse_transform(predIds)
 
        print classification_report(le.transform(test_data['Category']), predIds, target_names=list(le.classes_))

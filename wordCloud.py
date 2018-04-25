import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk import word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn import svm, preprocessing


def main():
    trainData = pd.read_csv('datasets/train_set.csv', sep='\t')
    
    # Labels for categories
    le = preprocessing.LabelEncoder()
    categoryIds = le.fit_transform(trainData['Category'])

    categLen = len(set(categoryIds))
    groups = ['' for i in range(categLen)]
    
    # Create stopWords list
    stopWords = stopwords.words('english')
    stopWords.extend(['saying', 'said', 'say', 'yes', 'instead', 'meanwhile', 'right', 'really', 'finally', 'now', 
                       'one', 'suggested', 'says', 'added', 'think', 'know', 'though', 'let', 'going', 'back',
                       'well', 'example', 'us', 'yet', 'perhaps', 'actually', 'oh', 'year', 'lastyear',
                       'last', 'old', 'first', 'good', 'maybe', 'ask', '.', ',', ':', 'take' 'made', 'n\'t', 'go', 
                       'make', 'two', 'got', 'took', 'want', 'much', 'may', 'never', 'second'])
    
    # pos=3 is content, pos=2 is title, pos=4 is category
    for elem in np.array(trainData):
        pos = le.transform([elem[4]])[0]

        contentText = elem[3].decode('utf-8').lower()
        tokens = word_tokenize(contentText)
        
        for wrd in tokens: # words from content
            if wrd not in stopWords:
                groups[pos] += wrd + ' '
        
        titleText = elem[2].decode('utf-8').lower()
        tokens = word_tokenize(titleText)

        for wrd in tokens: # words from title 
            if wrd not in stopWords:
                groups[pos] += wrd + ' '
    
    for i in range(categLen):
        wordcloud = WordCloud(max_font_size=40).generate(groups[i])
        fig = plt.figure()
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        fig.savefig('static/' + le.inverse_transform(i) + '.png')


if __name__ == '__main__':
    main()

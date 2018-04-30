import pandas as pd
from classifiers import classifiers


def main():
    trainData = pd.read_csv('datasets/my_train_set.csv', sep='\t')
    testData = pd.read_csv('datasets/test_set.csv', sep='\t')
    
    classifiers(trainData, testData, ['My Method'])


if __name__ == '__main__':
    main()

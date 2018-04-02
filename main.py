import numpy as np
import pandas as pd
from classifiers import classifiers


def main():
    train_data = pd.read_csv('datasets/train_set.csv', sep='\t')
    test_data = pd.read_csv('datasets/test_set.csv', sep='\t')

    classifiers(train_data, test_data, ['SVC', 'RandomForest', 'MultinomialNB'])


if __name__ == '__main__':
    main()

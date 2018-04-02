import numpy as np
import pandas as pd
from svm import SVC


def main():
    train_data = pd.read_csv('datasets/train_set.csv', sep='\t')
    test_data = pd.read_csv('datasets/test_set.csv', sep='\t')

    SVC(train_data, test_data)


if __name__ == '__main__':
    main()

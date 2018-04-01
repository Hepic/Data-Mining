import numpy as np
import pandas as pd

def main():
    train_data = pd.read_csv('train_set.csv', sep='\t')
    print train_data

if __name__ == '__main__':
    main()

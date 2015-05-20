import pandas as pd
import numpy as np
import sys

DATA = sys.argv[1]
COUNT_COL = 'score'
PARTITION_COL = 'partition'
THRESHOLD = 10

def get_partitions(df, count_column, partition_column, threshold = 1):
    '''Takes a pandas df, the column name that we are sorting by;
    adds a count_column + "_partition" column, which stores which
    partition each item is in'''
    df[partition_column] = np.nan
    partition = 1
    while df[partition_column].isnull().sum() > threshold:
        # Get median count of count_column where partition_column is still null
        median = df.loc[df.loc[:,partition_column].isnull(),
                count_column].median()

        # Update everything that is null and is less than the median to the current partition
        df.loc[(df[partition_column].isnull()) &
                (df[count_column] < median),
                partition_column] = partition

        #Increment
        partition += 1

    # give everyone else the last partition
    df[partition_column].fillna(partition - 1, inplace=True)

def main():
    data_frame = pd.read_csv(DATA, escapechar="\\")
    get_partitions(data_frame, COUNT_COL, PARTITION_COL, THRESHOLD)

if __name__ == "__main__":
        main()

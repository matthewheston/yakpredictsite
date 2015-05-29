#!/usr/bin/env python3
from __future__ import division
import pandas as pd
import numpy as np
import sys

DATA = sys.argv[1]
OUTPUT = sys.argv[2]
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

        # Update everything that is null and is less than the median 
        # to the current partition
        df.loc[(df[partition_column].isnull()) &
                (df[count_column] < median),
                partition_column] = partition

        #Increment
        partition += 1

    # Give everyone else the last partition
    df[partition_column].fillna(partition - 1, inplace=True)

def get_time_info(df, timestamp_col):
    '''Takes a data frame and the column name of the timestamp.
    Adds a "time_of_day" and "day_of_week" column'''

    time_cutoffs = [0,4,8,12,16,20,24]
    time_of_day = ['Late night', 'Early morning', 'Morning',
                    'Afternoon', 'Evening', 'Night']
    df['day_of_week'] = df[timestamp_col].dt.dayofweek
    df['time_of_day'] = pd.cut(df[timestamp_col].dt.hour, time_cutoffs,
                                labels=time_of_day)

def get_message_info(df, message_col):
    '''Takes a data frame and the column name of the message
    text. Adds "message_len", "num_words", and "mean_word_len" columns'''
    df['message_len'] = df[message_col].apply(len)
    df['num_words'] = df[message_col].apply(lambda x: len(x.split(' ')))
    df['mean_word_len'] = df[message_col].apply(get_mean_word_len)

def get_mean_word_len(string):
    words = string.split(' ')
    mean_len = sum([len(w) for w in words])/len(words)
    return mean_len


def main():
    # Read in data
    data_frame = pd.read_csv(DATA, escapechar="\\")
    data_frame['time'] = pd.to_datetime(data_frame['time'])
    # Get partitions
    get_partitions(data_frame, COUNT_COL, PARTITION_COL, THRESHOLD)
    get_time_info(data_frame, 'time')
    get_message_info(data_frame, 'message')
    # Write to file
    data_frame.to_csv(OUTPUT, escapechar="\\")

if __name__ == "__main__":
        main()

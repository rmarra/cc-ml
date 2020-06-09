#!/usr/bin/env python
# coding: utf-8
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split


def is_botnet(label):
    regex = re.compile('^.*Botnet.*$', re.I)
    if regex.match(label):
        return True
    return False


def calc_packets_avg(total_bytes, total_packets):
    return total_bytes/total_packets


def calc_bps_avg(total_bytes, duration):
    try:
        total_bits = total_bytes * 8
        return total_bits / duration
    except ZeroDivisionError:
        return total_bits 


def load_database(filepath):
    return pd.read_csv(filepath)


def process_dataset(df):
    columns = ['Dur', 'Proto', 'State', 'TotBytes', 'TotPkts' , 'Label']
    df = df[columns]
    df['PktsAvg'] = df.apply(
        lambda row: calc_packets_avg(row['TotBytes'], row['TotPkts']),
        axis=1
    )
    df['BpsAvg'] = df.apply(
        lambda row: calc_bps_avg(row['TotBytes'], row['Dur']),
        axis=1
    )
    state_dummy = pd.get_dummies(df.State, prefix='STATE')
    protocol_dummy = pd.get_dummies(df.Proto, prefix='PROTOCOL')
    label = df.Label.apply(lambda value: 1 if is_botnet(value) else -1)
    df = df.drop(['Label', 'TotPkts', 'State', 'Proto'], axis=1)
    df = pd.concat([df, state_dummy, protocol_dummy], axis=1)
    return df, label

def run_ml(x, y):
    clf = svm.LinearSVC(max_iter=10000)
    scoring = ['recall','f1','accuracy', 'precision', 'roc_auc']
    return cross_validate(clf, df, label, cv=10, scoring=scoring, n_jobs=4)

def write_to_output_file(line):
    OUTPUT_FILE = "output.csv"
    with open(OUTPUT_FILE, 'a+') as fh:
        fh.write(line)

if __name__ == '__main__':
    HEADER = "file,fit_time,score_time,recall,f1,accuracy,precision,roc_auc\n"
    write_to_output_file(HEADER)
    dataset_folder = "dataset/CTU-13-Dataset/"
    dataset_paths = [list(x.glob("*.binetflow"))[0] for x in Path(dataset_folder).iterdir()]

    for dataset_path in dataset_paths[:1]:
        df = load_database(str(dataset_path))
        df, label = process_dataset(df)
        scores = run_ml(df, label)

        line = "{file},{fit_time},{score_time},{recall},{f1},{accuracy},{precision},{roc_auc}\n"
        write_to_output_file(
            line.format(
                file=str(dataset_path),
                fit_time=scores['fit_time'].mean(),
                score_time=scores['score_time'].mean(),
                recall=scores['test_recall'].mean(),
                f1=scores['test_f1'].mean(),
                accuracy=scores['test_accuracy'].mean(),
                precision=scores['test_precision'].mean(),
                roc_auc=scores['test_roc_auc'].mean()
            )
        )

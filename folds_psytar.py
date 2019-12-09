import csv
import os

import pandas as pd
from sklearn.model_selection import KFold

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--psytar', '-psytar', type=str, default='data/psytar_folds')
parser.add_argument('--psytar_text', '-psytar_text', type=str, default='data/Copy of PsyTAR_dataset.csv')
parser.add_argument('--psytar_adr', '-psytar_adr', type=str, default='data/Copy of PsyTAR_dataset_adr.csv')
parser.add_argument('--psytar_disease', '-psytar_disease', type=str, default='data/Copy of PsyTAR_dataset_disease.csv')
parser.add_argument('--psytar_symptoms', '-psytar_symptoms', type=str, default='data/Copy of PsyTAR_dataset_symptoms.csv')
parser.add_argument('--folds', '-folds', type=int, default=5)
args = parser.parse_args()


folds_path = args.psytar
text_path = args.psytar_text
adr_path = args.psytar_adr
disease_path = args.psytar_disease
symptom_path = args.psytar_symptoms
n_folds = args.folds


def get_text(filename):
    text = {}
    f = open(filename)
    csv_reader = csv.reader(f, delimiter=',')
    line = 0
    for row in csv_reader:
        if line == 0:
            line += 1
            continue
        else:
            text[row[3]] = row[6].replace("\n", " ") + " " + row[7].replace("\n", " ")
    f.close()
    return text


def get_adr_entity(filename, full_text):
    f = open(filename)
    text = {}
    csv_reader = csv.reader(f, delimiter=',')
    line = 0
    s = 0
    for row in csv_reader:
        if line == 0:
            line += 1
            continue
        else:
            for col in row[4:]:
                if col == '':
                    break
                start = full_text[row[1]].find(col)
                end = start + len(col)
                if row[1] in text:
                    text[row[1]].append({'start': start, 'end': end, 'entity': 'adr', 'text': col})
                else:
                    text[row[1]] = [{'start': start, 'end': end, 'entity': 'adr', 'text': col}]
    return text


def get_disease_entity(filename, full_text):
    f = open(filename)
    text = {}
    csv_reader = csv.reader(f, delimiter=',')
    line = 0
    s = 0
    for row in csv_reader:
        if line == 0:
            line += 1
            continue
        else:
            for col in row[4:]:
                if col == '':
                    break
                start = full_text[row[1]].find(col)
                end = start + len(col)
                if row[1] in text:
                    text[row[1]].append({'start': start, 'end': end, 'entity': 'disease', 'text': col})
                else:
                    text[row[1]] = [{'start': start, 'end': end, 'entity': 'disease', 'text': col}]
    return text


def get_symptom_entity(filename, full_text):
    f = open(filename)
    text = {}
    csv_reader = csv.reader(f, delimiter=',')
    line = 0
    s = 0
    for row in csv_reader:
        if line == 0:
            line += 1
            continue
        else:
            for col in row[4:]:
                if col == '':
                    break
                start = full_text[row[1]].find(col)
                end = start + len(col)
                if row[1] in text:
                    text[row[1]].append({'start': start, 'end': end, 'entity': 'symptom', 'text': col})
                else:
                    text[row[1]] = [{'start': start, 'end': end, 'entity': 'symptom', 'text': col}]
    return text


def make_entity_dict(text, adr, disease, symtom):
    result = {}
    for key in text:
        d = {}
        i = 0
        if adr.get(key) is not None:
            for j in range(len(adr[key])):
                d[i] = adr[key][j]
                i += 1
        if disease.get(key) is not None:
            for j in range(len(disease[key])):
                d[i] = disease[key][j]
                i += 1
        if symtom.get(key) is not None:
            for j in range(len(symtom[key])):
                d[i] = symtom[key][j]
                i += 1
        result[key] = d
    return result


df = pd.DataFrame(columns=['text', 'entities'])
text = get_text(text_path)
adr_entity = get_adr_entity(adr_path, text)
disease_entity = get_disease_entity(disease_path, text)
symptom_entity = get_symptom_entity(symptom_path, text)
entity_dict = make_entity_dict(text,adr_entity,disease_entity,symptom_entity)
print(entity_dict)

for key in text:
    df = df.append({'text': text[key], 'entities': entity_dict[key]}, ignore_index=True)


rkf = KFold(n_splits=n_folds)

n_fold = 0
for i_train, i_test in rkf.split(df):
    fold_path = os.path.join(folds_path,str(n_fold // 10) + str(n_fold))
    train = df.iloc[i_train]
    test = df.iloc[i_test]
    train.to_json(os.path.join(fold_path,"train.json"), orient='table')
    test.to_json(os.path.join(fold_path, "test.json"), orient='table')
    n_fold += 1

import os
import re
from sklearn.model_selection import KFold
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cadec', '-cadec', type=str, default="data/cadec")
parser.add_argument('--cadec_text', '-cadec_text', type=str, default="data/cadec/text")
parser.add_argument('--cadec_sct', '-cadec_sct', type=str, default="data/cadec/sct")
parser.add_argument('--cadec_original', '-cadec_original;', type=str, default="data/cadec/original")
parser.add_argument('--cadec_folds', '-cadec_folds', type=str, default="data/cadec_folds")
parser.add_argument('--folds', '-folds', type=int, default=5)

args = parser.parse_args()


cadec_dir = args.cadec
text_path = args.cadec_text
sct_path = args.cadec_sct
original_path = args.cadec_original
folds_path = args.cadec_folds
n_folds = args.folds




def get_entity_and_slice(directory):
    result = []
    for filename in sorted(os.listdir(directory)):
        f = open(os.path.join(directory, filename))
        text = f.read().lower().split("\n")
        rev = []
        for elem in text[:-1]:
            sentence = elem.replace("\t", " ")
            words = sentence.split(" ")
            if words[0][0] == "#":
                continue
            rev.append(words[1:4])
        result.append(rev)
    return result


def get_sct_info(directory):
    result = []
    reg_name = re.compile(r'\|[^+\n]*\|')
    reg_id = re.compile(r'[0-9]+.?\|')
    for filename in sorted(os.listdir(directory)):
        f = open(os.path.join(directory, filename))
        rev = []
        for line in f:
            name = reg_name.findall(line)
            name = [i[1:-1].strip() for i in name]
            text = line.split("\t")[-1][:-1]
            id = reg_id.findall(line)
            id = [int(i[:-1].strip()) for i in id]
            if len(name) == 0 or len(id) == 0:
                name.append(['unknown'])
                id.append([None])
            rev.append([id, name, text])
        result.append(rev)
    return result





def get_text(directory):
    result = []
    for filename in sorted(os.listdir(directory)):
        f = open(os.path.join(directory, filename))
        text = f.read()
        text = text.replace("\n", " ")
        result.append(text)
        f.close()
    return result


def make_entity_dictionary(text, ent, sct, len_dataset):
    result = {}
    for i in range(len_dataset):
        d = {}
        if len(sct[i]) == 0:
            continue
        for j in range(len(ent[i])):
            if ';' in ent[i][j][1]:
                ent[i][j][1] = int(ent[i][j][1].split(";")[-1])
            if ';' in ent[i][j][2]:
                ent[i][j][2] = int(ent[i][j][2].split(";")[-1])
            d[j] = {'start': ent[i][j][1], 'end': ent[i][j][2],
                    'entity': ent[i][j][0], 'sct_name':sct[i][j][1],
                    'sct_id': sct[i][j][0], 'text':sct[i][j][2]}
        result[i] = d
    return result


df = pd.DataFrame(columns=['text', 'entities'])

text = get_text(text_path)
ent_slice = get_entity_and_slice(original_path)
sct = get_sct_info(sct_path)

len_dataset = len(list(os.listdir(text_path)))
entity_dict = make_entity_dictionary(text, ent_slice, sct, len_dataset)
number = 0
for i in list(entity_dict):
    df = df.append({'text': text[i], 'entities':entity_dict[i]}, ignore_index=True)

rkf = KFold(n_splits=n_folds)

n_fold = 0
for i_train,i_test in rkf.split(df):
    fold_path = os.path.join(folds_path,str(n_fold // 10) + str(n_fold))
    train = df.iloc[i_train]
    test = df.iloc[i_test]
    train.to_json(os.path.join(fold_path,"train.json"), orient='table')
    test.to_json(os.path.join(fold_path, "test.json"), orient='table')
    n_fold += 1


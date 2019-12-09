import os
import re
import csv
from collections import OrderedDict
from operator import itemgetter
import json


import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--cadec', '-cadec', type=str, default='data/cadec_folds')
parser.add_argument('--psytar', '-psytar', type=str, default='data/psytar_folds')
parser.add_argument('--stat', '-stat', type=str, default="data/stat.txt")

args = parser.parse_args()

cadec_folds = args.cadec
psytar_folds = args.psytar
stat_path = args.stat


def make_freq_dict(res):
    freq_dict = {}
    for word in res:
        if word in freq_dict:
            freq_dict[word] += 1
        else:
            freq_dict[word] = 1
    return freq_dict


def words_intersect(dict1, dict2):
    intersect_dict = {}
    s = 0
    for key in dict2:
        if key in dict1:
            intersect_dict[key] = min(dict1[key], dict2[key])
            s += min(dict1[key], dict2[key])
    return intersect_dict, s


def get_text_intersect(cadec_path, psytar_path):
    f = open(cadec_path)
    f1 = open(psytar_path)
    js_cadec = json.load(f)
    js_psytar = json.load(f1)
    text = ""
    del js_cadec['schema']
    del js_psytar['schema']
    for i in js_cadec['data']:
        text += i['text']
    p = re.compile("([a-zA-Z-']+)")
    res = p.findall(text)
    freq_dict = make_freq_dict(res)
    text = ""
    for i in js_psytar['data']:
        text += i['text']
    res = p.findall(text)
    freq_dict1 = make_freq_dict(res)
    intersect_dict, s = words_intersect(freq_dict, freq_dict1)
    return intersect_dict, s


def get_entity_stat(entity, cadec_path, psytar_path):
    f = open(cadec_path)
    f1 = open(psytar_path)
    js_cadec = json.load(f)
    js_psytar = json.load(f1)
    text = ""
    n_entities = 0
    del js_cadec['schema']
    del js_psytar['schema']
    for i in js_cadec['data']:
        for j in i['entities']:
            if i['entities'][j]['entity'] == entity:
                text += i['entities'][j]['text']
                n_entities += 1
    p = re.compile("([a-zA-Z-']+)")
    res = p.findall(text)
    freq_dict = make_freq_dict(res)
    text = ""
    n_entities1 = 0
    for i in js_psytar['data']:
        for j in i['entities']:
            if i['entities'][j]['entity'] == entity:
                text += i['entities'][j]['text']
                n_entities1 += 1
    freq_dict1 = make_freq_dict(res)
    intersect_dict, s = words_intersect(freq_dict, freq_dict1)
    return intersect_dict, s, n_entities, n_entities1


for directory in sorted(os.listdir(cadec_folds)):
    fold_path_cadec = os.path.join(cadec_folds, directory)
    train_cadec = os.path.join(fold_path_cadec, "train.json")
    test_cadec = os.path.join(fold_path_cadec, "test.json")

    fold_path_psytar = os.path.join(cadec_folds, directory)
    train_psytar = os.path.join(fold_path_cadec, "train.json")
    test_psytar = os.path.join(fold_path_cadec, "test.json")

    out_file = open(stat_path, 'a+')
    out_file.write("fold:{}\n".format(directory))

    text_intersection_train, s_text_train = get_text_intersect(train_cadec, train_psytar)
    text_intersection_test, s_text_test = get_text_intersect(test_cadec, test_psytar)

    out_file.write("text intersection train:{}\ntext intersection test: {}\n".format(s_text_train,s_text_test))

    adr_intersection_train, s_adr_train, adr_entities_cadec_train, adr_entities_psytar_train = get_entity_stat('adr',
                                                                                                               train_cadec,
                                                                                                               train_psytar)
    adr_intersection_test, s_adr_test, adr_entities_cadec_test, adr_entities_psytar_test = get_entity_stat('adr',
                                                                                                         test_cadec,
                                                                                                         test_psytar)
    out_file.write("adr intersection train:{}\nadr intersection test: {}\n".format(s_adr_train, s_adr_test))
    out_file.write("number of adr entities cadec train:{}\nnumber of adr entities cadec test: {}\n".format(adr_entities_cadec_train, adr_entities_cadec_test))
    out_file.write(
        "number of adr entities psytar train:{}\nnumber of adr entities psytar test: {}\n".format(adr_entities_psytar_train,
                                                                                              adr_entities_psytar_test))

    disease_intersection_train, s_disease_train, disease_entities_cadec_train, disease_entities_psytar_train = get_entity_stat(
        'disease', train_cadec,
        train_psytar)
    disease_intersection_test, s_disease_test, disease_entities_cadec_test, disease_entities_psytar_test = get_entity_stat(
        'disease', test_cadec,
        test_psytar)

    out_file.write("disease intersection train:{}\n intersection test: {}\n".format(s_disease_train, s_disease_test))
    out_file.write(
        "number of disease entities cadec train:{}\nnumber of disease entities cadec test: {}\n".format(
            disease_entities_cadec_train,
            disease_entities_cadec_test))
    out_file.write(
        "number of disease entities psytar train:{}\nnumber of disease entities psytar test: {}\n".format(
            disease_entities_psytar_train,
            disease_entities_psytar_test))

    symptom_intersection_train, s_symptom_train, symptom_entities_cadec_train, symptom_entities_psytar_train = get_entity_stat(
        'symptom', train_cadec,
        train_psytar)
    symptom_intersection_test, s_symptom_test, symptom_entities_cadec_test, symptom_entities_psytar_test = get_entity_stat(
        'symptom', test_cadec,
        test_psytar)

    out_file.write("symptom intersection train:{}\n intersection test: {}\n".format(s_symptom_train, s_symptom_test))
    out_file.write(
        "number of symptom entities cadec train:{}\nnumber of symptom entities cadec test: {}\n".format(
            symptom_entities_cadec_train,
            symptom_entities_cadec_test))
    out_file.write(
        "number of symptom entities psytar train:{}\nnumber of symptom entities psytar test: {}\n".format(
            symptom_entities_psytar_train,
            symptom_entities_psytar_test))
    out_file.close()



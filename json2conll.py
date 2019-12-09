import codecs
import json
import os

from nltk.tokenize import wordpunct_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import wordnet
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--cadec', '-cadec', type=str, default='data/cadec_folds')
parser.add_argument('--psytar', '-psytar', type=str, default='data/psytar_folds')
parser.add_argument('--entity', '-entity', type=str, default='adr')
parser.add_argument('--tagger', '-tagger', type=str, default='taggers/maxent_treebank_pos_tagger/english.pickle')

args = parser.parse_args()

cadec_folds = args.cadec
psytar_folds = args.psytar
entity_type = args.entity


tagger = nltk.data.load(args.tagger)
lemmatizer = WordNetLemmatizer()


def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def get_token_position_in_text(token, w_start, text):
    delimitter_start = None
    while text[w_start:w_start+len(token)] != token or (delimitter_start == None and w_start != 0):
        w_start += 1
        delimitter_start = delimitter_start or w_start
    return w_start, w_start + len(token), text[delimitter_start:w_start]


def get_bio_tag(w_start, w_end, entities, entity_type):
    for key, entity in entities.items():
        try:
            start = int(entity['start'])
            end = int(entity['end'])
        except Exception:
            raise Exception("Entity start and end must be an integer")

        if entity['entity'] == entity_type:
            if w_start > start and w_end <= end:
                adding = entity_type
                return 'I-' + adding
            elif w_start == start and w_end <= end:
                adding = entity_type
                return 'B-' + adding
    return '0'



def json_to_conll(corpus_json_location, output_location, entity_type, by_sent = False):
    with codecs.open(corpus_json_location, encoding='utf-8') as in_file:
        reviews = list(map(json.loads, in_file.readlines()))
        reviews = reviews[0]['data']
    with codecs.open(output_location, 'w', encoding='utf-8') as out_file:
        for review in reviews:
            documents = sent_tokenize(review['text']) if by_sent else [review['text']]
            w_start = 0
            w_end = 0
            for document in documents:
                tokens = wordpunct_tokenize(document)
                pos_tags = tagger.tag(tokens)
                for token, temp in zip(tokens, pos_tags):
                    token_corr = temp[0].lower()
                    pos_tag = temp[1]
                    w_start, w_end, delimitter = get_token_position_in_text(token, w_start, review['text'])
                    bio_tag = get_bio_tag(w_start, w_end, review['entities'], entity_type)
                    lemm = lemmatizer.lemmatize(token_corr, get_wordnet_pos(pos_tag))
                    out_file.write(u'{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(token, lemm, pos_tag, bio_tag, w_start, w_end, delimitter, review['index']))
                    w_start = w_end - 1
                out_file.write('\n')


for directory in os.listdir(cadec_folds):
    fold_path_cadec = os.path.join(cadec_folds, directory)
    train_cadec = os.path.join(fold_path_cadec, "train.json")
    json_to_conll(train_cadec,os.path.join(fold_path_cadec, "train.conll"), entity_type)
    test_cadec = os.path.join(fold_path_cadec, "test.json")
    json_to_conll(test_cadec, os.path.join(fold_path_cadec, "test.conll"), entity_type)

    fold_path_psytar = os.path.join(psytar_folds, directory)
    train_psytar = os.path.join(fold_path_psytar, "train.json")
    json_to_conll(train_psytar, os.path.join(fold_path_psytar, "train.conll"), entity_type)
    test_psytar = os.path.join(fold_path_psytar, "test.json")
    json_to_conll(test_psytar, os.path.join(fold_path_psytar, "test.conll"), entity_type)


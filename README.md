
python folds_cadec.py -cadec data/cadec -cadec_text data/cadec/text -cadec_sct data/cadec/sct -cadec_original data/cadec/original -cadec_folds data/cadec_folds -folds 5

python folds_psytar.py -psytar data/psytar_folds  -psytar_text data/Copy_of_PsyTAR_dataset.csv  -psytar_adr data/Copy_of_PsyTAR_dataset_adr.csv  -psytar_disease data/Copy_of_PsyTAR_dataset_disease.csv  -psytar_symptoms data/Copy_of_PsyTAR_dataset_symptoms.csv -folds 5

python json2conll.py -cadec data/cadec_folds -psytar data/psytar_folds -entity adr -tagger taggers/maxent_treebank_pos_tagger/english.pickle

python statistic.py -cadec data/cadec_folds -psytar data/psytar_folds -stat data/stat.txt

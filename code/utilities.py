from typing import List
import spacy
from spacy.lang.en import STOP_WORDS as eng_stop_words
from nltk.stem.porter import PorterStemmer

import numpy as np
import pandas as pd
from itertools import product
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


class TextCleaner:

    def __init__(self,
                 spacy_vocab : str = 'en_core_web_sm',
                 stopwords_adjustments: dict = {},
                 remove_stopwords: bool = True,
                 remove_numbers: bool = True,
                 remove_recognized_entities: bool = True,
                 remove_punct: bool = True,
                 word_red:str = 'stem'):
        
        
        
        self.nlp = spacy.load(spacy_vocab)
        self.stopwords = eng_stop_words.union(set(
            stopwords_adjustments['+'] if '+' in stopwords_adjustments.keys() else [])) - set(
                stopwords_adjustments['-'] if '-' in stopwords_adjustments.keys() else [])
        self.removing = {'stopwords': remove_stopwords, 
                         'numbers': remove_numbers, 
                         'entities': remove_recognized_entities, 
                         'punct': remove_punct}
        self.word_red = word_red

        if self.word_red=='stem':

            self.stemmer = PorterStemmer()

        self.check_input(spacy_vocab, stopwords_adjustments, self.removing, word_red)

        

    def check_input(self, 
                    vocab: str, 
                    stopwords_adjustments: dict, 
                    removing_dict: dict, 
                    word_red: str) -> None:

        admitted_vocab_str = ['en_core_web_sm', 'en_core_web_md', 'en_core_web_lg']
        
        assert vocab in admitted_vocab_str, f'Vocabulary string {vocab} not admitted. Managed are {admitted_vocab_str}'
        
        admitted_keys = ['+','-']
        
        assert set(stopwords_adjustments.keys()).issubset(set(admitted_keys)), f'Stopwords append/removal failed due to non admitted keys. Managed keys are {admitted_keys}'

        removing_dict_check_dtypes = [isinstance(cond, bool) for cond in removing_dict.values()]

        assert len(set(removing_dict_check_dtypes))==1 and removing_dict_check_dtypes[0], f'Only boolean admitted'

        admitted_word_reductions = ['stem', 'lem', None]

        assert word_red in admitted_word_reductions, f'{word_red} not admitted. Managed are {admitted_word_reductions}'


    def clean_single_str(self, hl: str) -> str:

        doc = self.nlp(hl)

        to_remove = list(self.stopwords) if self.removing['stopwords'] else []

        to_remove += [token.text for token in doc if token.is_digit] if self.removing['numbers'] else []
        to_remove += [token.text for token in doc if token.is_punct] if self.removing['punct'] else []

        entities = [e.text for e in doc.ents] if self.removing['entities'] else []

        if self.word_red=='stem':

            clean_l = [self.stemmer.stem(token.text.lower()) for token in doc if token.text.lower() not in to_remove and token.text not in entities]


        elif self.word_red=='lem':


            clean_l = [token.lemma_.lower() for token in doc if token.text.lower() not in to_remove and token.text not in entities]

        else:

            clean_l = [token.text.lower() for token in doc if token.text.lower() not in to_remove and token.text not in entities]


        return ' '.join(clean_l)
    
    
    def clean_ls(self, ls: List[str]) -> str:

        cleaned_strings = [self.clean_single_str(single_str) for single_str in ls]

        return ' '.join(cleaned_strings)
    




class EvalClfPipeParams:

    def __init__(self,
                 vectorizers: list,
                 vects_grid: dict,
                 models_grid: dict,
                 text: list,
                 y: list,
                 test_size: int,
                 cv: int):
        

        self.vects = vectorizers
        self.vects_combos = {vc: [dict(zip(vects_grid.keys(), combo)) for combo in product(*vects_grid.values())] for vc in vectorizers}

        
        self.base_models = models_grid.keys()
        self.models_combos = {mdl: [dict(zip(models_grid[mdl]['tuning'].keys(), combo), **models_grid[mdl]['fixed']) for combo in product(*models_grid[mdl]['tuning'].values())] for mdl in models_grid.keys()}
        self.text = text
        self.y = y
        self.test_size = test_size
        self.cv = cv
        
        self.results = None


    @staticmethod
    def split_dataset_points(test_size: int, cv: int, size: int) -> list:

        output = [size - (j+1)*test_size for j in range(cv)]

        return output

    @staticmethod
    def setup_feature_extraction(vectorizer_str: str, parameters: dict):

        fe = CountVectorizer(**parameters) if vectorizer_str=='cv' else TfidfVectorizer(**parameters)

        return fe
    
    @staticmethod
    def setup_model(model_str: str, parameters: dict):

        match model_str:

            case 'DecisionTreeClassifier':

                return DecisionTreeClassifier(**parameters)
            
            case 'RandomForestClassifier':

                return RandomForestClassifier(**parameters)
            
            case 'GradientBoostingClassifier':

                return GradientBoostingClassifier(**parameters)

    
    def eval(self):

        split_points = self.split_dataset_points(test_size = self.test_size, cv = self.cv, size = len(self.text))

        results = list()

        for vct_str in self.vects:

            for vct_par_dict in self.vects_combos[vct_str]:

                for base_model in self.base_models:

                    for mdl_params in self.models_combos[base_model]:

                        acc_cv_results, mcc_cv_results = list(), list()

                        for sp in split_points:

                            train_text, test_text = self.text[:sp], self.text[sp:sp+self.test_size]

                            fe = self.setup_feature_extraction(vct_str, vct_par_dict)

                            fe.fit(train_text)

                            X_train, X_test = fe.transform(train_text).toarray(), fe.transform(test_text).toarray()

                            y_train, y_test = self.y[:sp], self.y[sp:sp+self.test_size]

                            model = self.setup_model(base_model, mdl_params)

                            model.fit(X_train, y_train)

                            predictions = model.predict(X_test)

                            mcc, accuracy = matthews_corrcoef(y_test, predictions), accuracy_score(y_test, predictions)

                            acc_cv_results.append(accuracy)
                            mcc_cv_results.append(mcc)

                        to_store = (vct_str, vct_par_dict, model.__class__.__name__, mdl_params, np.mean(mcc_cv_results), np.mean(acc_cv_results))

                        results.append(to_store)

        df_results = pd.DataFrame(results, columns=['Vectorizer', 'Vect_parameters', 'Model', 'Model_parameters', 'MCC', 'accuracy']).sort_values(['MCC','accuracy'], ascending=[False,False])

        self.results = df_results









                        













        












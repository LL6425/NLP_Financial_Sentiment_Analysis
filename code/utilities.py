from typing import List
import spacy
from spacy.lang.en import STOP_WORDS as eng_stop_words
from nltk.stem.porter import PorterStemmer

import numpy as np
import pandas as pd
from itertools import product
from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from skopt import BayesSearchCV



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
    






class CustomEval:

    def __init__(self,
                 clf: str = None,
                 metrics = ['MCC','accuracy'],
                 summary_metric = 'wavg',
                 sm_weights: dict = None
                 ):
        
        self.clf = clf
        self.metrics = metrics

        self.summary_metric = summary_metric
        self.sm_weights = {mt: 1/len(metrics) for mt in metrics} if sm_weights is None else sm_weights

        self.standard_metrics = {'MCC','accuracy','F1','roc_auc'}
        self.custom_metrics = {'pnl_sum','MDD'}

    @staticmethod
    def max_drawdown(y_true: np.array, y_pred: np.array, rets: np.array, clf: str) -> float:

        if clf=='binary':

            mult_factors = np.where(y_true==y_pred, 1+abs(rets), 1-abs(rets))

        elif clf=='ternary':

            mult_factors = np.where(np.logical_or(np.logical_and(y_pred==2,rets>=0),np.logical_and(y_pred==0,rets<0)), 1+abs(rets),
                                     np.where(y_pred==1, 1, 1-abs(rets)))

        else:

            raise ValueError(f'Classification {clf} not admitted. Available are binary, ternary')

        compound_returns = np.cumprod(mult_factors)

        drawdowns = (np.maximum.accumulate(compound_returns) - compound_returns) / np.maximum.accumulate(compound_returns)

        max_dd = -np.max(drawdowns) if -np.max(drawdowns) < 0 else np.nan

        return max_dd
    

    @staticmethod
    def pnl_sum(y_true: np.array, y_pred: np.array, rets: np.array, clf: str) -> float:

        if clf=='binary':

            realized_rets = np.where(y_true==y_pred, abs(rets), -abs(rets))

        elif clf=='ternary':

            realized_rets = np.where(np.logical_or(np.logical_and(y_pred==2,rets>=0),np.logical_and(y_pred==0,rets<0)), abs(rets),
                                     np.where(y_pred==1, 0, -abs(rets)))

        else:

            raise ValueError(f'Classification {clf} not admitted. Available are binary, ternary')

        return np.sum(realized_rets)
    
    @staticmethod
    def compute_wavg_summary_metric(metrics_results: dict, weights: dict) -> float:

        assert set(weights.keys()).issubset(set(metrics_results.keys())), f'Check weights and metrics keys, {set(weights.keys()) - set(metrics_results.keys())} not found in metrics_results keys' 

        weights_sum = np.sum(list(weights.values()))

        relative_weights = {mt: wt/weights_sum for mt,wt in weights.items()}

        summary_metric = np.sum([metrics_results[mt]*relative_weights[mt] for mt in weights.keys()])

        return summary_metric


    def eval(self, y_true: np.array, y_pred: np.array, rets=None) -> dict:

        standard_metrics_to_comp = set(self.metrics).intersection(self.standard_metrics)

        custom_metrics_to_comp = set(self.metrics).intersection(self.custom_metrics)

        metric_func = {'MCC': matthews_corrcoef,
                       'accuracy': accuracy_score,
                       'F1': f1_score,
                       'roc_auc':roc_auc_score,
                       'pnl_sum': self.pnl_sum,
                       'MDD':self.max_drawdown}
        
        results = {}

        for st_metric in standard_metrics_to_comp:

            try:

                results[st_metric] = metric_func[st_metric](y_true, y_pred)

            except:

                results[st_metric] = np.nan

        for cust_metric in custom_metrics_to_comp:

            results[cust_metric] = metric_func[cust_metric](y_true, y_pred, rets, self.clf)

        if self.summary_metric=='wavg':

            results['summary_metric'] = self.compute_wavg_summary_metric(results, self.sm_weights)

        elif callable(self.summary_metric):

            results['summary_metric'] = self.summary_metric(results, self.sm_weights)

        else:

            pass

        return results





class NlpPipeBayesSearch:

    def __init__(self, 
                 vects_dict: dict,
                 clfs_dict: dict,
                 cv_object,
                 n_iter: int,
                 random_state,
                 n_jobs: int,
                 scoring = None,
                 std_penalty: bool = True):
        
        self.vects_dict = vects_dict
        self.clfs_dict = clfs_dict
        
        self.cv, self.n_iter = cv_object, n_iter

        self.random_state = random_state
        self.n_jobs = n_jobs
        self.scoring = scoring

        self.std_penalty = std_penalty

        self.pipelines_instructions, self.combos_results = None, None

    @staticmethod
    def build_pipelines(vects_dict, clfs_dict):

        pipelines_dict = {}

        combos = product([(k,v['object']) for k,v in vects_dict.items()], [(k,v['object']) for k,v in clfs_dict.items()])

        for (vect_str,vect_object), (model_str, model_object) in combos:

            pipe = Pipeline([('vec',vect_object), ('to_array', FunctionTransformer(lambda x: x.toarray(), accept_sparse=True)), ('clf', model_object)])

            param_space_vect = {f'vec__{par}':sp for par,sp in vects_dict[vect_str]['space'].items()}

            param_space_mdl = {f'clf__{par}':sp for par,sp in clfs_dict[model_str]['space'].items()}

            param_space = dict(**param_space_vect, **param_space_mdl)

            key = (vect_str, model_str)

            pipelines_dict[key] = {'pipe':pipe, 'space':param_space}

        return pipelines_dict


    @staticmethod
    def best_stdpen_params(cv_results):

        cv_results['penalized_mean_test_score'] = cv_results['mean_test_score'] - cv_results['std_test_score']

        idx_best = np.argmax(cv_results['penalized_mean_test_score'])

        best_params = dict(cv_results['params'][idx_best])

        return cv_results, best_params 
    
    
    def search(self, X: np.array, y: np.array):

        cv_res_dict, selected_params = {}, {}

        pipelines_dict = self.build_pipelines(self.vects_dict, self.clfs_dict)

        for (vect_str, model_str), instructions in tqdm(pipelines_dict.items()):

            pipe, space = instructions['pipe'], instructions['space']

            bscv = BayesSearchCV(estimator = pipe, 
                                 search_spaces = space, 
                                 n_iter = self.n_iter, 
                                 random_state=self.random_state, 
                                 cv=self.cv,
                                 n_jobs=self.n_jobs,
                                 scoring=self.scoring)
            
            np.int = int
            
            bscv.fit(X,y)

            cv_res = bscv.cv_results_.copy()

            cv_res, best_params = (self.best_stdpen_params(cv_res)) if self.std_penalty else (cv_res, bscv.best_params_)

            cv_res_dict[(vect_str, model_str)], selected_params[(vect_str, model_str)] = cv_res, dict(best_params)
            

        self.pipelines_instructions = selected_params
        self.combos_results = cv_res_dict



            








    
    

















                        













        












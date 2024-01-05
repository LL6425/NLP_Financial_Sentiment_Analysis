import numpy as np
from itertools import product
from tqdm.notebook import tqdm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from skopt import BayesSearchCV
from sklearn.base import TransformerMixin

class ToArrayTransformer(TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.toarray()


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
    def build_pipelines(vects_dict: dict, clfs_dict: dict) -> dict:

        pipelines_dict = {}

        combos = product([(k,v['object']) for k,v in vects_dict.items()], [(k,v['object']) for k,v in clfs_dict.items()])

        for (vect_str,vect_object), (model_str, model_object) in combos:

            pipe = Pipeline([('vec',vect_object), ('to_array', ToArrayTransformer()), ('clf', model_object)])

            param_space_vect = {f'vec__{par}':sp for par,sp in vects_dict[vect_str]['space'].items()}

            param_space_mdl = {f'clf__{par}':sp for par,sp in clfs_dict[model_str]['space'].items()}

            param_space = dict(**param_space_vect, **param_space_mdl)

            key = (vect_str, model_str)

            pipelines_dict[key] = {'pipe':pipe, 'space':param_space}

        return pipelines_dict


    @staticmethod
    def custom_refit(cv_results: dict) -> int:

        cv_results['penalized_mean_test_score'] = cv_results['mean_test_score'] - cv_results['std_test_score']

        idx_best = np.argmax(cv_results['penalized_mean_test_score'])

        return idx_best 
    
    
    def search(self, X: np.array, y: np.array):

        cv_res_dict, selected_pipes = {}, {}

        pipelines_dict = self.build_pipelines(self.vects_dict, self.clfs_dict)

        for (vect_str, model_str), instructions in tqdm(pipelines_dict.items()):

            pipe, space = instructions['pipe'], instructions['space']

            bscv = BayesSearchCV(estimator = pipe, 
                                 search_spaces = space, 
                                 n_iter = self.n_iter, 
                                 random_state=self.random_state, 
                                 cv=self.cv,
                                 n_jobs=self.n_jobs,
                                 scoring=self.scoring,
                                 refit=self.custom_refit if self.std_penalty else True)
            
            np.int = int
            
            bscv.fit(X,y)

            cv_res = bscv.cv_results_.copy()

            best_pipe = [(el_str,el) for el_str,el in bscv.best_estimator_.named_steps.items()]  

            cv_res_dict[(vect_str, model_str)], selected_pipes[(vect_str, model_str)] = cv_res, best_pipe
            

        self.pipelines_instructions = selected_pipes
        self.cv_results = cv_res_dict



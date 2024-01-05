import numpy as np
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, roc_auc_score


class CustomEval:

    def __init__(self,
                 clf: str = None,
                 metrics = ['MCC','accuracy'],
                 summary_metric = False,
                 sm_weights: dict = None
                 ):
        
        self.clf = clf
        self.metrics = metrics

        self.summary_metric = summary_metric
        self.sm_weights = {mt: 1/len(metrics) for mt in metrics} if sm_weights is None else sm_weights

        self.standard_metrics = {'MCC','Accuracy','F1','Roc_Auc'}
        self.custom_metrics = {'PnL_sum','MDD'}

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

        summary_metric = np.sum([metrics_results[mt]*relative_weights[mt] for mt in weights.keys()], where=np.invert(np.isnan([metrics_results[mt] for mt in weights.keys()])))

        return summary_metric


    def eval(self, y_true: np.array, y_pred: np.array, rets=None) -> dict:

        standard_metrics_to_comp = set(self.metrics).intersection(self.standard_metrics)

        custom_metrics_to_comp = set(self.metrics).intersection(self.custom_metrics)

        metric_func = {'MCC': matthews_corrcoef,
                       'Accuracy': accuracy_score,
                       'F1': f1_score,
                       'Roc_Auc':roc_auc_score,
                       'PnL_sum': self.pnl_sum,
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
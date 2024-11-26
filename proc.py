import numpy as np
import pandas as pd
import scipy.stats as stats 
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, QuantileRegressor
from quantile_forest import RandomForestQuantileRegressor
import sksurv
from sksurv.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, ExtraSurvivalTrees, GradientBoostingSurvivalAnalysis
import warnings
from sksurv.metrics import (
    as_concordance_index_ipcw_scorer,
    as_cumulative_dynamic_auc_scorer,
    as_integrated_brier_score_scorer,
)


def compute_emp_pvalue(test_statistic, null_statistics):
    return (1 + np.sum(null_statistics >= test_statistic)) / (len(null_statistics)+1)


class TwoSidedCP(object):
    def __init__(self, X, Y, 
                 alpha0,alpha1,
                 black_box_surv, 
                 black_box_event, 
                 black_box_lb=None, 
                 plug_in_lower_bounds=None, 
                 surv_parameters_grid=None, 
                 clf_parameters_grid=None,  
                 random_state=2020):
        """
        alpha0, alpha1: see paper 
        black_box_event: classifier for the classification of the event variable (non-conformity score is 1-probability of being in class y given x)
        black_box_surv: non-conformity score for the two-sided prediction (see scores.py) 
        clf_parameters_grid: grid of hyper-parameters for fitting the classifier (using 3-fold CV)
        surv_parameters_grid: grid of hyper-parameters for fitting the two-sided score <black_box_surv>: only applicable if <black_box_surv> is survival-based
        """

        self.alpha0=alpha0
        self.alpha1=alpha1
        self.ymax = int(np.max([y[1] for y in Y]))

        # Split data into training/calibration sets
        X_train, X_calib, Y_train, Y_calib = train_test_split(X, Y, test_size=0.5, random_state=random_state)

        # Fit classifier model 
        event_tr, _  = zip(*Y_train) 
        event_cal, _  = zip(*Y_calib) 
        event_tr=np.array(list(event_tr))*1
        event_cal=np.array(list(event_cal))*1
        if clf_parameters_grid is not None: 
            self.model_event = GridSearchCV(black_box_event, clf_parameters_grid, cv=3)
        else: self.model_event = black_box_event
        self.model_event.fit(X_train, event_tr)

        # Compute null scores for conformalized classification 
        event_cal, _ = zip(*Y_calib) 
        event_cal = np.array(list(event_cal))*1
        self.scores_null = self.model_event.predict_proba(X_calib)[:, 1][event_cal==0] 
        
        # Fit survival model for two-sided PIs
        lower, upper = np.percentile([y[1] for y in Y_train], [10, 90])
        times = np.arange(lower, upper + 1)
        self.model_surv = black_box_surv
        if surv_parameters_grid is not None:
            #set parameters of self.model_surv.model using cross-validation
            cv = GridSearchCV(as_concordance_index_ipcw_scorer(black_box_surv.model, tau=times[-1]), 
                                  param_grid=surv_parameters_grid,
                                    cv=3)
            cv.fit(X_train, Y_train)
            best_params = {key.split("__")[1]:value for key,value in cv.best_params_.items()}
            self.model_surv.model.set_params(**best_params) 
        self.model_surv.fit(X_train, Y_train)
        
        # Compute calibration term for two-sided PIs
        n_cal_noncens= np.sum(event_cal ==True)
        scores_calib_noncens = self.model_surv.predict_score(X_calib[event_cal ==True], Y_calib[event_cal ==True])
        q_level = np.minimum(np.ceil((n_cal_noncens+1)*(1-self.alpha1))/n_cal_noncens, 1)
        self.qhat_noncens = np.quantile(scores_calib_noncens, q_level, method='higher')


        self.plug_in_lower_bounds=plug_in_lower_bounds 
        # Fit model & Compute calibration term for one-sided PIs if no lower bounds are provided (naive approach)
        if plug_in_lower_bounds is None: 
            self.plug_in=False
            if black_box_lb is None:
                """
                use the two-sided non-conformity score for one-sided prediction (except one-sided version of it)
                """
                self.black_box_lb = self.model_surv
            else:
                self.black_box_lb=black_box_lb
                self.black_box_lb.fit(X_train, Y_train)

            n_cal=len(Y_calib)
            scores_calib_all = self.black_box_lb.predict_one_sided_score(X_calib, Y_calib)
            q_level = np.ceil((n_cal+1)*(1-alpha0))/n_cal
            self.qhat_all = np.quantile(scores_calib_all, q_level, method='higher')


    def _predict_event(self, X):
        m=len(X)
        
        scores_test = self.model_event.predict_proba(X)[:,1]
        pvalues = np.array([compute_emp_pvalue(score_test, self.scores_null) for score_test in scores_test]) 

        event_pred = np.zeros(m)
        event_pred[pvalues < self.alpha1] =1 
        return event_pred
    
    def predict(self, X):
        m=len(X)

        Shat=np.array([None]*m)

        event_preds=self._predict_event(X)

        positives = np.arange(m)[event_preds==1]
        negatives = np.arange(m)[event_preds==0]
        
        Shat[positives] = self.model_surv.compute_prediction_set(X[positives], self.qhat_noncens)
        
        if self.plug_in_lower_bounds is None:
            Shat[negatives] = self.black_box_lb.compute_prediction_lb(X[negatives], self.qhat_all) 
        else: 
            Shat[negatives] = list(zip(self.plug_in_lower_bounds[negatives], [None]*m)) 
    
        return Shat, event_preds



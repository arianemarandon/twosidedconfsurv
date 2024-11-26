import numpy as np

class surv_cdf_score(object):
    """
    CDF-based score 
    Learns a model of T given X 
    Fits a survival model to survival data and uses as score function nu(x,y)= 0.5-F_x(y), with F_x()the estimated c.d.f. of T at X=x 
    """
    def __init__(self, model):
        """
        model is a scikit-survival object 
        """
        self.model = model 

    def fit(self, X,y):
        """
        y is for the form (event, outcome) 
        """
        self.model.fit(X,y)
        
    def predict_score(self, X,y):
        """
        y is for the form (event, outcome) 
        """
        
        surv_fn=self.model.predict_survival_function(X) #returns m step functions P(T>t)
        times=self.model.unique_times_
        surv_times = [np.clip(out[1], a_min=None, a_max=times[-1]) for out in y]

        scores = [ np.abs(fn(t)-0.5) for fn, t in zip(surv_fn,surv_times)] 
        return np.array(scores) 
        
    def predict_one_sided_score(self, X,y):
        """
        y is for the form (event, outcome) 
        """
        
        surv_fn=self.model.predict_survival_function(X) #returns m step functions P(T>t)
        times=self.model.unique_times_
        surv_times = [np.clip(out[1], a_min=None, a_max=times[-1]) for out in y]

        scores = [ fn(t)-0.5 for fn, t in zip(surv_fn,surv_times)] 
        return np.array(scores) 


    def compute_prediction_set(self, X, qhat):

        surv_fn=self.model.predict_survival_function(X)
        y_tr_max=int(self.model.unique_times_[-1])
        cdf_values = np.array([[1-fn(np.clip(t, a_min=None, a_max=y_tr_max)) for t in range(1, y_tr_max+1)] for fn in surv_fn]) #cdf 

        m=len(X)
        lower_bounds, upper_bounds = [0.]*m, [0.]*m
        
        for i, cdf_value in enumerate(cdf_values):
            cdf_min, cdf_max = np.min(cdf_value), np.max(cdf_value)
        
            if cdf_min >= 0.5-qhat: lower_bounds[i]=0
            elif cdf_max <=0.5-qhat: lower_bounds[i]= y_tr_max
            else: 
                lower_bounds[i] = np.arange(1, y_tr_max+1)[cdf_value >= 0.5-qhat][0] 

            if cdf_max <= 0.5+qhat: upper_bounds[i]=y_tr_max
            elif cdf_min >=0.5+qhat: upper_bounds[i]=0 
            else: 
                upper_bounds[i] = np.arange(1, y_tr_max+1)[cdf_value <= 0.5 +qhat][-1]
        return list(zip(lower_bounds, upper_bounds))  

    def compute_prediction_lb(self, X, qhat):
        m=len(X)
        bounds = self.compute_prediction_set(X, qhat)
        lower_bounds, _ = zip(*bounds)
        return list(zip(lower_bounds, [None]*m))


class quantile_based_score(object):
    """
    Quantile-based score 
    Learns a model of the censored outcome given X (NOT T given X) 
    Fits a quantile regressor to censored survival times 
    """
    def __init__(self, model_high, model_low):
        """
        model_high is a scikit-learn object e.g. QuantileRegressor() 
        model_low is a scikit-learn object e.g. QuantileRegressor() 
        """
        self.model_high = model_high
        self.model_low = model_low

    def fit(self, X,y):
        """
        y is for the form (event, outcome) 
        """
        y_out=np.array([y_[1] for y_ in y])
        self.model_low.fit(X,y_out)
        self.model_high.fit(X,y_out)
        
    def predict_score(self, X,y):
        y_out=np.array([y_[1] for y_ in y])
        quant_low = self.model_low.predict(X)-y_out
        quant_high = y_out-self.model_high.predict(X)
        return np.maximum(quant_low, quant_high) 
        
    def predict_one_sided_score(self, X,y):
        y_out=np.array([y_[1] for y_ in y])
        return self.model_low.predict(X)-y_out

    def compute_prediction_set(self, X, qhat):
        lower_bounds = self.model_low.predict(X) - qhat
        upper_bounds = self.model_high.predict(X) + qhat
        return list(zip(lower_bounds, upper_bounds))

    def compute_prediction_lb(self, X, qhat):
        m=len(X)
        bounds = self.compute_prediction_set(X, qhat)
        lower_bounds, _ = zip(*bounds)
        return list(zip(lower_bounds, [None]*m))


class reg_quantile_based_score(quantile_based_score):
    """
    Version of the quantile based-score where the predictive model is fitted using only non-censored individuals
    Learns a model of T given X conditional on event=1 
    """
    def __init__(self, model_high, model_low):
        #inherits from quantile_based_score

        quantile_based_score.__init__(self, model_high, model_low)

    def fit(self, X,y):
        """
        y is for the form (event, outcome) 
        """
        event, y_out = zip(*y) 
        event, y_out = np.array(event), np.array(y_out) 
        y_out1=y_out[event==1] 
        X1=X[event==1] 
        self.model_low.fit(X1,y_out1)
        self.model_high.fit(X1,y_out1)
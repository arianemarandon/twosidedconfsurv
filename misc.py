import numpy as np

def compute_error_bounds(y, bounds_pred, event_pred):
    """
    Lower bound and upper bound on the coverage error given: 

    y: a list of tuples of the form (event,outcome) for the test sample 
    bounds_pred: a list of tuples of the form (LB, UB) (predictive bounds on the true outcome) for the test sample
    event_pred: event predictions for the test sample 
    """
    m=len(y)
    event_test, y_test  = zip(*y) 
    y_test=np.array(y_test)
    
    lower_bounds, upper_bounds = zip(*bounds_pred) 
    lower_bounds, upper_bounds = np.array(lower_bounds), np.array(upper_bounds)
    
    event_test = np.array(list(event_test))*1
    event_pred = np.array(list(event_pred))
    
    true_positives = (event_test==1)
    true_negatives = (event_test==0)
    positives = (event_pred==1)
    negatives = (event_pred==0)

    err1 = np.sum((y_test[positives & true_positives] < lower_bounds[positives & true_positives]) \
                  | (y_test[positives & true_positives] > upper_bounds[positives & true_positives]))
    err0 = np.sum(y_test[negatives] < lower_bounds[negatives]) 
    err_clf = np.sum(positives & true_negatives) 
    err_lb = (err1+err0+err_clf)/m 

    err11 = np.sum((y_test[positives & true_positives] < lower_bounds[positives & true_positives]) \
                  | (y_test[positives & true_positives] > upper_bounds[positives & true_positives]))
    err10 = np.sum(y_test[negatives & true_positives] < lower_bounds[negatives & true_positives])
    err01 = np.sum(y_test[positives & true_negatives] > upper_bounds[positives & true_negatives])
    err_ub = (err11+err10+err01)/m 

    return err_lb, err_ub

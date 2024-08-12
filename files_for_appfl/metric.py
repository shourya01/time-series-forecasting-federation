import numpy as np

def mape(y_true, y_pred):
    
    assert len(y_true.shape) == len(y_pred.shape), "Cannot calculate metric when y_true and y_pred have different number of dimensions."
    
    if len(y_true.shape) == 3:
        num = np.abs( y_true[:,-1,-1] - y_pred[:,-1,-1] )
        den = np.abs( y_true[:,-1,-1] )
        return 100 * np.mean( num/den )   
    else:
        if len(y_true.shape) == 2:
            num = np.abs( y_true[-1,-1] - y_pred[-1,-1] )
            den = np.abs( y_true[-1,-1] )
            return 100 * np.mean( num/den )   
        else:
            raise ValueError('MAPE: Wrong dimensions of input and/or output.')     
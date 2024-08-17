import numpy as np

EPSILON = 1e-5

def mape(target, prediction):
    
    target0 = np.expand_dims(target[:,:,0], axis=-1)
    min, max = np.expand_dims(target[:,:,1], axis=-1), np.expand_dims(target[:,:,2], axis=-1)
    prediction = (max-min)*prediction + min
    
    num = np.abs(target0 - prediction)
    den = np.abs(target0) + EPSILON
    
    return 100 * np.mean( num / den )
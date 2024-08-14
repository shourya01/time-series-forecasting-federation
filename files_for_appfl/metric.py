import numpy as np

EPSILON = 1e-5

def mape(target, prediction):
    
    target0 = np.expand_dims(target[:,:,0], axis=-1)
    mean, std = np.expand_dims(target[:,:,1], axis=-1), np.expand_dims(target[:,:,2], axis=-1)
    target = std*target0 + mean
    
    num = np.abs(target - prediction)
    den = np.abs(prediction) + EPSILON
    
    return 100 * np.mean( num / den )
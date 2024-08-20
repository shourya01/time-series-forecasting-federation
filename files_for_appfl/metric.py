import numpy as np

EPSILON = 1e-5

def mape(target, prediction, normalization_type = 'minmax'):
    
    if normalization_type == 'minmax':
        target0 = np.expand_dims(target[:,:,0], axis=-1)
        min, max = np.expand_dims(target[:,:,1], axis=-1), np.expand_dims(target[:,:,2], axis=-1)
        prediction = (max-min)*prediction + min
    else:
        if normalization_type == 'z':
            target0 = np.expand_dims(target[:,:,0], axis=-1)
            mean, std = np.expand_dims(target[:,:,1], axis=-1), np.expand_dims(target[:,:,2], axis=-1)
            prediction = std*prediction + mean
        else:
            raise ValueError('normalize_type must be either of <minmax> or <z>')
    
    num = np.abs(target0[:,-1,:] - prediction[:,-1,:])
    den = np.abs(target0[:,-1,:]) + EPSILON
    
    return 100 * np.mean( num / den )
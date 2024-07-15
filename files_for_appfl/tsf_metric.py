import numpy as np
from typing import Union, List, Tuple

EPSILON = 1e-8

def mape_metric(target: Union[List,Tuple,np.ndarray], prediction: Union[List,Tuple,np.ndarray]):
    
    mae = []
    
    if isinstance(target[0],list):
        if target[0][0].shape == prediction[0].shape:
            for tar, pred in zip(target,prediction):
                mae.append(np.mean(np.abs(tar[0]-pred)/(np.abs(pred)+EPSILON)))
        else:
            if target[0][1].shape == prediction[0].shape:
                for tar, pred in zip(target,prediction):
                    mae.append(np.mean(np.abs(tar[1]-pred)/(np.abs(pred)+EPSILON)))
            else:
                raise ValueError('Shape of prediction doesnt match any element in target.')
    else:
        if isinstance(target[0],np.ndarray):
            if target[0].shape == prediction[0].shape:
                for tar, pred in zip(target,prediction):
                    mae.append(np.mean(np.abs(tar-pred)/(np.abs(pred)+EPSILON)))
            else:
                raise ValueError('Shape of prediction doesnt match any element in target.')
        else:
            raise ValueError('Targets are of incompatible data type.')
        
    return 100 * (sum(mae) / len(mae)) # mean of collected mae's
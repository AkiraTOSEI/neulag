import os

from omegaconf import DictConfig
import numpy as np


def twobody_task_simulator(X:np.ndarray, cfg:DictConfig, only_Y=True, logscale=False)->np.ndarray:
    '''calculate U and T after De-standardization
    '''
    data_dir = cfg.general.data_dir
    assert X.shape[1]==4 and len(X.shape)==2

    # De-standardization
    d = np.load(os.path.join(data_dir, 'stats.npz'))
    mean_array = d['mean_array'].astype(np.float32)
    std_array = d['std_array'].astype(np.float32)
    raw_X = X*std_array+mean_array
    #raw_X = X
    
    T = np.square(raw_X[:, 2:4]).sum(axis=-1)/2
    U = -1./np.sqrt(np.square(raw_X[:, 0:2]).sum(axis=-1))
    if logscale:
        U = -np.log(-U)
        T = np.log(T)
        
    Y = np.concatenate([U[:,np.newaxis],T[:, np.newaxis]],axis=-1)

        
    if only_Y:
        return Y
    else:
        return Y,T,U
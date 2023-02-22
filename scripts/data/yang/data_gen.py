import os
import sys 
sys.path.append('../../simulators/')

import numpy as np
import pandas as pd
from yang import creat_mm_dataset, predict_ensemble_for_all

def generate_meta_material(data_num:int)->np.array:
    x_dim = 14
    # Generate random number
    data_x = np.random.uniform(size=(data_num,x_dim), low=-1, high=1)
    print('data_x now has shape:', np.shape(data_x))
    return data_x

def createY_from_X(model_dir:str, data_dir:str):
    creat_mm_dataset(data_dir=data_dir, model_dir=model_dir)
    _dir = os.path.join(data_dir, 'state_dicts')
    predict_ensemble_for_all(_dir, './', no_plot=True) 

def data_division(data_dir:str):
    x_data = pd.read_csv(os.path.join(data_dir,'data_x.csv'),header=None, delimiter=' ')
    x_data.to_csv(os.path.join(data_dir,'data_x.csv'),index=False, header=False)
    x_data[:10000].to_csv(os.path.join(data_dir,'train_x.csv'),index=False, header=False)
    x_data[10000:].to_csv(os.path.join(data_dir,'test_x.csv'),index=False, header=False)
    
    y_data = pd.read_csv(os.path.join(data_dir,'data_y.csv'),header=None, delimiter=' ')
    y_data.to_csv(os.path.join(data_dir,'data_y.csv'),index=False, header=False)
    y_data[:10000].to_csv(os.path.join(data_dir,'train_y.csv'),index=False, header=False)
    y_data[10000:].to_csv(os.path.join(data_dir,'test_y.csv'),index=False, header=False)

if __name__ == '__main__':

    # create X data
    ndata = 10500   # Training and validation set
    data_x = generate_meta_material(ndata)
    os.makedirs('dataIn',exist_ok=True)
    np.savetxt('dataIn/data_x.csv', data_x)

    # create Y and train/test division
    data_dir = ''
    model_dir = '../../simulators/data4deng/'
    createY_from_X(model_dir, data_dir)
    data_division(os.path.join(data_dir,'dataIn'))


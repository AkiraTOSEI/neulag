import numpy as np
import pandas as pd
import os
import sys

sys.path.append('../../src')
from eval.evaluation import AMD_resim_body

def generate_meta_material(data_num):
    x_dim = 14
    # Generate random number
    data_x = np.random.uniform(size=(data_num,x_dim), low=-1, high=1)
    print('data_x now has shape:', np.shape(data_x))
    return data_x

def create_data(fname='data',ftest='test',ndata=10000,ntest=500, create_test=False):
    data_x = generate_meta_material(ndata)
    data_y = AMD_resim_body(data_x, "../../src/simulators/data4yang/")
    pd.DataFrame(data_x).to_csv(f'./{fname}_x.csv',index=False, header=False)
    pd.DataFrame(data_y).to_csv(f'./{fname}_y.csv',index=False, header=False)

    if create_test:
      data_x = generate_meta_material(ntest)
      data_y = AMD_resim_body(data_x, "../../src/simulators/data4yang/")
      pd.DataFrame(data_x).to_csv(f'./{ftest}_x.csv',index=False, header=False)
      pd.DataFrame(data_y).to_csv(f'./{ftset}_y.csv',index=False, header=False)

if __name__ == '__main__':
  create_data(create_test=True)
  
  # for medium surrogator model
  create_data(fname='data-mid',ndata=400,create_test=False)



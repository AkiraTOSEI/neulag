import os
import yaml
from omegaconf import OmegaConf
import time
import shutil
from tqdm import tqdm

import pandas as pd
import numpy as np

import sys
sys.path.append('../../src')

from utils.config import cfg_add_dirs
from utils.task_utils import task_data_size
from forward_train.NALike.ADM_forward_train import train_forward_model 
from backward_optimization.NALike.AMD_BwOpt import backward_optimization
from eval.evaluation import evaluation 
from exp_scripts.method_test import method_test


def baseline_test(
    task:str, 
    model_name:str, 
    test_run_mode:bool=False, 
    only_no_constraint=False, 
    only_NeuralLagrangian=False

):
  
   na_config_path = f"../config_script_files/NA_{task}_{model_name}-model_config.yaml"
   nl_config_path = f"../config_script_files/NeuralLagrangian_{task}_{model_name}-model_config.yaml"
   
   all_results = []
   num_trial = 1
   num_samples=200
   path = '/home/afujii/awesome_physics_project/workspace/setting_files/random_ids.npy'
   id_list = pd.read_csv(path)['0'].values.tolist()
   eq_IDs = range(9)
   mate_feat_dic = {"Stack":5,"Shell":8,"ADM":14}
   no_mate_IDs = range(mate_feat_dic[task])
   target_ids = range(500)
   nm_id_list = id_list[:50]
   eq_id_list = id_list[:50]
   
   ### target id の修正
   #target_ids = list(np.array([target_ids[:250][::-1], target_ids[250:]]).T.reshape(-1))
   #nm_id_lsit = list(np.array([id_list[:50][::-1], id_list[50:]]).T.reshape(-1))
   #eq_id_lsit = list(np.array([id_list[:50][::-1], id_list[50:]]).T.reshape(-1))
   
   if test_run_mode==True:
     id_list,eq_id_list,nm_id_list = [0],[0],[0]
     eq_IDs = [0]
     no_mate_IDs = [0]
     num_trial = 1
     num_samples=2
     target_ids = range(2)
   
   # forward model train
   cfg = OmegaConf.load(nl_config_path)
   cfg_add_dirs(cfg)
   train_forward_model(cfg, test=test_run_mode)
   
   ### backward experiments
   result_df = pd.DataFrame([])
   if only_no_constraint == 'constraint_only':
     target_ids = []
   for target_id in target_ids:
       '''
       No constraint
       '''
       condition_dict = {
         'condition_type':None,
         'trial_id':0,
         'target_id':target_id,
       } 
       target_id = int(target_id)
       if not only_NeuralLagrangian:
           result_df = pd.concat([result_df, method_test(na_config_path, 'NA',test_run_mode, condition_dict)])
       result_df = pd.concat([result_df, method_test(nl_config_path, 'NeuralLagrangian',test_run_mode, condition_dict)])
       result_df.to_csv(f'{task}_{model_name}_normal_results.csv',index=False)
       if os.path.exists('lightning_logs'):
           shutil.rmtree('lightning_logs')
   
   if only_no_constraint==True:
       return None

   result_df = pd.DataFrame([])
   for target_id in nm_id_list:
       '''
       NoMaterial
       '''
       for no_mate_ID in no_mate_IDs:
           condition_dict = {
             'num_samples':num_samples,
             'condition_type':'NoMaterial',
             'NM-feat':no_mate_ID,
             #'condition_type':'Constrain',
             #'eq_constraint':eq_ID,
             #'condition_type':'input_boundary',
             #'input_boundary':0.0,
             'trial_id':0,
             'target_id':target_id,
           } 
           target_id = int(target_id)
           result_df = pd.concat([result_df, method_test(nl_config_path, 'NeuralLagrangian',test_run_mode, condition_dict)])
           result_df = pd.concat([result_df, method_test(na_config_path, 'NA',test_run_mode, condition_dict)])
           result_df.to_csv(f'{task}_{model_name}_NoMate_results.csv',index=False)
           if os.path.exists('lightning_logs'):
               shutil.rmtree('lightning_logs')
   
   
   result_df = pd.DataFrame([])
   for target_id in eq_id_list:
       '''
       Equation Constraint
       '''
       for eq_ID in eq_IDs:
           condition_dict = {
             'num_samples':num_samples,
             #'contition_type':'NoMaterial',
             #'NM-feat':no_mate_ID,
             'condition_type':'Constrain',
             'eq_constraint':eq_ID,
             #'condition_type':'input_boundary',
             #'input_boundary':0.0,
             'trial_id':0,
             'target_id':target_id,
           } 
           target_id = int(target_id)
           result_df = pd.concat([result_df, method_test(nl_config_path, 'NeuralLagrangian',test_run_mode, condition_dict)])
           result_df = pd.concat([result_df, method_test(na_config_path, 'NA',test_run_mode, condition_dict)])
           result_df.to_csv(f'{task}_{model_name}_constrain_results.csv',index=False)
           if os.path.exists('lightning_logs'):
               shutil.rmtree('lightning_logs')
   

if __name__ == "__main__":  
    task = 'Stack'
    
    for model_name,only_no_constraint in zip(['medium','small','base','medium'],[True, True,True,'constraint_only']):
        baseline_test(
            task=task,
            model_name=model_name,
            test_run_mode=True,
            only_NeuralLagrangian=True,
            only_no_constraint=only_no_constraint 
        )

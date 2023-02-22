import os

import numpy as np
import pandas as pd
import torch
from torch import nn
from omegaconf.dictconfig import DictConfig

from simulators.chen import Stack_task_simulator
from simulators.peurifoy import Shell_task_simulator
from simulators.yang import creat_mm_dataset, predict_ensemble_for_all


def calculate_ReSimulationError(cfg: DictConfig,test_run_mode:bool):
    """
    Args:
        cfg(DictConfig) : config
    """
    y_gt = np.load(os.path.join(cfg.output_dirs.result_dir, "targets.npz"))["y_gt"]
    opt_input = np.load(os.path.join(cfg.output_dirs.result_dir, "optimized_input.npy"))
    
    if test_run_mode:
        y_gt = y_gt[:10]
        opt_input = opt_input[:10]
    
    eval_size = cfg.inverse_problem.eval_batch_size
    opt_input = opt_input[:eval_size]
    y_gt = y_gt[:eval_size]

    with torch.no_grad():
        # Re-simulation error
        if cfg.general.task=="Chen":
            y_resim = Stack_task_simulator(opt_input, cfg)
        elif cfg.general.task=="Peurifoy":
            y_resim = Shell_task_simulator(opt_input, cfg)            
        else:
            raise Exception("task name is invalid")

        resimulation_error = torch.square(torch.tensor(y_gt)-torch.tensor(y_resim)).mean(dim=-1)
        resim_mean = resimulation_error.mean().detach().cpu().numpy()
        resim_std = resimulation_error.std().detach().cpu().numpy()

    return resim_mean, resim_std, y_resim


def createY_from_X(model_dir:str, data_dir:str):
    creat_mm_dataset(data_dir=data_dir, model_dir=model_dir)
    states_dict_dir = os.path.join(model_dir, 'state_dicts')
    assert(os.path.exists(states_dict_dir ))
    predict_ensemble_for_all(model_dir=states_dict_dir, Xpred_file_dirs='./', no_plot=True)
    

def AMD_resimulation_error(cfg: DictConfig,test_run_mode:bool):
    '''
    resimulation error calculation on AMD task
    Args:
        cfg(DictConfig) : config file
        test_run_mode(bool) : use test run mode that is for debug
    '''
    
    # create Y and train/test division
    workspace_dir = './ws'
    os.makedirs(os.path.join(workspace_dir,'dataIn'), exist_ok=True)
    model_dir = '../scripts/simulators/data4yang/'
    

    # re-save optimized x-data to evaluation workspace
    opt_input = np.load(os.path.join(cfg.output_dirs.result_dir, "optimized_input.npy"))
    if test_run_mode:
        opt_input = opt_input[:10]
    pd.DataFrame(opt_input).to_csv(os.path.join(workspace_dir,'dataIn','data_x.csv'),index=False, header=False,sep=',')

    # re-simulation
    createY_from_X(model_dir, data_dir=workspace_dir)

    # get prediction results
    y_resim = np.array(pd.read_csv(os.path.join(workspace_dir,'dataIn','data_y.csv'),header=None,delimiter=' '))
    y_gt = np.load(os.path.join(cfg.output_dirs.result_dir, "targets.npz"))["y_gt"]
    if test_run_mode:
        y_gt = y_gt[:10]

    resimulation_error = torch.square(torch.tensor(y_gt)-torch.tensor(y_resim)).mean(dim=-1)
    resim_mean = resimulation_error.mean().detach().cpu().numpy()
    resim_std = resimulation_error.std().detach().cpu().numpy()

    return resim_mean, resim_std, y_resim


def evaluation(cfg: DictConfig,test_run_mode=False)->np.array:
    """evaluation
    """
    if cfg.general.task=='AMD':
        resimulation_error_mean, resimulation_error_std, y_resim = AMD_resimulation_error(cfg,test_run_mode)
    elif cfg.general.task in ['Stack','Shell']
        resimulation_error_mean, resimulation_error_std, y_resim = calculate_ReSimulationError(cfg,test_run_mode)
    else:
        raise Exception('')
    df0 = pd.read_csv(
        os.path.join(cfg.output_dirs.result_dir, "ForwardModelResults.csv")
    )
    df1 = pd.read_csv(os.path.join(cfg.output_dirs.result_dir, "ReSimulationError.csv"))
    df = pd.concat([df0, df1])
    df.rename(columns={"Unnamed: 0": "item"}, inplace=True)
    df.set_index("item")
    df.to_csv(os.path.join(cfg.output_dirs.result_dir, "results.csv"))

    # 別名で保存する。
    targets = np.load(os.path.join(cfg.output_dirs.result_dir, "targets.npz"))
    opt_input = np.load(os.path.join(cfg.output_dirs.result_dir, "optimized_input.npy"))

    test_result_dir = os.path.join(cfg.output_dirs.result_dir, cfg.general.exp_name)
    os.makedirs(test_result_dir, exist_ok=True)
    np.savez(
        os.path.join(test_result_dir, "result_arrays.npz"),
        y_gt=targets["y_gt"],
        x_gt=targets["x_gt"],
        opt_input=opt_input,
    )
    df.to_csv(os.path.join(test_result_dir, "results.csv"))
    
    return resimulation_error_mean, resimulation_error_std, y_resim

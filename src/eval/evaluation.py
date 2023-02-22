import os
from typing import Any, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig

from simulators.chen import Stack_task_simulator
from simulators.peurifoy import Shell_task_simulator
from simulators.yang import creat_mm_dataset, predict_ensemble_for_all


def calculate_ReSimulationError(
    cfg: Union[DictConfig, ListConfig],
    opt_input: np.ndarray,
    file_name: str,
    test_run_mode: bool,
):
    """
    calculate resimulation error with ground truth simulator f
    """
    y_gt = np.load(os.path.join(cfg.output_dirs.result_dir, "targets.npz"))["y_gt"]
    y_gt = np.concatenate([y_gt[:1]] * len(opt_input), axis=0)
    sample_size = len(opt_input)

    if test_run_mode:
        y_gt = y_gt[:10]
        opt_input = opt_input[:10]

    eval_size = cfg.inverse_problem.num_T_samples
    opt_input = opt_input[:eval_size]
    y_gt = y_gt[:eval_size]

    with torch.no_grad():
        # Re-simulation with ground truth simulator f
        if cfg.general.task == "Stack":
            y_resim = Stack_task_simulator(opt_input, cfg)
        elif cfg.general.task == "Shell":
            y_resim = Shell_task_simulator(opt_input, cfg)
        elif cfg.general.task == "ADM":
            y_resim = AMD_task_simulator(opt_input, cfg)
        else:
            raise Exception("task name is invalid")

        resimulation_error = ResimError(y_gt, y_resim)

    resim_1 = resimulation_error[0].detach().cpu().numpy()
    resim_mean = resimulation_error.mean().detach().cpu().numpy()
    resim_T = resimulation_error.min().detach().cpu().numpy()

    # Forward model loss
    loss_file_name = file_name.replace("optimized_input", "FW_loss")
    fw_loss = np.load(os.path.join(cfg.output_dirs.result_dir, loss_file_name))
    fw_loss_all_mean = fw_loss.mean()
    fw_loss = fw_loss[:eval_size]
    fw_loss_T_mean = fw_loss.mean()
    fw_loss_min = fw_loss[0]
    print(fw_loss_min, fw_loss.min(), fw_loss[-1])
    assert fw_loss_min == fw_loss.min()

    print(file_name)
    print(f"evaluation resimulation error T :{eval_size}, sample_size:{sample_size}")
    print(f"resim_1:{resim_1}, resim_T:{resim_T}")
    print(
        f"fw_loss_all_mean:{fw_loss_all_mean}, fw_loss_T_mean:{fw_loss_T_mean}, fw_loss_min:{fw_loss_min}"
    )
    print("")

    return resim_mean, resim_1, resim_T, y_resim, fw_loss


def createY_from_X(model_dir: str, data_dir: str):
    creat_mm_dataset(data_dir=data_dir, model_dir=model_dir)
    states_dict_dir = os.path.join(model_dir, "state_dicts")
    assert os.path.exists(states_dict_dir)
    predict_ensemble_for_all(
        model_dir=states_dict_dir, Xpred_file_dirs="./", no_plot=True
    )


def ResimError(y_gt, y_resim):
    return torch.square(torch.tensor(y_gt) - torch.tensor(y_resim)).mean(dim=-1)


def AMD_task_simulator(opt_input: np.ndarray, cfg: Union[DictConfig, ListConfig]):
    """
    resimulation error calculation on AMD task
    Args:
        cfg(DictConfig) : config file
        test_run_mode(bool) : use test run mode that is for debug
    """

    y_gt = np.load(os.path.join(cfg.output_dirs.result_dir, "targets.npz"))["y_gt"]

    eval_size = cfg.inverse_problem.num_T_samples
    opt_input = opt_input[:eval_size]
    y_gt = y_gt[:eval_size]

    y_resim = AMD_resim_body(opt_input, cfg.simulator_files_dir["ADM"])

    return y_resim


def AMD_resim_body(opt_input: np.ndarray, model_dir: str):
    workspace_dir = "./ws"
    os.makedirs(os.path.join(workspace_dir, "dataIn"), exist_ok=True)

    pd.DataFrame(opt_input).to_csv(
        os.path.join(workspace_dir, "dataIn", "data_x.csv"),
        index=False,
        header=False,
        sep=",",
    )

    # re-simulation
    createY_from_X(model_dir, data_dir=workspace_dir)

    # get prediction results
    y_resim = np.array(
        pd.read_csv(
            os.path.join(workspace_dir, "dataIn", "data_y.csv"),
            header=None,
            delimiter=" ",
        )
    )
    return y_resim


def get_OptInput_files(
    cfg: Union[DictConfig, ListConfig],
) -> Tuple[List[str], List[str]]:
    """get paths of file in which optimized input saved"""
    dir_path = os.path.join(cfg.output_dirs.result_dir)
    files = os.listdir(dir_path)
    paths, file_infos = [], []
    for _f in files:
        if _f.startswith("optimized_input") and _f.endswith(".npy"):
            paths.append(os.path.join(dir_path, _f))
            file_infos.append(_f.replace("optimized_input_", "").replace(".npy", ""))
    return paths, file_infos


def evaluation(
    cfg: Union[DictConfig, ListConfig], test_run_mode: bool = False
) -> Tuple[List[Any], List[str]]:
    """evaluation"""
    num_eval_sample_list = cfg.inverse_problem.num_eval_sample_list

    if cfg.inverse_problem.method == "NA" and len(num_eval_sample_list) > 1:
        raise Exception("For NA, single eval_samples can be evaluated")

    result_dict = {}
    opt_input_file_paths, file_infos = get_OptInput_files(cfg)
    for file_path, info in zip(opt_input_file_paths, file_infos):
        opt_input = np.load(file_path)
        fname = file_path.split("/")[-1]
        output = calculate_ReSimulationError(cfg, opt_input, fname, test_run_mode)
        resim_mean, resim_1, resim_T, y_resim, fw_loss = output
        result_dict[f"{info}-resim_mean"] = resim_mean
        result_dict[f"{info}-resim_1"] = resim_1
        result_dict[f"{info}-resim_T"] = resim_T
        result_dict[f"{info}-top-T_FW-loss-mean"] = fw_loss.mean()
        result_dict[f"{info}-top-T_FW-loss-min"] = fw_loss.min()
        np.savez(
            os.path.join(
                cfg.output_dirs.result_dir, f"{fname.replace('.npy','')}_loss_resim.npz"
            ),
            y_resim=y_resim,
            fw_loss=fw_loss,
        )

    cols = [key for key in result_dict.keys()]
    results = [result_dict[key] for key in result_dict.keys()]
    result_df = pd.DataFrame(np.array([results]), columns=cols)
    result_df.to_csv(os.path.join(cfg.output_dirs.result_dir, "results.csv"))

    return results, cols

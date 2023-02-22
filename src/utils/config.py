import os
from typing import Union

import omegaconf
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig


def cfg_add_dirs(cfg: Union[DictConfig, ListConfig]):
    """add output directory for a forward model, Inverse problem results on config"""

    # csv output file saved dirctory
    result_dir = os.path.join(
        "/home/afujii/awesome_physics_project/outputs",
        f"{cfg.general.task}_result",
        cfg.inverse_problem.method,
        cfg.general.exp_name,
    )

    # simulator saved directory
    cfg.general.data_dir = os.path.join(cfg.general.master_dir, cfg.general.data_dir)
    for key in cfg.simulator_files_dir.keys():
        cfg.simulator_files_dir[key] = os.path.join(
            str(cfg.general.master_dir), str(cfg.simulator_files_dir[key])
        )

    # model saved directory
    model_dir = os.path.join(
        str(cfg.general.master_dir),
        "models",
        f"{cfg.general.task}_result",
        str(cfg.inverse_problem.method),
    )
    if cfg.inverse_problem.method in ["NA", "NeuralLagrangian"]:
        fw_method = "NALike"
    elif cfg.inverse_problem.method == "INN":
        raise Exception("Upps, INN have not implemented yet.")
        # fw_method =
    else:
        raise Exception(
            "please specify correct method name. current value:",
            cfg.inverse_problem.method,
        )
    fw_model_dir = os.path.join(
        cfg.general.master_dir, "models", f"{cfg.general.task}_result", fw_method
    )
    fw_model_dir = os.path.join(fw_model_dir, "foward_model")
    fw_ckpt_dir = os.path.join(model_dir, "foward_model_ckpt")
    iv_ckpt_dir = os.path.join(model_dir, "inverse_problem_ckpt")
    os.makedirs(fw_model_dir, exist_ok=True)
    os.makedirs(fw_ckpt_dir, exist_ok=True)
    os.makedirs(iv_ckpt_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    # add new keys on config file
    omegaconf.OmegaConf.set_struct(cfg, False)
    cfg.output_dirs.result_dir = result_dir
    cfg.output_dirs.fw_model_dir = fw_model_dir
    cfg.output_dirs.fw_ckpt_dir = fw_ckpt_dir
    cfg.output_dirs.iv_ckpt_dir = iv_ckpt_dir
    cfg.output_dirs.iv_ckpt_dir = iv_ckpt_dir
    cfg.experiment_name = cfg.general.exp_name
    omegaconf.OmegaConf.set_struct(cfg, True)

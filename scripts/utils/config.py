import os

import omegaconf
from omegaconf import OmegaConf

def cfg_add_dirs(cfg:omegaconf.dictconfig.DictConfig):
    """add output directory for a forward model, Inverse problem results on config
    """
    result_dir = os.path.join(
        f"./{cfg.general.task}_result", cfg.inverse_problem.method
    )
    fw_model_dir = os.path.join(result_dir, "foward_model")
    fw_ckpt_dir = os.path.join(result_dir, "foward_model_ckpt")
    iv_ckpt_dir = os.path.join(result_dir, "inverse_problem_ckpt")

    os.makedirs(fw_model_dir, exist_ok=True)
    os.makedirs(fw_ckpt_dir, exist_ok=True)
    os.makedirs(iv_ckpt_dir, exist_ok=True)

    omegaconf.OmegaConf.set_struct(cfg, False)
    cfg.output_dirs.result_dir = result_dir
    cfg.output_dirs.fw_model_dir = fw_model_dir
    cfg.output_dirs.fw_ckpt_dir = fw_ckpt_dir
    cfg.output_dirs.iv_ckpt_dir = iv_ckpt_dir
    cfg.experiment_name = cfg.general.exp_name
    omegaconf.OmegaConf.set_struct(cfg, True)

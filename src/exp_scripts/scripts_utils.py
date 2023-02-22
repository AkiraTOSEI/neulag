import typing
from typing import Union

import omegaconf
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig

from utils.config import cfg_add_dirs


def config_test_function(cfg, method):
    """
    test function for config setting
    """
    if method == "NA":
        assert cfg.inverse_problem.method == "NA"
    elif method == "NeuralLagrangian":
        assert cfg.inverse_problem.method == "NeuralLagrangian"

    if (
        cfg.inverse_problem.method_parameters.optimization_mode
        == "single_target_for_multi_solution"
    ):
        pass
    elif cfg.inverse_problem.method_parameters.optimization_mode == "single_target":
        print(" ----------------------------------------")
        print(" ----------------------------------------")
        print(" ----------------------------------------")
        print(" ----------------------------------------")
        print("            WARNING ")
        print(
            f" you use {cfg.inverse_problem.method_parameters.optimization_mode} mode for {cfg.inverse_problem.method} "
        )
        print(" ----------------------------------------")
        print(" ----------------------------------------")
        print(" ----------------------------------------")
        print(" ----------------------------------------")


def experiment_name(
    cfg: Union[DictConfig, ListConfig],
    trial_id: int,
    constrain_name: str,
    test_run_mode: bool,
):
    """
    set input boudary related name
    """

    target_id = cfg.inverse_problem.target_id
    model_name = cfg.fw_model.model_name
    exp_name = f"ID-{target_id}_{constrain_name}__{model_name}-model_trial-{trial_id}__"

    if not cfg.inverse_problem.method_parameters.BL_coef == 0.1:
        bl_coef = cfg.inverse_problem.method_parameters.BL_coef
        exp_name += f"-BLcoef{bl_coef}"
    if not cfg.inverse_problem.method_parameters.iv_batch_size == 2048:
        batch_size = cfg.inverse_problem.method_parameters.iv_batch_size
        exp_name += f"-{batch_size}batch"

    if exp_name.endswith("__"):
        exp_name = exp_name[:-2]

    test = "__TEST" if test_run_mode else ""
    exp_name += test

    cfg.general.exp_name = exp_name
    print("-------------------------------------")
    print(" ", cfg.general.task, "-", cfg.inverse_problem.method, " | ", exp_name)
    print("-------------------------------------")


def read_constrains(cfg: Union[DictConfig, ListConfig], condition_dict: dict) -> str:

    omegaconf.OmegaConf.set_struct(cfg, False)
    cfg.inverse_problem.input_boundary.type = condition_dict["condition_type"]
    if cfg.inverse_problem.input_boundary.type == "Inputlimit":
        constrain_name = "IB"
        cfg.inverse_problem.input_boundary.input_boudary = condition_dict[
            "input_boundary"
        ]
    elif cfg.inverse_problem.input_boundary.type == "Constrain":
        eq_id = int(condition_dict["eq_constraint"])
        constrain_name = f"CS-Eq{eq_id}"
        cfg.inverse_problem.input_boundary.ConstrainEqId = eq_id
    elif cfg.inverse_problem.input_boundary.type == "NoMaterial":
        feat_id = int(condition_dict["NM-feat"])
        constrain_name = f"NM-Feat{feat_id}"
        cfg.inverse_problem.input_boundary.ConstrainNMId = condition_dict["NM-feat"]
    elif cfg.inverse_problem.input_boundary.type == "interpolation":
        orbital_id = cfg.inverse_problem.method_parameters.interpolation_ids.orbital_id
        (
            init_id,
            fin_id,
        ) = cfg.inverse_problem.method_parameters.interpolation_ids.init_fin
        constrain_name = f"Interpolation-Orbit{orbital_id}-{init_id}-{fin_id}"
    elif cfg.inverse_problem.input_boundary.type is None:
        constrain_name = "No-constrain"
    else:
        boundary_method = cfg.inverse_problem.input_boundary.type
        raise Exception('"boundary_method" is invalid. You input:', boundary_method)
    omegaconf.OmegaConf.set_struct(cfg, True)

    return constrain_name


def ReflectExpCondition2config(
    cfg_path: str, method: str, test_run_mode: bool, condition_dict: dict
) -> typing.Tuple[Union[DictConfig, ListConfig], str]:
    """
    add or change config to fit the experiment
    """
    # read config
    cfg = OmegaConf.load(cfg_path)

    # read and reflect setting
    cfg.inverse_problem.target_id = condition_dict["target_id"]
    if "iv_batch_size" in condition_dict.keys():
        cfg.inverse_problem.method_parameters.iv_batch_size = condition_dict[
            "iv_batch_size"
        ]
    constrain_name = read_constrains(cfg, condition_dict)

    # add experiment name
    trial_id = condition_dict["trial_id"]
    experiment_name(cfg, trial_id, constrain_name, test_run_mode)

    # add directory on config
    cfg_add_dirs(cfg)

    # test function
    config_test_function(cfg, method)

    return cfg, constrain_name

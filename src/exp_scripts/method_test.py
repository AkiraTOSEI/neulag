import pandas as pd

from eval.evaluation import evaluation
from exp_scripts.scripts_utils import ReflectExpCondition2config
from inverse_optimization.inv_opt import inverse_optimization
from utils.task_utils import experiment_skip


def method_test(cfg_path: str, method: str, test_run_mode: bool, condition_dict: dict):

    # 条件を変える

    cfg, constrain_name = ReflectExpCondition2config(
        cfg_path, method, test_run_mode, condition_dict
    )

    # read condition
    trial_id = condition_dict["trial_id"]
    target_id = condition_dict["target_id"]

    if experiment_skip(cfg, test_run_mode):
        print(f"{cfg.general.exp_name} is already done. Skipped!!!")
        return pd.DataFrame([])

    # solving inverse problem
    time_cols, time_results = inverse_optimization(cfg, test=test_run_mode)
    results, cols = evaluation(cfg, test_run_mode=test_run_mode)
    # type: ignore
    # save metrics
    if method == "NA":
        method_name = "NeuralAdjoint"
    else:
        method_name = method
    exp_name = cfg.general.exp_name
    basic_metrics = [method_name, target_id, trial_id, constrain_name, exp_name]
    basic_cols = ["method", "target_id", "trial_id", "constrain_name", "exp_name"]
    metrics = basic_metrics + time_results + results
    metrics_cols = basic_cols + time_cols + cols

    return pd.DataFrame([metrics], columns=metrics_cols)

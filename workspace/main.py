import os
import shutil
import sys

import pandas as pd
from omegaconf import OmegaConf

sys.path.append("../src")

from eval.evaluation import evaluation
from exp_scripts.method_test import method_test
from forward_train.forward_train import train_forward_model
from utils.config import cfg_add_dirs


def baseline_test(
    task: str,
    model_name: str,
    test_run_mode: bool = False,
    only_NeuralLagrangian = False,
    only_no_const = True
):

    na_config_path = f"configs/NA_{task}_{model_name}-model_config.yaml"
    nl_config_path = f"configs/NeuralLagrangian_{task}_{model_name}-model_config.yaml"
    

    num_samples = 200
    eq_IDs = range(9)
    mate_feat_dic = {"Stack": 5, "Shell": 8, "ADM": 14}
    no_mate_IDs = range(mate_feat_dic[task])
    target_ids = range(500)
    nm_id_list = range(50)
    eq_id_list = range(50)

    if test_run_mode:
        id_list, eq_id_list, nm_id_list = [0], [0], [0]
        eq_IDs = [0]
        no_mate_IDs = [0]
        num_samples = 2
        target_ids = range(2)

    # forward model train
    cfg = OmegaConf.load(nl_config_path)
    cfg_add_dirs(cfg)
    train_forward_model(cfg, test=test_run_mode)

    # backward experiments

    result_df = pd.DataFrame([])
    for target_id in target_ids:
        """
        No constraint
        """
        condition_dict = {
            "condition_type": None,
            "trial_id": 0,
            "target_id": target_id,
        }
        target_id = int(target_id)
        if not only_NeuralLagrangian:
            result_df = pd.concat(
                [
                    result_df,
                    method_test(na_config_path, "NA", test_run_mode, condition_dict),
                ]
            )
        result_df = pd.concat(
            [
                result_df,
                method_test(
                    nl_config_path, "NeuralLagrangian", test_run_mode, condition_dict
                ),
            ]
        )
        result_df.to_csv(f"{task}_{model_name}_normal_results.csv", index=False)
        if os.path.exists("lightning_logs"):
            shutil.rmtree("lightning_logs")

    if only_no_const:
        return
    """
    Hard Constraint (No Material)
    """
    result_df = pd.DataFrame([])
    for target_id in nm_id_list:

        for no_mate_ID in no_mate_IDs:
            condition_dict = {
                "num_samples": num_samples,
                "condition_type": "NoMaterial",
                "NM-feat": no_mate_ID,
                "trial_id": 0,
                "target_id": target_id,
            }
            target_id = int(target_id)
            result_df = pd.concat(
                [
                    result_df,
                    method_test(
                        nl_config_path,
                        "NeuralLagrangian",
                        test_run_mode,
                        condition_dict,
                    ),
                ]
            )
            result_df = pd.concat(
                [
                    result_df,
                    method_test(na_config_path, "NA", test_run_mode, condition_dict),
                ]
            )
            result_df.to_csv(
                f"{task}_{model_name}_hard-constraint_results.csv", index=False
            )
            if os.path.exists("lightning_logs"):
                shutil.rmtree("lightning_logs")

    """
  Soft Constraint (Equation Constraint)
  """
    result_df = pd.DataFrame([])
    for target_id in eq_id_list:
        for eq_ID in eq_IDs:
            condition_dict = {
                "num_samples": num_samples,
                "condition_type": "Constrain",
                "eq_constraint": eq_ID,
                "trial_id": 0,
                "target_id": target_id,
            }
            target_id = int(target_id)
            result_df = pd.concat(
                [
                    result_df,
                    method_test(
                        nl_config_path,
                        "NeuralLagrangian",
                        test_run_mode,
                        condition_dict,
                    ),
                ]
            )
            result_df = pd.concat(
                [
                    result_df,
                    method_test(na_config_path, "NA", test_run_mode, condition_dict),
                ]
            )
            result_df.to_csv(
                f"{task}_{model_name}_soft-constraint_results.csv", index=False
            )
            if os.path.exists("lightning_logs"):
                shutil.rmtree("lightning_logs")


if __name__ == "__main__":
    task = "Stack" # "ADM" or "Stack" or "Shell"
    model_name = "base" # "base" or "small" or "medium"
    
    baseline_test(
        task=task,
        model_name=model_name,
        test_run_mode=False,
    )

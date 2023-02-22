import sys
sys.path.append('../')

from omegaconf import OmegaConf

from utils.config import cfg_add_dirs
from forward_train.Neural_Adjoint.chen import train_forward_model
from backward_optimization.Neural_Adjoint.chen import backward_optimization
from eval.evaluation import evaluation


config_path =  "../configs/NA_chen_config.yaml"
cfg = OmegaConf.load(config_path)

test_run_mode =True
T = 200

if test_run_mode:
    T=2

resimulation_results = []
cfg_add_dirs(cfg)
train_forward_model(cfg, test=test_run_mode)
for _trial in range(T):
    backward_optimization(cfg, test=test_run_mode)
    resimulation_results.append(evaluation(cfg))

df = pd.DataFrame(resimulation_results,columns=['re-simulation error'])
print("")
print("")
print("")
print(f'Re-Simulation Error Results T={T}')
print(df.describe())
test_result_dir = os.path.join(cfg.output_dirs.result_dir, cfg.general.exp_name)
df.to_csv(os.path.join(test_result_dir, "results.csv"),index=False)

import subprocess
import os
import hydra
from omegaconf import DictConfig




@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # List of preprocessor variants you want to try
    val_sizes = ".1, .2"
    balances_factors = ".2"
    scaler_types = "standard"


    # Tuning different models
    models = "nn,tree"



    cmds = [
        "python",
        "-m",
        "scripts.tuning",
        "-m", # --multirun
        f"model_type={models}",
        f"preprocessor.val_size={val_sizes}",
        f"preprocessor.balance_factor={balances_factors}",
        f"preprocessor.scaler_type={scaler_types}"
    ]
    
    # Pass the list
    subprocess.run(cmds, check=True)



    # call evaluate script
    cmds_eval = [
        "python",
        "-m",
        "scripts.evaluation",
    ]
    
    # Pass the list
    subprocess.run(cmds_eval, check=True)




if __name__ == "__main__":
    main()



"""
cmds = [
    "python",
    "-m",
    "scripts.tuning",
    "-m",
    "hydra/launcher=joblib",
    "hydra.launcher.n_jobs=4",
    f"model_type={model_types}",
    f"preprocessor.val_size={val_sizes}",
    f"preprocessor.balance_factor={balances_factors}",
    f"preprocessor.scaler_type={scaler_types}"
]
subprocess.run(cmds, check=True)
"""




"""
python -m scripts.tuning --multirun \
    model_type=nn,tree \
    preprocessor.val_size=0.1,0.2,0.4 \
    preprocessor.scaler_type=standard,minma
"""





#python -m scripts.tuning -m model_type=nn,tree preprocessor.val_size=.4,.6
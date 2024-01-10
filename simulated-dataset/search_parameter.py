import optuna
from uci_tabular import train, seed_everything

import argparse

# DATASET = "airfoil"
# DATASET = "concrete"
# DATASET = "housing"
# DATASET = "abalone"
DATASET = "boston"

def objective(trial):

    args = argparse.Namespace()
    # hyper-parameter search space
    # args.momentum = trial.suggest_categorical("momentum", [0.80, 0.90, 0.95, 0.99, 0.999])
    args.momentum = trial.suggest_categorical("momentum", [0.95, 0.99, 0.999])

    args.temperature = trial.suggest_categorical("temperature", [0.07, 0.1, 0.5, 2.0])

    args.loas_w1 = trial.suggest_categorical("loas_w1", [0.01, 0.1, 0.5, 1.0, 2.0, 10.0])
    args.loas_w2 = trial.suggest_categorical("loas_w2", [0.01, 0.1, 0.5, 1.0, 2.0, 10.0])
    args.loas_w3 = trial.suggest_categorical("loas_w3", [0.01, 0.1, 0.5, 1.0, 2.0, 10.0])

    args.n = trial.suggest_categorical("n", [100, 1000, 2000, 5000, 10000])
    args.epochs = trial.suggest_categorical("epochs", [200, 500, 1000, 2000])
    args.batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 256])

    args.lr = trial.suggest_categorical("lr", [1e-3, 2e-3, 5e-3, 1e-4])
    args.lr_seq2seq = trial.suggest_categorical("lr_seq2seq", [1e-4, 2e-3, 5e-3, 1e-5])

    args.dataset = DATASET
    args.feature_dim = 10
    
    args.print_freq = 200
    args.baseline = False
    args.seq2seq = "mlp"
    args.path_to_save_figures = "optuna"
    args.device = "cuda:0"
    args.visualize = False
    args.seed = 0

    test_mae = []
    for i in range(1, 6):
        args.fold = i
        test_mae.append(train(args))
    
    return sum(test_mae) / len(test_mae)

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize", study_name=f"{DATASET}-optuna-3", storage=f"sqlite:///optuna/uci.db", load_if_exists=True)
    study.optimize(objective, n_trials=200)

    print(study.best_trial)    
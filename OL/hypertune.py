# use optuna to tune the hyperparameters

from main import main
import torch, random, numpy as np
import optuna

def seed_everything(seed=2024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def objective(trial):

    w1 = trial.suggest_categorical('w1', [0, 1e-4, 5e-3, 1e-3, 5e-2, 1e-2])
    w2 = trial.suggest_categorical('w2', [0, 1e-4, 5e-3, 1e-3, 5e-2, 1e-2])
    w3 = trial.suggest_categorical('w3', [0, 1e-4, 5e-3, 1e-3, 5e-2, 1e-2])
    temp = trial.suggest_categorical('temp', [0.07, 0.1, 0.3, 0.5, 1.0, 2.0])
    
    seed_everything(seed=2024)
    kwargs = {'w1': w1, 'w2': w2, 'w3': w3, 'temp': temp}
    args  = {'linear': True, 'dir': True, 'dfr': True, 'oe': False, 'print_performance': False, 'times':1}
    # make args to namespaced object
    args = type('args', (object,), args)
    test_mse, test_std = main(args, kwargs=kwargs)
    return test_mse

if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)

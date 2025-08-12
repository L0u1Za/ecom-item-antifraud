def objective(trial):
    # Define the hyperparameters to optimize
    param = {
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),
        'batch_size': trial.suggest_int('batch_size', 16, 128),
        'num_layers': trial.suggest_int('num_layers', 1, 5),
        'hidden_size': trial.suggest_int('hidden_size', 32, 256),
        'dropout': trial.suggest_uniform('dropout', 0.0, 0.5),
    }

    # Create the model using the suggested hyperparameters
    model = create_model(param)

    # Train the model and get the validation score
    score = train_and_validate(model, param)

    return score

def run_optuna_study():
    import optuna

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    print("Best hyperparameters: ", study.best_params)
    print("Best score: ", study.best_value)

if __name__ == "__main__":
    run_optuna_study()
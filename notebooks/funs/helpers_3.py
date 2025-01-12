"""
This is a boilerplate pipeline 'modeling'
generated using Kedro 0.19.10
"""

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold
import optuna
import pandas as pd
import numpy as np

def _xgb_param_space(trial):
    trial.set_user_attr('objective', 'reg:squarederror')
    return {
        'objective': trial.user_attrs['objective'],
        'max_depth': trial.suggest_int("max_depth", 2, 10),
        'min_child_weight': trial.suggest_int("min_child_weight", 0, 10),
        'gamma': trial.suggest_int("gamma", 0, 10),
        'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.5)
    }

def _ada_param_space(trial):
    return {
        'n_estimators': trial.suggest_int("n_estimators", 50, 500),
        'learning_rate': trial.suggest_float("learning_rate", 0.01, 1.0, log=True),
        'loss': trial.suggest_categorical("loss", ["linear", "square", "exponential"]),
    }

def _lstm_param_space(trial):
    return {
        'units': trial.suggest_int('units', 10, 100),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
        'epochs': trial.suggest_int('epochs', 10, 50),
        'time_steps': trial.suggest_int('epochs', 5, 20)
    }

def _create_lstm_model(params):

    units = params.get('units', 50)
    learning_rate = params.get('learning_rate', 0.001)
    input_shape = params['input_shape']

    model = Sequential()
    model.add(LSTM(units, activation='relu', input_shape=input_shape))
    model.add(Dense(1))  # Regression output layer
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model

def _train_evaluate_lstm(x_train_fold, y_train_fold, x_val_fold, y_val_fold, params):
    """
    Trains an LSTM model on a single fold and returns the MSE score.
    """
    time_steps = params['time_steps']

    num_samples = x_train_fold.shape[0] // time_steps
    num_samples_2 = x_val_fold.shape[0] // time_steps
    params['input_shape']=(time_steps,x_train_fold.shape[1])

    x_train_fold = x_train_fold.reshape((num_samples, time_steps, x_train_fold.shape[1]))
    x_val_fold = x_val_fold.reshape((num_samples_2, time_steps, x_val_fold.shape[1]))

    y_train_fold = y_train_fold[:num_samples]
    y_val_fold = y_val_fold[:num_samples_2] 
    # Create and train the LSTM model
    lstm_model = _create_lstm_model(params)
    lstm_model.fit(x_train_fold, y_train_fold, epochs=params['epochs'], batch_size=32, verbose=0)

    # Predict and evaluate
    y_pred = lstm_model.predict(x_val_fold)
    return mean_squared_error(y_val_fold, y_pred)

def _lstm_cross_val(x_train, y_train, params=None, n_splits=4):
    default_params = {
        'units': 50,
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 5,
        'time_steps': 1
    }
    if params is None:
        params = default_params
    else:
        params = {**default_params, **params}

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mse_scores = []

    for train_idx, val_idx in kf.split(x_train):
        x_train_fold, x_val_fold = x_train[train_idx], x_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        mse = _train_evaluate_lstm(x_train_fold, y_train_fold, x_val_fold, y_val_fold, params)
        mse_scores.append(mse)

    return -np.mean(mse_scores)


def get_initial_score(train, params):
    """
    Evaluates initial model performance using cross-validation.
    Uses K-Fold CV for standard models and custom CV for LSTM.
    """
    target_column = params['target']
    scoring_method = params['scoring']
    n_splits = 4  # Number of cross-validation folds
    result = {}

    models = {
        'xgb': XGBRegressor(),
        'lr': LinearRegression(),
        'ada': AdaBoostRegressor(),
        'lstm': 'lstm'
    }

    x_train = train.drop(columns=[target_column]).values
    y_train = train[target_column].values

    for name, model in models.items():
        if name == 'lstm':
            result[name] = _lstm_cross_val(x_train, y_train, n_splits=n_splits)
        else:
            res = cross_val_score(model, x_train, y_train, scoring=scoring_method, cv=n_splits)
            result[name] = res.mean()
    return result

def _get_optuna_params(train, model_class, param_space, params):
    """
    Function to optimize hyperparameters for any regression model using Optuna.
    """
    target_column = params['target']
    scoring_method = params['scoring']
    n_trials = params['n_trials']

    def objective(trial):
        model_params = param_space(trial)
        x_train = train.drop(columns=[target_column])
        y_train = train[target_column]
        model = model_class(**model_params)
        score = cross_val_score(model, x_train, y_train,
                                scoring=scoring_method,
                                cv=5).mean()
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    best_params.update(study.best_trial.user_attrs)
    return best_params

def optimize_lr_hyperparams(train, params):
    return {'fit_intercept': True}

def optimize_ada_hyperparams(train, params):
    return _get_optuna_params(train, AdaBoostRegressor, _ada_param_space, params)

def optimize_xgb_hyperparams(train, params):
    return _get_optuna_params(train, XGBRegressor, _xgb_param_space, params)

def optimize_lstm_hyperparams(train, params):
    target_column = params['target']
    n_trials = params['n_trials']

    x_train = train.drop(columns=[target_column]).values
    y_train = train[target_column].values

    def objective(trial):
        lstm_params = _lstm_param_space(trial)
        mse_score = _lstm_cross_val(x_train, y_train, lstm_params, n_splits=4)
        return -mse_score  # higher -MSE is better

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    return study.best_params

def _train_model(train, model, params, model_type='regression'):
    target_column = params['target']
    x_train = train.drop(columns=[target_column])
    y_train = train[target_column]

    if model_type == 'lstm':
        x_train = x_train.values.reshape((x_train.shape[0], 1, x_train.shape[1]))  
        model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)
    else:
        model.fit(x_train, y_train)
    return model

def _evaluate_model(model, train, test, params, model_type='regression'):
    target_column = params['target']

    x_train = train.drop(columns=[target_column])
    y_train = train[target_column]
    x_test = test.drop(columns=[target_column])
    y_test = test[target_column]

    if model_type == 'lstm':
        x_train = x_train.values.reshape((x_train.shape[0], 1, x_train.shape[1]))  
        x_test = x_test.values.reshape((x_test.shape[0], 1, x_test.shape[1]))  
        y_pred_train = model.predict(x_train)
        y_pred_test = model.predict(x_test)
    else:
        y_pred_train = model.predict(x_train)
        y_pred_test = model.predict(x_test)

    res_dict = {
        'train_mse': mean_squared_error(y_train, y_pred_train),
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'train_r2': r2_score(y_train, y_pred_train),
        'train_mape': np.mean(np.where(y_train != 0, np.abs(y_train - y_pred_train) / np.abs(y_train), 0)) * 100,
        'test_mse': mean_squared_error(y_test, y_pred_test),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
        'test_r2': r2_score(y_test, y_pred_test),
        'test_mape': np.mean(np.where(y_test != 0, np.abs(y_test - y_pred_test) / np.abs(y_test), 0)) * 100,
    }

    y_pred_train_df = pd.DataFrame({'true_label': y_train, 'predicted_label': y_pred_train.flatten()})
    y_pred_test_df = pd.DataFrame({'true_label': y_test, 'predicted_label': y_pred_test.flatten()})

    return res_dict, y_pred_train_df, y_pred_test_df

def eval_best_models(train, test, LR_optuna_params, ADA_optuna_params, XGB_optuna_params, LSTM_optuna_params, params):
    models = {
        'xgb': XGBRegressor(**XGB_optuna_params),
        'lr': LinearRegression(**LR_optuna_params),
        'ada': AdaBoostRegressor(**ADA_optuna_params),
        'lstm': _create_lstm_model(input_shape=(1, train.shape[1] - 1), **LSTM_optuna_params)
    }

    results = {}
    for name, model in models.items():
        model_type = 'lstm' if name == 'lstm' else 'regression'
        trained_model = _train_model(train, model, params, model_type=model_type)
        results.update({f'{name}_{metric}': value for metric, value in _evaluate_model(trained_model, train, test, params, model_type=model_type)[0].items()})
    return results
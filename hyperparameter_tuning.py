"""
Hyperparameter Tuning for Forex Prediction Models
- Allows tuning of CNN-LSTM, TFT, and XGBoost models
- Supports both single currency pair models and bagging models
"""

import os
import json
import argparse
import logging
from datetime import datetime
import random
import numpy as np
import pandas as pd

import optuna
from forex_prediction import ForexPrediction

import xgboost as xgb

# Set up logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"hyperparameter_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define hyperparameter search spaces
def get_hyperparameter_space(model_type):
    """Define hyperparameter search space for each model type."""
    if model_type == 'cnn_lstm':
        return {
            'cnn_filters': (32, 256),  # Number of filters in CNN layers (expanded range)
            'cnn_kernel_size': (2, 5),  # Size of CNN kernels
            'lstm_units': (50, 300),    # Number of LSTM units (expanded range)
            'dropout_rate': (0.1, 0.7),  # Dropout rate (expanded range)
            'learning_rate': (1e-5, 5e-2)  # Learning rate (expanded range)
        }
    elif model_type == 'tft':
        return {
            'hidden_units': (32, 256),  # Number of hidden units (expanded range)
            'num_heads': (1, 12),        # Number of attention heads (expanded range)
            'dropout_rate': (0.1, 0.7),  # Dropout rate (expanded range)
            'learning_rate': (1e-5, 5e-2)  # Learning rate (expanded range)
        }
    elif model_type == 'xgboost':
        return {
            'max_depth': (3, 15),        # Maximum tree depth (expanded range)
            'learning_rate': (0.001, 0.5),  # Learning rate (expanded range)
            'n_estimators': (50, 500),    # Number of estimators (expanded range)
            'subsample': (0.5, 1.0),      # Subsample ratio
            'colsample_bytree': (0.5, 1.0),  # Column sample ratio
            'gamma': (0, 10),             # Minimum loss reduction for partition (expanded range)
            'reg_alpha': (0, 10),         # L1 regularization (expanded range)
            'reg_lambda': (0.1, 10)       # L2 regularization (expanded range)
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def objective_cnn_lstm(trial, forex_pred, pair):
    """Objective function for CNN-LSTM hyperparameter tuning."""
    # Get hyperparameters with expanded ranges
    space = get_hyperparameter_space('cnn_lstm')
    hyperparams = {
        'cnn_filters': trial.suggest_int('cnn_filters', space['cnn_filters'][0], space['cnn_filters'][1]),
        'cnn_kernel_size': trial.suggest_int('cnn_kernel_size', space['cnn_kernel_size'][0], space['cnn_kernel_size'][1]),
        'lstm_units': trial.suggest_int('lstm_units', space['lstm_units'][0], space['lstm_units'][1]),
        'dropout_rate': trial.suggest_float('dropout_rate', space['dropout_rate'][0], space['dropout_rate'][1]),
        'learning_rate': trial.suggest_float('learning_rate', space['learning_rate'][0], space['learning_rate'][1], log=True)
    }
    
    logger.info(f"Trial {trial.number}: Testing CNN-LSTM with hyperparams: {hyperparams}")
    
    # Get data
    if pair == 'Bagging':
        # Create a more diverse dataset for bagging
        currency_pairs = forex_pred.config['currency_pairs']
        # Select a random subset of pairs for each trial
        selected_pairs = random.sample(currency_pairs, max(1, len(currency_pairs) // 2))
        logger.info(f"Using random subset of pairs for Bagging: {selected_pairs}")
        
        # Combine data from selected pairs
        train_dfs = []
        for sel_pair in selected_pairs:
            pair_df = forex_pred.selected_features[sel_pair]['train'].copy()
            pair_df['CurrencyPair'] = sel_pair
            train_dfs.append(pair_df)
        
        # Create training data for this trial
        train_data = pd.concat(train_dfs, axis=0)
        train_data.sort_index(inplace=True)
    else:
        train_data = forex_pred.selected_features[pair]['train']
    
    # Prepare data for LSTM
    X_train, y_train, _, _ = forex_pred.prepare_model_data(train_data, is_lstm=True)
    
    # Robustly handle NaN and Inf values
    if np.isnan(X_train).any() or np.isnan(y_train).any() or np.isinf(X_train).any() or np.isinf(y_train).any():
        logger.warning(f"NaN or Inf values detected in {pair} CNN-LSTM training data. Fixing...")
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        y_train = np.nan_to_num(y_train, nan=0.0, posinf=1.0, neginf=0.0)
    
    # For Bagging, use randomized train-validation split for each trial
    if pair == 'Bagging':
        # Shuffle data
        indices = np.arange(len(X_train))
        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]
    
    # Split training and validation sets
    val_size = int(len(X_train) * forex_pred.config['validation_size'])
    X_val, y_val = X_train[-val_size:], y_train[-val_size:]
    X_train, y_train = X_train[:-val_size], y_train[:-val_size]
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    # Build model with trial hyperparameters
    model = forex_pred.build_cnn_lstm_model(input_shape, hyperparams)
    
    # Define callbacks with more patience for complex models
    from tensorflow.keras.callbacks import EarlyStopping
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, min_delta=0.001)
    ]
    
    # Train the model with verbose output
    try:
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,  # Increase epochs for better convergence
            batch_size=forex_pred.config['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        
        # Return the validation accuracy
        val_accuracy = history.history['val_accuracy'][-1]
        logger.info(f"Trial {trial.number}: val_accuracy = {val_accuracy:.4f}")
        return val_accuracy
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        # Return very low score to reject this trial
        return 0.0

def objective_tft(trial, forex_pred, pair):
    """Objective function for TFT hyperparameter tuning."""
    # Get hyperparameters with expanded ranges
    space = get_hyperparameter_space('tft')
    hyperparams = {
        'hidden_units': trial.suggest_int('hidden_units', space['hidden_units'][0], space['hidden_units'][1], step=32),
        'num_heads': trial.suggest_int('num_heads', space['num_heads'][0], space['num_heads'][1]),
        'dropout_rate': trial.suggest_float('dropout_rate', space['dropout_rate'][0], space['dropout_rate'][1]),
        'learning_rate': trial.suggest_float('learning_rate', space['learning_rate'][0], space['learning_rate'][1], log=True)
    }
    
    logger.info(f"Trial {trial.number}: Testing TFT with hyperparams: {hyperparams}")
    
    # Get data
    if pair == 'Bagging':
        # Create a more diverse dataset for bagging
        currency_pairs = forex_pred.config['currency_pairs']
        # Select a random subset of pairs for each trial
        selected_pairs = random.sample(currency_pairs, max(1, len(currency_pairs) // 2))
        logger.info(f"Using random subset of pairs for Bagging: {selected_pairs}")
        
        # Combine data from selected pairs
        train_dfs = []
        for sel_pair in selected_pairs:
            pair_df = forex_pred.selected_features[sel_pair]['train'].copy()
            pair_df['CurrencyPair'] = sel_pair
            train_dfs.append(pair_df)
        
        # Create training data for this trial
        train_data = pd.concat(train_dfs, axis=0)
        train_data.sort_index(inplace=True)
    else:
        train_data = forex_pred.selected_features[pair]['train']
    
    # Prepare data for TFT
    X_train, y_train, _, _ = forex_pred.prepare_model_data(train_data, is_lstm=True)
    
    # Robustly handle NaN and Inf values
    if np.isnan(X_train).any() or np.isnan(y_train).any() or np.isinf(X_train).any() or np.isinf(y_train).any():
        logger.warning(f"NaN or Inf values detected in {pair} TFT training data. Fixing...")
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        y_train = np.nan_to_num(y_train, nan=0.0, posinf=1.0, neginf=0.0)
    
    # For Bagging, use randomized train-validation split for each trial
    if pair == 'Bagging':
        # Shuffle data
        indices = np.arange(len(X_train))
        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]
    
    # Split training and validation sets
    val_size = int(len(X_train) * forex_pred.config['validation_size'])
    X_val, y_val = X_train[-val_size:], y_train[-val_size:]
    X_train, y_train = X_train[:-val_size], y_train[:-val_size]
    
    input_dim = X_train.shape[2]
    
    # Build model with trial hyperparameters
    model = forex_pred.build_tft_model(input_dim, hyperparams)
    
    # Define callbacks with more patience for complex models
    from tensorflow.keras.callbacks import EarlyStopping
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, min_delta=0.001)
    ]
    
    # Train the model with verbose output
    try:
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,  # Increase epochs for better convergence
            batch_size=forex_pred.config['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        
        # Return the validation accuracy
        val_accuracy = history.history['val_accuracy'][-1]
        logger.info(f"Trial {trial.number}: val_accuracy = {val_accuracy:.4f}")
        return val_accuracy
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        # Return very low score to reject this trial
        return 0.0

def objective_xgboost(trial, forex_pred, pair):
    """Objective function for XGBoost hyperparameter tuning."""
    # Get hyperparameters with expanded ranges
    space = get_hyperparameter_space('xgboost')
    hyperparams = {
        'max_depth': trial.suggest_int('max_depth', space['max_depth'][0], space['max_depth'][1]),
        'learning_rate': trial.suggest_float('learning_rate', space['learning_rate'][0], space['learning_rate'][1], log=True),
        'n_estimators': trial.suggest_int('n_estimators', space['n_estimators'][0], space['n_estimators'][1], step=50),
        'subsample': trial.suggest_float('subsample', space['subsample'][0], space['subsample'][1], step=0.1),
        'colsample_bytree': trial.suggest_float('colsample_bytree', space['colsample_bytree'][0], space['colsample_bytree'][1], step=0.1),
        'gamma': trial.suggest_float('gamma', space['gamma'][0], space['gamma'][1]),
        'reg_alpha': trial.suggest_float('reg_alpha', space['reg_alpha'][0], space['reg_alpha'][1]),
        'reg_lambda': trial.suggest_float('reg_lambda', space['reg_lambda'][0], space['reg_lambda'][1]),
        'objective': 'binary:logistic'
    }
    
    logger.info(f"Trial {trial.number}: Testing XGBoost with hyperparams: {hyperparams}")
    
    # Get data
    if pair == 'Bagging':
        # Create a more diverse dataset for bagging
        currency_pairs = forex_pred.config['currency_pairs']
        # Select a random subset of pairs for each trial
        selected_pairs = random.sample(currency_pairs, max(1, len(currency_pairs) // 2))
        logger.info(f"Using random subset of pairs for Bagging: {selected_pairs}")
        
        # Combine data from selected pairs
        train_dfs = []
        for sel_pair in selected_pairs:
            pair_df = forex_pred.selected_features[sel_pair]['train'].copy()
            pair_df['CurrencyPair'] = sel_pair
            train_dfs.append(pair_df)
        
        # Create training data for this trial
        train_data = pd.concat(train_dfs, axis=0)
        train_data.sort_index(inplace=True)
    else:
        train_data = forex_pred.selected_features[pair]['train']
    
    # Prepare data for XGBoost
    X_train, y_train, _, _ = forex_pred.prepare_model_data(train_data, is_lstm=False)
    
    # Robustly handle NaN and Inf values
    if np.isnan(X_train).any() or np.isnan(y_train).any() or np.isinf(X_train).any() or np.isinf(y_train).any():
        logger.warning(f"NaN or Inf values detected in {pair} XGBoost training data. Fixing...")
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        y_train = np.nan_to_num(y_train, nan=0.0, posinf=1.0, neginf=0.0)
    
    # For Bagging, use randomized train-validation split for each trial
    if pair == 'Bagging':
        # Shuffle data
        indices = np.arange(len(X_train))
        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]
    
    # Split training and validation sets
    val_size = int(len(X_train) * forex_pred.config['validation_size'])
    X_val, y_val = X_train[-val_size:], y_train[-val_size:]
    X_train, y_train = X_train[:-val_size], y_train[:-val_size]
    
    # Build model with trial hyperparameters
    model = forex_pred.build_xgboost_model(hyperparams)
    
    # Train the model
    try:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            # early_stopping_rounds=20,
            verbose=True
        )
        
        # Return the validation accuracy
        val_accuracy = model.score(X_val, y_val)
        logger.info(f"Trial {trial.number}: val_accuracy = {val_accuracy:.4f}")
        return val_accuracy
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        # Return very low score to reject this trial
        return 0.0

def tune_hyperparameters(model_type, pair, n_trials=25):
    """Tune hyperparameters for a specific model and currency pair."""
    logger.info(f"Starting hyperparameter tuning for {model_type} on {pair}")
    
    # Initialize the Forex Prediction system (data loading and preprocessing)
    forex_pred = ForexPrediction()
    
    # Load and preprocess data
    forex_pred.load_data()
    forex_pred.preprocess_data()
    forex_pred.calculate_technical_indicators()
    forex_pred.select_features()
    
    # Create output directory for hyperparameter tuning results
    output_dir = "output"
    tuning_dir = os.path.join(output_dir, "hyperparameter_tuning")
    os.makedirs(tuning_dir, exist_ok=True)
    
    # Select the appropriate objective function
    if model_type == 'cnn_lstm':
        objective = lambda trial: objective_cnn_lstm(trial, forex_pred, pair)
    elif model_type == 'tft':
        objective = lambda trial: objective_tft(trial, forex_pred, pair)
    elif model_type == 'xgboost':
        objective = lambda trial: objective_xgboost(trial, forex_pred, pair)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create and run the Optuna study
    study = optuna.create_study(direction='maximize', 
                               study_name=f"{model_type}_{pair}_tuning")
    study.optimize(objective, n_trials=n_trials)
    
    # Get the best hyperparameters
    best_params = study.best_params
    best_value = study.best_value
    
    # Log the results
    logger.info(f"Best hyperparameters for {model_type} on {pair}: {best_params}")
    logger.info(f"Best validation accuracy: {best_value:.4f}")
    
    # Save the best hyperparameters to file
    output_file = os.path.join(tuning_dir, f"{pair}_{model_type}_best_hyperparams.json")
    with open(output_file, 'w') as f:
        json.dump(best_params, f, indent=2)
    
    # Plot optimization history
    try:
        import matplotlib.pyplot as plt
        from optuna.visualization import plot_optimization_history
        
        plt.figure(figsize=(10, 6))
        optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.title(f"Optimization History - {model_type} on {pair}")
        plt.tight_layout()
        plt.savefig(os.path.join(tuning_dir, f"{pair}_{model_type}_optimization_history.png"))
        plt.close()
        
        # Plot parameter importances
        plt.figure(figsize=(10, 6))
        optuna.visualization.matplotlib.plot_param_importances(study)
        plt.title(f"Parameter Importances - {model_type} on {pair}")
        plt.tight_layout()
        plt.savefig(os.path.join(tuning_dir, f"{pair}_{model_type}_param_importances.png"))
        plt.close()
    except Exception as e:
        logger.warning(f"Error creating plots: {e}")
    
    return best_params

def main():
    """Main function for hyperparameter tuning."""
    parser = argparse.ArgumentParser(description='Hyperparameter Tuning for Forex Prediction Models')
    parser.add_argument('--model', type=str, choices=['cnn_lstm', 'tft', 'xgboost', 'all'], default='all',
                        help='Model type to tune (default: all)')
    parser.add_argument('--pair', type=str, choices=['EURUSD', 'GBPUSD', 'USDJPY', 'Bagging', 'all'], default='all',
                        help='Currency pair to tune (default: all)')
    parser.add_argument('--trials', type=int, default=25, help='Number of optimization trials (default: 25)')
    
    args = parser.parse_args()
    
    # Define model types and currency pairs to tune
    if args.model == 'all':
        model_types = ['cnn_lstm', 'tft', 'xgboost']
    else:
        model_types = [args.model]
    
    if args.pair == 'all':
        pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'Bagging']
    else:
        pairs = [args.pair]
    
    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Run hyperparameter tuning for each model type and currency pair
    results = {}
    for model_type in model_types:
        results[model_type] = {}
        for pair in pairs:
            logger.info(f"Tuning {model_type} for {pair} with {args.trials} trials")
            best_params = tune_hyperparameters(model_type, pair, args.trials)
            results[model_type][pair] = best_params
    
    # Save overall results
    output_dir = "output"
    tuning_dir = os.path.join(output_dir, "hyperparameter_tuning")
    with open(os.path.join(tuning_dir, "tuning_results_summary.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("Hyperparameter tuning completed")

if __name__ == "__main__":
    main()
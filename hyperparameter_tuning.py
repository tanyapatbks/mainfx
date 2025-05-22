"""
Enhanced Hyperparameter Tuning for Master's Thesis Forex Prediction
- Advanced Bayesian optimization with multi-objective support
- Comprehensive visualization and reporting
- Integration with enhanced pipeline
- Walk-forward validation during tuning
- Advanced pruning and early stopping strategies
"""

import os
import json
import argparse
import logging
import warnings
from datetime import datetime
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing

# Advanced optimization libraries
import optuna
from optuna.samplers import TPESampler, CmaEsSampler, NSGAIISampler
from optuna.pruners import MedianPruner, HyperbandPruner, PercentilePruner
from optuna.integration import SklearnIntegration
from optuna.visualization import plot_optimization_history, plot_param_importances
from optuna.visualization import plot_parallel_coordinate, plot_slice

# Scientific computing
from scipy.optimize import differential_evolution
from skopt import gp_minimize, forest_minimize, gbrt_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt.plots import plot_convergence, plot_objective

# Import enhanced forex prediction system
from forex_prediction import EnhancedForexPrediction

warnings.filterwarnings('ignore')

# Set up enhanced logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"enhanced_hyperparameter_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedHyperparameterOptimizer:
    """
    Enhanced hyperparameter optimizer for Master's thesis research.
    
    Features:
    - Multi-objective optimization (return vs risk)
    - Advanced pruning strategies
    - Parallel optimization
    - Comprehensive reporting
    - Walk-forward validation
    """
    
    def __init__(self, config=None):
        """Initialize the enhanced optimizer."""
        self.config = config or {}
        self.forex_pred = None
        self.study_results = {}
        self.multi_objective_studies = {}
        
        # Enhanced optimization settings
        self.optimization_methods = {
            'bayesian_tpe': {
                'sampler': TPESampler,
                'pruner': MedianPruner,
                'direction': 'maximize'
            },
            'bayesian_cmaes': {
                'sampler': CmaEsSampler,
                'pruner': HyperbandPruner,
                'direction': 'maximize'
            },
            'multi_objective': {
                'sampler': NSGAIISampler,
                'pruner': PercentilePruner,
                'directions': ['maximize', 'minimize']  # Return, Risk
            }
        }
        
        logger.info("Enhanced Hyperparameter Optimizer initialized for Master's thesis")
    
    def get_enhanced_search_space(self, model_type):
        """Define enhanced search spaces for thesis research."""
        if model_type == 'enhanced_cnn_lstm':
            return {
                'cnn_filters': Integer(32, 256, name='cnn_filters'),
                'cnn_kernel_size': Integer(2, 7, name='cnn_kernel_size'),
                'lstm_units': Integer(50, 300, name='lstm_units'),
                'dropout_rate': Real(0.1, 0.7, name='dropout_rate'),
                'learning_rate': Real(1e-5, 5e-2, prior='log-uniform', name='learning_rate'),
                # Additional advanced parameters
                'cnn_layers': Integer(1, 3, name='cnn_layers'),
                'lstm_layers': Integer(1, 3, name='lstm_layers'),
                'attention_heads': Integer(2, 8, name='attention_heads'),
                'batch_norm': Categorical([True, False], name='batch_norm'),
                'l1_reg': Real(1e-6, 1e-2, prior='log-uniform', name='l1_reg'),
                'l2_reg': Real(1e-6, 1e-2, prior='log-uniform', name='l2_reg')
            }
        elif model_type == 'advanced_tft':
            return {
                'hidden_units': Integer(32, 256, name='hidden_units'),
                'num_heads': Integer(2, 16, name='num_heads'),
                'dropout_rate': Real(0.1, 0.6, name='dropout_rate'),
                'learning_rate': Real(1e-5, 5e-2, prior='log-uniform', name='learning_rate'),
                # Additional TFT parameters
                'num_encoder_layers': Integer(1, 4, name='num_encoder_layers'),
                'num_decoder_layers': Integer(1, 4, name='num_decoder_layers'),
                'feed_forward_dim': Integer(64, 512, name='feed_forward_dim'),
                'attention_dropout': Real(0.0, 0.3, name='attention_dropout'),
                'layer_norm': Categorical([True, False], name='layer_norm'),
                'residual_connections': Categorical([True, False], name='residual_connections')
            }
        elif model_type == 'enhanced_xgboost':
            return {
                'max_depth': Integer(3, 15, name='max_depth'),
                'learning_rate': Real(0.001, 0.5, prior='log-uniform', name='learning_rate'),
                'n_estimators': Integer(50, 1000, name='n_estimators'),
                'subsample': Real(0.5, 1.0, name='subsample'),
                'colsample_bytree': Real(0.5, 1.0, name='colsample_bytree'),
                'colsample_bylevel': Real(0.5, 1.0, name='colsample_bylevel'),
                'colsample_bynode': Real(0.5, 1.0, name='colsample_bynode'),
                'gamma': Real(0, 10, name='gamma'),
                'reg_alpha': Real(0, 10, name='reg_alpha'),
                'reg_lambda': Real(0.1, 10, name='reg_lambda'),
                'min_child_weight': Integer(1, 20, name='min_child_weight'),
                'max_delta_step': Integer(0, 10, name='max_delta_step'),
                # Advanced XGBoost parameters
                'grow_policy': Categorical(['depthwise', 'lossguide'], name='grow_policy'),
                'max_leaves': Integer(0, 100, name='max_leaves'),
                'scale_pos_weight': Real(0.5, 2.0, name='scale_pos_weight')
            }
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def create_optuna_objective(self, model_type, pair, validation_method='standard'):
        """Create enhanced objective function for Optuna optimization."""
        
        def enhanced_objective(trial):
            try:
                # Sample hyperparameters
                params = {}
                space = self.get_enhanced_search_space(model_type)
                
                for param_name, param_space in space.items():
                    if hasattr(param_space, 'low') and hasattr(param_space, 'high'):
                        if isinstance(param_space.low, int):
                            params[param_name] = trial.suggest_int(param_name, param_space.low, param_space.high)
                        else:
                            if hasattr(param_space, 'prior') and param_space.prior == 'log-uniform':
                                params[param_name] = trial.suggest_float(param_name, param_space.low, param_space.high, log=True)
                            else:
                                params[param_name] = trial.suggest_float(param_name, param_space.low, param_space.high)
                    elif hasattr(param_space, 'categories'):
                        params[param_name] = trial.suggest_categorical(param_name, param_space.categories)
                
                # Enhanced validation
                if validation_method == 'walkforward':
                    score = self._evaluate_with_walkforward(model_type, pair, params, trial)
                else:
                    score = self._evaluate_standard(model_type, pair, params, trial)
                
                # Report intermediate value for pruning
                trial.report(score, step=1)
                
                # Check if trial should be pruned
                if trial.should_prune():
                    raise optuna.TrialPruned()
                
                return score
                
            except optuna.TrialPruned:
                raise
            except Exception as e:
                logger.error(f"Error in trial {trial.number}: {e}")
                return 0.0
        
        return enhanced_objective
    
    def create_multi_objective_function(self, model_type, pair):
        """Create multi-objective function optimizing return and risk."""
        
        def multi_objective(trial):
            try:
                # Sample hyperparameters
                params = {}
                space = self.get_enhanced_search_space(model_type)
                
                for param_name, param_space in space.items():
                    if hasattr(param_space, 'low') and hasattr(param_space, 'high'):
                        if isinstance(param_space.low, int):
                            params[param_name] = trial.suggest_int(param_name, param_space.low, param_space.high)
                        else:
                            if hasattr(param_space, 'prior') and param_space.prior == 'log-uniform':
                                params[param_name] = trial.suggest_float(param_name, param_space.low, param_space.high, log=True)
                            else:
                                params[param_name] = trial.suggest_float(param_name, param_space.low, param_space.high)
                    elif hasattr(param_space, 'categories'):
                        params[param_name] = trial.suggest_categorical(param_name, param_space.categories)
                
                # Evaluate model and get both return and risk metrics
                annual_return, max_drawdown = self._evaluate_return_and_risk(model_type, pair, params, trial)
                
                # Return objectives: maximize return, minimize risk (max drawdown)
                return annual_return, max_drawdown
                
            except Exception as e:
                logger.error(f"Error in multi-objective trial {trial.number}: {e}")
                return 0.0, 100.0  # Bad return, high risk
        
        return multi_objective
    
    def _evaluate_standard(self, model_type, pair, params, trial):
        """Standard evaluation using validation set."""
        try:
            # Get training and validation data
            if pair == 'Bagging':
                train_data = self.forex_pred.bagging_data['train']
                val_data = self.forex_pred.bagging_data['validation']
            else:
                train_data = self.forex_pred.selected_features[pair]['train']
                val_data = self.forex_pred.selected_features[pair]['validation']
            
            # Temporarily update hyperparameters
            original_params = self.forex_pred.hyperparameters.get(model_type, {}).get(pair, {})
            self.forex_pred.hyperparameters.setdefault(model_type, {})[pair] = params
            
            # Prepare data
            is_lstm = model_type in ['enhanced_cnn_lstm', 'advanced_tft']
            X_train, y_train, scaler, _ = self.forex_pred.prepare_model_data(train_data, is_lstm=is_lstm)
            X_val, y_val, _, _ = self.forex_pred.prepare_model_data(val_data, is_lstm=is_lstm)
            
            # Handle NaN values
            if np.isnan(X_train).any() or np.isnan(y_train).any():
                X_train = np.nan_to_num(X_train)
                y_train = np.nan_to_num(y_train)
            if np.isnan(X_val).any() or np.isnan(y_val).any():
                X_val = np.nan_to_num(X_val)
                y_val = np.nan_to_num(y_val)
            
            # Build and train model
            if model_type == 'enhanced_cnn_lstm':
                input_shape = (X_train.shape[1], X_train.shape[2])
                model = self.forex_pred.build_enhanced_cnn_lstm_model(input_shape, pair)
                
                from tensorflow.keras.callbacks import EarlyStopping
                callbacks = [EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)]
                
                history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                                  epochs=75, batch_size=32, callbacks=callbacks, verbose=0)
                score = max(history.history['val_accuracy'])
                
            elif model_type == 'advanced_tft':
                input_dim = X_train.shape[2]
                model = self.forex_pred.build_advanced_tft_model(input_dim, pair)
                
                from tensorflow.keras.callbacks import EarlyStopping
                callbacks = [EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)]
                
                history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                                  epochs=75, batch_size=32, callbacks=callbacks, verbose=0)
                score = max(history.history['val_accuracy'])
                
            elif model_type == 'enhanced_xgboost':
                model = self.forex_pred.build_enhanced_xgboost_model(pair)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
                score = model.score(X_val, y_val)
            
            # Restore original parameters
            self.forex_pred.hyperparameters[model_type][pair] = original_params
            
            return score
            
        except Exception as e:
            logger.error(f"Error in standard evaluation: {e}")
            return 0.0
    
    def _evaluate_with_walkforward(self, model_type, pair, params, trial):
        """Enhanced evaluation using walk-forward validation."""
        try:
            # Implement mini walk-forward for hyperparameter tuning
            if pair == 'Bagging':
                full_data = self.forex_pred.bagging_data['train']
            else:
                full_data = pd.concat([
                    self.forex_pred.selected_features[pair]['train'],
                    self.forex_pred.selected_features[pair]['validation']
                ])
            
            # Create 3 walk-forward windows for faster evaluation
            window_size = len(full_data) // 4
            scores = []
            
            for i in range(3):
                start_idx = i * window_size // 2
                train_end_idx = start_idx + window_size
                val_end_idx = min(start_idx + int(window_size * 1.2), len(full_data))
                
                if val_end_idx - train_end_idx < 50:  # Minimum validation size
                    continue
                
                window_train = full_data.iloc[start_idx:train_end_idx]
                window_val = full_data.iloc[train_end_idx:val_end_idx]
                
                # Temporarily update hyperparameters
                original_params = self.forex_pred.hyperparameters.get(model_type, {}).get(pair, {})
                self.forex_pred.hyperparameters.setdefault(model_type, {})[pair] = params
                
                # Quick model evaluation
                try:
                    is_lstm = model_type in ['enhanced_cnn_lstm', 'advanced_tft']
                    X_train, y_train, _, _ = self.forex_pred.prepare_model_data(window_train, is_lstm=is_lstm)
                    X_val, y_val, _, _ = self.forex_pred.prepare_model_data(window_val, is_lstm=is_lstm)
                    
                    if np.isnan(X_train).any() or np.isnan(y_train).any():
                        X_train = np.nan_to_num(X_train)
                        y_train = np.nan_to_num(y_train)
                    if np.isnan(X_val).any() or np.isnan(y_val).any():
                        X_val = np.nan_to_num(X_val)
                        y_val = np.nan_to_num(y_val)
                    
                    if len(X_train) < 10 or len(X_val) < 5:
                        continue
                    
                    # Quick training with reduced epochs
                    if model_type == 'enhanced_cnn_lstm':
                        input_shape = (X_train.shape[1], X_train.shape[2])
                        model = self.forex_pred.build_enhanced_cnn_lstm_model(input_shape, pair)
                        
                        history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                                          epochs=25, batch_size=32, verbose=0)
                        score = max(history.history['val_accuracy']) if history.history['val_accuracy'] else 0.5
                        
                    elif model_type == 'advanced_tft':
                        input_dim = X_train.shape[2]
                        model = self.forex_pred.build_advanced_tft_model(input_dim, pair)
                        
                        history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                                          epochs=25, batch_size=32, verbose=0)
                        score = max(history.history['val_accuracy']) if history.history['val_accuracy'] else 0.5
                        
                    elif model_type == 'enhanced_xgboost':
                        model = self.forex_pred.build_enhanced_xgboost_model(pair)
                        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
                        score = model.score(X_val, y_val)
                    
                    scores.append(score)
                    
                except Exception as e:
                    logger.warning(f"Error in walk-forward window {i}: {e}")
                    continue
                finally:
                    # Restore original parameters
                    self.forex_pred.hyperparameters[model_type][pair] = original_params
            
            # Return average score across windows
            return np.mean(scores) if scores else 0.0
            
        except Exception as e:
            logger.error(f"Error in walk-forward evaluation: {e}")
            return 0.0
    
    def _evaluate_return_and_risk(self, model_type, pair, params, trial):
        """Evaluate model for multi-objective optimization (return vs risk)."""
        try:
            # Use validation data for faster evaluation
            if pair == 'Bagging':
                train_data = self.forex_pred.bagging_data['train']
                val_data = self.forex_pred.bagging_data['validation']
            else:
                train_data = self.forex_pred.selected_features[pair]['train']
                val_data = self.forex_pred.selected_features[pair]['validation']
            
            # Temporarily update hyperparameters
            original_params = self.forex_pred.hyperparameters.get(model_type, {}).get(pair, {})
            self.forex_pred.hyperparameters.setdefault(model_type, {})[pair] = params
            
            try:
                # Train model
                is_lstm = model_type in ['enhanced_cnn_lstm', 'advanced_tft']
                X_train, y_train, _, _ = self.forex_pred.prepare_model_data(train_data, is_lstm=is_lstm)
                X_val, y_val, _, _ = self.forex_pred.prepare_model_data(val_data, is_lstm=is_lstm)
                
                # Handle NaN values
                if np.isnan(X_train).any() or np.isnan(y_train).any():
                    X_train = np.nan_to_num(X_train)
                    y_train = np.nan_to_num(y_train)
                if np.isnan(X_val).any() or np.isnan(y_val).any():
                    X_val = np.nan_to_num(X_val)
                    y_val = np.nan_to_num(y_val)
                
                # Build and train model
                if model_type == 'enhanced_cnn_lstm':
                    input_shape = (X_train.shape[1], X_train.shape[2])
                    model = self.forex_pred.build_enhanced_cnn_lstm_model(input_shape, pair)
                    
                    model.fit(X_train, y_train, validation_data=(X_val, y_val),
                            epochs=50, batch_size=32, verbose=0)
                    
                    y_pred_proba = model.predict(X_val)
                    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
                    y_pred_proba = y_pred_proba.flatten()
                    
                elif model_type == 'advanced_tft':
                    input_dim = X_train.shape[2]
                    model = self.forex_pred.build_advanced_tft_model(input_dim, pair)
                    
                    model.fit(X_train, y_train, validation_data=(X_val, y_val),
                            epochs=50, batch_size=32, verbose=0)
                    
                    y_pred_proba = model.predict(X_val)
                    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
                    y_pred_proba = y_pred_proba.flatten()
                    
                elif model_type == 'enhanced_xgboost':
                    model = self.forex_pred.build_enhanced_xgboost_model(pair)
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
                    
                    y_pred_proba = model.predict_proba(X_val)[:, 1]
                    y_pred = (y_pred_proba > 0.5).astype(int)
                
                # Quick trading simulation for return and risk
                val_data_subset = val_data.iloc[-len(y_pred):] if len(val_data) > len(y_pred) else val_data
                trading_perf = self.forex_pred.evaluate_enhanced_trading_performance(
                    pair, model_type, y_val, y_pred, y_pred_proba, val_data_subset
                )
                
                annual_return = trading_perf['annual_return']
                max_drawdown = trading_perf['max_drawdown']
                
                return annual_return, max_drawdown
                
            finally:
                # Restore original parameters
                self.forex_pred.hyperparameters[model_type][pair] = original_params
            
        except Exception as e:
            logger.error(f"Error in return/risk evaluation: {e}")
            return 0.0, 100.0
    
    def run_enhanced_optimization(self, model_type, pair, n_trials=100, 
                                 optimization_method='bayesian_tpe', 
                                 validation_method='standard'):
        """Run enhanced hyperparameter optimization."""
        logger.info(f"🔬 Starting enhanced optimization: {model_type} on {pair}")
        logger.info(f"Method: {optimization_method}, Validation: {validation_method}, Trials: {n_trials}")
        
        # Configure optimization method
        method_config = self.optimization_methods[optimization_method]
        
        if optimization_method == 'multi_objective':
            # Multi-objective optimization
            sampler = method_config['sampler'](seed=42)
            pruner = method_config['pruner'](percentile=25.0, n_startup_trials=10, n_warmup_steps=20)
            
            study = optuna.create_study(
                directions=method_config['directions'],
                sampler=sampler,
                pruner=pruner,
                study_name=f"enhanced_multi_obj_{model_type}_{pair}"
            )
            
            objective = self.create_multi_objective_function(model_type, pair)
            study.optimize(objective, n_trials=n_trials)
            
            # Store multi-objective results
            self.multi_objective_studies[f"{pair}_{model_type}"] = study
            
            # Select best solution using compromise programming
            best_trial = self._select_best_pareto_solution(study)
            best_params = best_trial.params if best_trial else {}
            best_value = best_trial.values if best_trial else [0.0, 100.0]
            
        else:
            # Single-objective optimization
            sampler = method_config['sampler'](seed=42)
            pruner = method_config['pruner'](n_startup_trials=10, n_warmup_steps=20)
            
            study = optuna.create_study(
                direction=method_config['direction'],
                sampler=sampler,
                pruner=pruner,
                study_name=f"enhanced_{optimization_method}_{model_type}_{pair}"
            )
            
            objective = self.create_optuna_objective(model_type, pair, validation_method)
            study.optimize(objective, n_trials=n_trials)
            
            best_params = study.best_params
            best_value = study.best_value
        
        # Store results
        self.study_results[f"{pair}_{model_type}"] = study
        
        # Log results
        logger.info(f"🎯 Optimization completed for {model_type} on {pair}")
        if optimization_method == 'multi_objective':
            logger.info(f"Best solution: Return={best_value[0]:.4f}, Risk={best_value[1]:.4f}")
        else:
            logger.info(f"Best score: {best_value:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        # Save results
        self._save_optimization_results(model_type, pair, study, best_params, optimization_method)
        
        # Create visualization
        self._create_optimization_visualization(model_type, pair, study, optimization_method)
        
        return best_params, study
    
    def _select_best_pareto_solution(self, study):
        """Select best solution from Pareto front using compromise programming."""
        try:
            # Get non-dominated solutions
            pareto_trials = []
            for trial in study.trials:
                if trial.state == optuna.trial.TrialState.COMPLETE and trial.values:
                    is_pareto = True
                    for other_trial in study.trials:
                        if (other_trial.state == optuna.trial.TrialState.COMPLETE and 
                            other_trial.values and other_trial != trial):
                            # Check if other_trial dominates trial
                            if (other_trial.values[0] >= trial.values[0] and  # Return
                                other_trial.values[1] <= trial.values[1] and  # Risk (lower is better)
                                (other_trial.values[0] > trial.values[0] or 
                                 other_trial.values[1] < trial.values[1])):
                                is_pareto = False
                                break
                    if is_pareto:
                        pareto_trials.append(trial)
            
            if not pareto_trials:
                return None
            
            # Select solution with best risk-adjusted return
            best_trial = max(pareto_trials, 
                           key=lambda t: t.values[0] / (t.values[1] + 1e-6))  # Return / Risk
            
            return best_trial
            
        except Exception as e:
            logger.error(f"Error selecting Pareto solution: {e}")
            return study.best_trial if hasattr(study, 'best_trial') else None
    
    def _save_optimization_results(self, model_type, pair, study, best_params, method):
        """Save comprehensive optimization results."""
        output_dir = "output/hyperparameter_tuning"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save best hyperparameters
        params_file = os.path.join(output_dir, f"{pair}_{model_type}_best_hyperparams.json")
        with open(params_file, 'w') as f:
            json.dump(best_params, f, indent=2)
        
        # Save detailed study results
        study_data = {
            'best_params': best_params,
            'best_value': study.best_value if hasattr(study, 'best_value') else None,
            'n_trials': len(study.trials),
            'optimization_method': method,
            'study_name': study.study_name,
            'trials_summary': []
        }
        
        # Add trial details (top 10)
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if method == 'multi_objective':
            # Sort by compromise solution (return/risk ratio)
            completed_trials.sort(key=lambda t: t.values[0] / (t.values[1] + 1e-6) if t.values else 0, reverse=True)
        else:
            completed_trials.sort(key=lambda t: t.value if t.value else 0, reverse=True)
        
        for trial in completed_trials[:10]:
            trial_data = {
                'number': trial.number,
                'params': trial.params,
                'value': trial.value if hasattr(trial, 'value') else None,
                'values': trial.values if hasattr(trial, 'values') else None,
                'state': trial.state.name
            }
            study_data['trials_summary'].append(trial_data)
        
        # Save study data
        study_file = os.path.join(output_dir, f"{pair}_{model_type}_optimization_study.json")
        with open(study_file, 'w') as f:
            json.dump(study_data, f, indent=2)
        
        logger.info(f"💾 Optimization results saved: {params_file}")
    
    def _create_optimization_visualization(self, model_type, pair, study, method):
        """Create comprehensive optimization visualizations."""
        try:
            output_dir = "output/hyperparameter_tuning"
            report_dir = os.path.join(output_dir, "reports")
            os.makedirs(report_dir, exist_ok=True)
            
            # Create figure with multiple subplots
            if method == 'multi_objective':
                fig, axes = plt.subplots(2, 3, figsize=(20, 12))
                fig.suptitle(f'Multi-Objective Optimization: {model_type} on {pair}', fontsize=16, fontweight='bold')
                
                # Plot 1: Pareto front
                if len(study.trials) > 0:
                    returns = []
                    risks = []
                    for trial in study.trials:
                        if trial.state == optuna.trial.TrialState.COMPLETE and trial.values:
                            returns.append(trial.values[0])
                            risks.append(trial.values[1])
                    
                    if returns and risks:
                        axes[0, 0].scatter(risks, returns, alpha=0.6, s=30)
                        axes[0, 0].set_xlabel('Risk (Max Drawdown %)')
                        axes[0, 0].set_ylabel('Return (Annual Return %)')
                        axes[0, 0].set_title('Pareto Front: Return vs Risk')
                        axes[0, 0].grid(True, alpha=0.3)
                
                # Plot 2: Parameter importance for return
                try:
                    if len(study.trials) > 10:
                        importance_return = optuna.importance.get_param_importances(study, target=lambda t: t.values[0])
                        params = list(importance_return.keys())[:8]  # Top 8 parameters
                        values = [importance_return[p] for p in params]
                        
                        axes[0, 1].barh(params, values, alpha=0.7)
                        axes[0, 1].set_title('Parameter Importance for Return')
                        axes[0, 1].set_xlabel('Importance')
                except:
                    axes[0, 1].text(0.5, 0.5, 'Insufficient data for\nparameter importance', 
                                   ha='center', va='center', transform=axes[0, 1].transAxes)
                
                # Plot 3: Parameter importance for risk
                try:
                    if len(study.trials) > 10:
                        importance_risk = optuna.importance.get_param_importances(study, target=lambda t: t.values[1])
                        params = list(importance_risk.keys())[:8]
                        values = [importance_risk[p] for p in params]
                        
                        axes[0, 2].barh(params, values, alpha=0.7, color='red')
                        axes[0, 2].set_title('Parameter Importance for Risk')
                        axes[0, 2].set_xlabel('Importance')
                except:
                    axes[0, 2].text(0.5, 0.5, 'Insufficient data for\nparameter importance', 
                                   ha='center', va='center', transform=axes[0, 2].transAxes)
                
            else:
                fig, axes = plt.subplots(2, 3, figsize=(20, 12))
                fig.suptitle(f'Single-Objective Optimization: {model_type} on {pair}', fontsize=16, fontweight='bold')
                
                # Plot 1: Optimization history
                values = [trial.value for trial in study.trials if trial.value is not None]
                if values:
                    axes[0, 0].plot(values, marker='o', alpha=0.7, linewidth=1, markersize=3)
                    axes[0, 0].set_title('Optimization History')
                    axes[0, 0].set_xlabel('Trial')
                    axes[0, 0].set_ylabel('Objective Value')
                    axes[0, 0].grid(True, alpha=0.3)
                
                # Plot 2: Best value progression
                best_values = []
                current_best = -float('inf')
                for value in values:
                    if value > current_best:
                        current_best = value
                    best_values.append(current_best)
                
                if best_values:
                    axes[0, 1].plot(best_values, marker='o', color='green', alpha=0.7, linewidth=2, markersize=3)
                    axes[0, 1].set_title('Best Value Progression')
                    axes[0, 1].set_xlabel('Trial')
                    axes[0, 1].set_ylabel('Best Objective Value')
                    axes[0, 1].grid(True, alpha=0.3)
                
                # Plot 3: Parameter importance
                try:
                    if len(study.trials) > 10:
                        importance = optuna.importance.get_param_importances(study)
                        params = list(importance.keys())[:8]  # Top 8 parameters
                        values = [importance[p] for p in params]
                        
                        axes[0, 2].barh(params, values, alpha=0.7)
                        axes[0, 2].set_title('Parameter Importance')
                        axes[0, 2].set_xlabel('Importance')
                        axes[0, 2].grid(True, alpha=0.3)
                except:
                    axes[0, 2].text(0.5, 0.5, 'Insufficient data for\nparameter importance', 
                                   ha='center', va='center', transform=axes[0, 2].transAxes)
            
            # Common plots for both methods
            
            # Plot 4: Trial distribution
            objective_values = []
            if method == 'multi_objective':
                # Use compromise metric for distribution
                for trial in study.trials:
                    if trial.state == optuna.trial.TrialState.COMPLETE and trial.values:
                        compromise = trial.values[0] / (trial.values[1] + 1e-6)
                        objective_values.append(compromise)
            else:
                objective_values = [trial.value for trial in study.trials if trial.value is not None]
            
            if objective_values:
                axes[1, 0].hist(objective_values, bins=min(20, len(objective_values)//2), 
                               alpha=0.7, edgecolor='black')
                axes[1, 0].set_title('Objective Value Distribution')
                axes[1, 0].set_xlabel('Objective Value')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 5: Convergence analysis
            if len(objective_values) > 10:
                window_size = max(5, len(objective_values)//10)
                moving_avg = pd.Series(objective_values).rolling(window=window_size).mean()
                
                axes[1, 1].plot(objective_values, alpha=0.3, label='Individual Trials', linewidth=0.5)
                axes[1, 1].plot(moving_avg, color='red', linewidth=2, 
                               label=f'Moving Average (window={window_size})')
                axes[1, 1].set_title('Convergence Analysis')
                axes[1, 1].set_xlabel('Trial')
                axes[1, 1].set_ylabel('Objective Value')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            
            # Plot 6: Best parameters radar plot
            try:
                best_params = study.best_params if hasattr(study, 'best_params') else study.best_trial.params
                if best_params:
                    param_names = list(best_params.keys())[:6]  # Limit to 6 for readability
                    param_values = []
                    
                    space = self.get_enhanced_search_space(model_type)
                    for param in param_names:
                        value = best_params[param]
                        if param in {ps.name: ps for ps in space.values()}:
                            param_space = next(ps for ps in space.values() if ps.name == param)
                            if hasattr(param_space, 'low') and hasattr(param_space, 'high'):
                                # Normalize to 0-1
                                normalized = (value - param_space.low) / (param_space.high - param_space.low)
                                param_values.append(normalized)
                            else:
                                param_values.append(0.5)  # Default for categorical
                        else:
                            param_values.append(0.5)
                    
                    if param_values:
                        angles = np.linspace(0, 2*np.pi, len(param_names), endpoint=False).tolist()
                        param_values += param_values[:1]  # Complete the circle
                        angles += angles[:1]
                        
                        ax_radar = fig.add_subplot(2, 3, 6, projection='polar')
                        ax_radar.plot(angles, param_values, 'o-', linewidth=2, label='Best Parameters')
                        ax_radar.fill(angles, param_values, alpha=0.25)
                        ax_radar.set_xticks(angles[:-1])
                        ax_radar.set_xticklabels(param_names)
                        ax_radar.set_title('Best Parameters (Normalized)')
                        ax_radar.grid(True)
            except Exception as e:
                logger.warning(f"Could not create radar plot: {e}")
            
            plt.tight_layout()
            plot_path = os.path.join(report_dir, f"{pair}_{model_type}_optimization_analysis.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"📊 Optimization visualization saved: {plot_path}")
            
        except Exception as e:
            logger.error(f"Error creating optimization visualization: {e}")
    
    def run_comprehensive_optimization(self, forex_pred, pairs=None, models=None, 
                                     n_trials=100, methods=None):
        """Run comprehensive optimization for all model-pair combinations."""
        logger.info("🚀 Starting comprehensive hyperparameter optimization for Master's thesis")
        
        self.forex_pred = forex_pred
        
        # Set defaults
        pairs = pairs or forex_pred.config['currency_pairs']
        if forex_pred.config.get('use_bagging', False):
            pairs = pairs + ['Bagging']
        
        models = models or forex_pred.config['models_to_train']
        methods = methods or ['bayesian_tpe', 'multi_objective']
        
        # Calculate total combinations
        total_combinations = len(pairs) * len(models) * len(methods)
        logger.info(f"📊 Total optimization combinations: {total_combinations}")
        
        results = {}
        current = 0
        
        for method in methods:
            results[method] = {}
            for model_type in models:
                results[method][model_type] = {}
                for pair in pairs:
                    current += 1
                    logger.info(f"🔄 Progress: {current}/{total_combinations} - {method} {model_type} on {pair}")
                    
                    try:
                        best_params, study = self.run_enhanced_optimization(
                            model_type, pair, n_trials, method, 'standard'
                        )
                        
                        results[method][model_type][pair] = {
                            'best_params': best_params,
                            'best_value': study.best_value if hasattr(study, 'best_value') else None,
                            'n_trials': len(study.trials),
                            'study_name': study.study_name
                        }
                        
                        # Update forex prediction hyperparameters with best found
                        forex_pred.hyperparameters.setdefault(model_type, {})[pair] = best_params
                        
                    except Exception as e:
                        logger.error(f"Error optimizing {method} {model_type} on {pair}: {e}")
                        continue
        
        # Create comprehensive summary report
        self._create_comprehensive_summary(results)
        
        # Save comprehensive results
        output_file = "output/hyperparameter_tuning/comprehensive_optimization_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("🎯 Comprehensive hyperparameter optimization completed!")
        logger.info(f"📄 Results saved to: {output_file}")
        
        return results
    
    def _create_comprehensive_summary(self, results):
        """Create comprehensive optimization summary report."""
        logger.info("Creating comprehensive optimization summary")
        
        # Create summary visualization
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle('Comprehensive Hyperparameter Optimization Summary', fontsize=16, fontweight='bold')
        
        # Extract data for visualization
        method_performance = {}
        model_performance = {}
        pair_performance = {}
        
        for method, method_results in results.items():
            method_scores = []
            for model_type, model_results in method_results.items():
                for pair, pair_results in model_results.items():
                    score = pair_results.get('best_value')
                    if score is not None:
                        method_scores.append(score)
                        
                        # Track by model and pair
                        model_performance.setdefault(model_type, []).append(score)
                        pair_performance.setdefault(pair, []).append(score)
            
            if method_scores:
                method_performance[method] = np.mean(method_scores)
        
        # Plot 1: Performance by optimization method
        if method_performance:
            methods = list(method_performance.keys())
            scores = list(method_performance.values())
            
            bars = axes[0, 0].bar(methods, scores, alpha=0.7)
            axes[0, 0].set_title('Average Performance by Optimization Method')
            axes[0, 0].set_ylabel('Average Best Score')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            for bar, score in zip(bars, scores):
                axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                               f'{score:.4f}', ha='center', va='bottom')
        
        # Plot 2: Performance by model type
        if model_performance:
            models = list(model_performance.keys())
            avg_scores = [np.mean(scores) for scores in model_performance.values()]
            
            bars = axes[0, 1].bar(models, avg_scores, alpha=0.7, color='green')
            axes[0, 1].set_title('Average Performance by Model Type')
            axes[0, 1].set_ylabel('Average Best Score')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            for bar, score in zip(bars, avg_scores):
                axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                               f'{score:.4f}', ha='center', va='bottom')
        
        # Plot 3: Performance by currency pair
        if pair_performance:
            pairs = list(pair_performance.keys())
            avg_scores = [np.mean(scores) for scores in pair_performance.values()]
            
            bars = axes[1, 0].bar(pairs, avg_scores, alpha=0.7, color='orange')
            axes[1, 0].set_title('Average Performance by Currency Pair')
            axes[1, 0].set_ylabel('Average Best Score')
            
            for bar, score in zip(bars, avg_scores):
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                               f'{score:.4f}', ha='center', va='bottom')
        
        # Plot 4: Performance distribution
        all_scores = []
        for method_results in results.values():
            for model_results in method_results.values():
                for pair_results in model_results.values():
                    score = pair_results.get('best_value')
                    if score is not None:
                        all_scores.append(score)
        
        if all_scores:
            axes[1, 1].hist(all_scores, bins=20, alpha=0.7, edgecolor='black')
            axes[1, 1].axvline(np.mean(all_scores), color='red', linestyle='--', linewidth=2, 
                              label=f'Mean: {np.mean(all_scores):.4f}')
            axes[1, 1].set_title('Distribution of Best Scores')
            axes[1, 1].set_xlabel('Best Score')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("output/hyperparameter_tuning/comprehensive_optimization_summary.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("📊 Comprehensive summary visualization created")

def main():
    """Enhanced main function for hyperparameter optimization."""
    parser = argparse.ArgumentParser(description='Enhanced Hyperparameter Optimization for Master\'s Thesis')
    
    # Enhanced optimization options
    parser.add_argument('--model', type=str, 
                       choices=['enhanced_cnn_lstm', 'advanced_tft', 'enhanced_xgboost', 'all'], 
                       default='all', help='Model type to optimize')
    parser.add_argument('--pair', type=str, 
                       choices=['EURUSD', 'GBPUSD', 'USDJPY', 'Bagging', 'all'], 
                       default='all', help='Currency pair to optimize')
    parser.add_argument('--trials', type=int, default=100, 
                       help='Number of optimization trials (increased for thesis)')
    parser.add_argument('--method', type=str, 
                       choices=['bayesian_tpe', 'bayesian_cmaes', 'multi_objective', 'all'], 
                       default='all', help='Optimization method')
    parser.add_argument('--validation', type=str, 
                       choices=['standard', 'walkforward'], 
                       default='standard', help='Validation method')
    parser.add_argument('--no-visualization', action='store_true', 
                       help='Skip creating visualizations')
    
    # Enhanced parallel processing
    parser.add_argument('--parallel', action='store_true', 
                       help='Enable parallel optimization')
    parser.add_argument('--n-jobs', type=int, default=multiprocessing.cpu_count()//2, 
                       help='Number of parallel jobs')
    
    args = parser.parse_args()
    
    # Initialize enhanced forex prediction system
    logger.info("🚀 Initializing Enhanced Forex Prediction System for optimization")
    
    try:
        # Load data and prepare for optimization
        forex_pred = EnhancedForexPrediction()
        forex_pred.load_data()
        forex_pred.preprocess_data()
        forex_pred.calculate_enhanced_features()
        forex_pred.select_features()
        
        logger.info("✅ Data preparation completed for optimization")
        
        # Initialize optimizer
        optimizer = EnhancedHyperparameterOptimizer()
        
        # Set up models and pairs
        if args.model == 'all':
            model_types = ['enhanced_cnn_lstm', 'advanced_tft', 'enhanced_xgboost']
        else:
            model_types = [args.model]
        
        if args.pair == 'all':
            pairs = ['EURUSD', 'GBPUSD', 'USDJPY']
            if forex_pred.config.get('use_bagging', True):
                pairs.append('Bagging')
        else:
            pairs = [args.pair]
        
        if args.method == 'all':
            methods = ['bayesian_tpe', 'multi_objective']
        else:
            methods = [args.method]
        
        # Run comprehensive optimization
        results = optimizer.run_comprehensive_optimization(
            forex_pred, 
            pairs=pairs, 
            models=model_types, 
            n_trials=args.trials, 
            methods=methods
        )
        
        # Print summary
        print("\n" + "="*80)
        print("🎯 ENHANCED HYPERPARAMETER OPTIMIZATION COMPLETED!")
        print("="*80)
        
        total_optimizations = sum(len(method_results) * len(model_results) 
                                for method_results in results.values() 
                                for model_results in method_results.values())
        
        print(f"📊 Total optimizations completed: {total_optimizations}")
        print(f"🔬 Optimization methods used: {', '.join(methods)}")
        print(f"🤖 Models optimized: {', '.join(model_types)}")
        print(f"💱 Currency pairs: {', '.join(pairs)}")
        print(f"⚙️ Trials per optimization: {args.trials}")
        
        # Show best results for each method
        for method, method_results in results.items():
            print(f"\n🏆 Best results for {method}:")
            best_overall = None
            best_score = -float('inf')
            
            for model_type, model_results in method_results.items():
                for pair, pair_results in model_results.items():
                    score = pair_results.get('best_value')
                    if score is not None and score > best_score:
                        best_score = score
                        best_overall = f"{pair}_{model_type}"
            
            if best_overall:
                print(f"  🥇 {best_overall}: {best_score:.4f}")
        
        print("="*80)
        
    except Exception as e:
        logger.error(f"❌ Error in enhanced hyperparameter optimization: {e}", exc_info=True)
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
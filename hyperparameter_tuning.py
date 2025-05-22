"""
Enhanced Hyperparameter Tuning for Forex Prediction Models
- Improved visualization and reporting
- Better handling of hyperparameter search spaces
- Integration with enhanced pipeline
"""

import os
import json
import argparse
import logging
from datetime import datetime
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import optuna
from forex_prediction import ForexPrediction

import xgboost as xgb

# Set up logging
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

def get_enhanced_hyperparameter_space(model_type):
    """Define enhanced hyperparameter search space for each model type."""
    if model_type == 'cnn_lstm':
        return {
            'cnn_filters': (32, 256),  # Number of filters in CNN layers
            'cnn_kernel_size': (2, 5),  # Size of CNN kernels
            'lstm_units': (50, 300),    # Number of LSTM units
            'dropout_rate': (0.1, 0.7),  # Dropout rate
            'learning_rate': (1e-5, 5e-2)  # Learning rate
        }
    elif model_type == 'tft':
        return {
            'hidden_units': (32, 256),  # Number of hidden units
            'num_heads': (1, 12),        # Number of attention heads
            'dropout_rate': (0.1, 0.7),  # Dropout rate
            'learning_rate': (1e-5, 5e-2)  # Learning rate
        }
    elif model_type == 'xgboost':
        return {
            'max_depth': (3, 15),        # Maximum tree depth
            'learning_rate': (0.001, 0.5),  # Learning rate
            'n_estimators': (50, 500),    # Number of estimators
            'subsample': (0.5, 1.0),      # Subsample ratio
            'colsample_bytree': (0.5, 1.0),  # Column sample ratio
            'gamma': (0, 10),             # Minimum loss reduction for partition
            'reg_alpha': (0, 10),         # L1 regularization
            'reg_lambda': (0.1, 10)       # L2 regularization
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def create_hyperparameter_report(study, model_type, pair, output_dir):
    """Create a comprehensive hyperparameter tuning report."""
    report_dir = os.path.join(output_dir, "hyperparameter_reports")
    os.makedirs(report_dir, exist_ok=True)
    
    # Create visualizations
    try:
        # Optimization history
        fig = plt.figure(figsize=(15, 10))
        
        # Plot 1: Optimization history
        plt.subplot(2, 3, 1)
        trials = study.trials
        values = [trial.value for trial in trials if trial.value is not None]
        plt.plot(values, marker='o', alpha=0.7)
        plt.title(f'{model_type} on {pair}\nOptimization History')
        plt.xlabel('Trial')
        plt.ylabel('Validation Accuracy')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Best value progression
        plt.subplot(2, 3, 2)
        best_values = []
        current_best = -float('inf')
        for value in values:
            if value > current_best:
                current_best = value
            best_values.append(current_best)
        plt.plot(best_values, marker='o', color='green', alpha=0.7)
        plt.title('Best Value Progression')
        plt.xlabel('Trial')
        plt.ylabel('Best Validation Accuracy')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Parameter importance (if available)
        try:
            if len(study.trials) > 10:  # Need enough trials for importance
                importance = optuna.importance.get_param_importances(study)
                params = list(importance.keys())
                values = list(importance.values())
                
                plt.subplot(2, 3, 3)
                plt.barh(params, values, alpha=0.7)
                plt.title('Parameter Importance')
                plt.xlabel('Importance')
                plt.grid(True, alpha=0.3)
        except Exception as e:
            logger.warning(f"Could not calculate parameter importance: {e}")
        
        # Plot 4: Trial distribution
        plt.subplot(2, 3, 4)
        plt.hist(values, bins=min(20, len(values)//2), alpha=0.7, edgecolor='black')
        plt.title('Trial Score Distribution')
        plt.xlabel('Validation Accuracy')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Plot 5: Convergence analysis
        plt.subplot(2, 3, 5)
        if len(values) > 10:
            window_size = max(5, len(values)//10)
            moving_avg = pd.Series(values).rolling(window=window_size).mean()
            plt.plot(values, alpha=0.3, label='Individual Trials')
            plt.plot(moving_avg, color='red', linewidth=2, label=f'Moving Average (window={window_size})')
            plt.title('Convergence Analysis')
            plt.xlabel('Trial')
            plt.ylabel('Validation Accuracy')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Plot 6: Best parameters visualization
        plt.subplot(2, 3, 6)
        best_params = study.best_params
        param_names = list(best_params.keys())
        param_values = list(best_params.values())
        
        # Normalize values for better visualization
        normalized_values = []
        for i, (param, value) in enumerate(best_params.items()):
            if isinstance(value, (int, float)):
                space = get_enhanced_hyperparameter_space(model_type)
                if param in space:
                    min_val, max_val = space[param]
                    normalized = (value - min_val) / (max_val - min_val)
                    normalized_values.append(normalized)
                else:
                    normalized_values.append(value)
            else:
                normalized_values.append(i)  # For categorical parameters
        
        plt.bar(range(len(param_names)), normalized_values, alpha=0.7)
        plt.xticks(range(len(param_names)), param_names, rotation=45)
        plt.title('Best Parameters (Normalized)')
        plt.ylabel('Normalized Value')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(report_dir, f"{pair}_{model_type}_hyperparameter_analysis.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Hyperparameter analysis plot saved to {plot_path}")
        
    except Exception as e:
        logger.warning(f"Error creating hyperparameter visualizations: {e}")
    
    # Create HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Hyperparameter Tuning Report - {model_type} on {pair}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }}
            .header {{
                background-color: #2c3e50;
                color: white;
                padding: 20px;
                text-align: center;
                border-radius: 10px;
            }}
            .section {{
                background-color: white;
                margin: 20px 0;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .metric {{
                display: inline-block;
                background-color: #ecf0f1;
                padding: 10px;
                margin: 5px;
                border-radius: 5px;
                min-width: 150px;
                text-align: center;
            }}
            .param-table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            .param-table th, .param-table td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            .param-table th {{
                background-color: #3498db;
                color: white;
            }}
            .best-param {{
                background-color: #d5f4e6;
                font-weight: bold;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🔧 Hyperparameter Tuning Report</h1>
            <h2>{model_type.upper()} on {pair}</h2>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <h2>📊 Optimization Summary</h2>
            <div class="metric">
                <div style="font-size: 24px; font-weight: bold; color: #27ae60;">{study.best_value:.4f}</div>
                <div style="font-size: 12px; color: #7f8c8d;">Best Validation Accuracy</div>
            </div>
            <div class="metric">
                <div style="font-size: 24px; font-weight: bold; color: #2c3e50;">{len(study.trials)}</div>
                <div style="font-size: 12px; color: #7f8c8d;">Total Trials</div>
            </div>
            <div class="metric">
                <div style="font-size: 24px; font-weight: bold; color: #e74c3c;">{study.best_trial.number}</div>
                <div style="font-size: 12px; color: #7f8c8d;">Best Trial Number</div>
            </div>
        </div>
        
        <div class="section">
            <h2>🎯 Best Parameters</h2>
            <table class="param-table">
                <tr>
                    <th>Parameter</th>
                    <th>Best Value</th>
                    <th>Search Range</th>
                </tr>
    """
    
    # Add parameter details
    space = get_enhanced_hyperparameter_space(model_type)
    for param, value in study.best_params.items():
        search_range = f"{space.get(param, ('N/A', 'N/A'))[0]} - {space.get(param, ('N/A', 'N/A'))[1]}"
        html_content += f"""
                <tr class="best-param">
                    <td>{param}</td>
                    <td>{value}</td>
                    <td>{search_range}</td>
                </tr>
        """
    
    html_content += f"""
            </table>
        </div>
        
        <div class="section">
            <h2>📈 Optimization Analysis</h2>
            <img src="{pair}_{model_type}_hyperparameter_analysis.png" alt="Hyperparameter Analysis" style="max-width: 100%; height: auto;">
        </div>
        
        <div class="section">
            <h2>📋 Trial History</h2>
            <p>Showing top 10 trials:</p>
            <table class="param-table">
                <tr>
                    <th>Trial</th>
                    <th>Validation Accuracy</th>
                    <th>Parameters</th>
                </tr>
    """
    
    # Sort trials by value and show top 10
    sorted_trials = sorted([t for t in study.trials if t.value is not None], 
                          key=lambda x: x.value, reverse=True)[:10]
    
    for trial in sorted_trials:
        params_str = ", ".join([f"{k}: {v}" for k, v in trial.params.items()])
        row_class = "best-param" if trial.number == study.best_trial.number else ""
        html_content += f"""
                <tr class="{row_class}">
                    <td>{trial.number}</td>
                    <td>{trial.value:.4f}</td>
                    <td style="font-size: 10px;">{params_str}</td>
                </tr>
        """
    
    html_content += """
            </table>
        </div>
    </body>
    </html>
    """
    
    # Save HTML report
    html_path = os.path.join(report_dir, f"{pair}_{model_type}_hyperparameter_report.html")
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"Hyperparameter tuning report saved to {html_path}")
    return html_path

def enhanced_tune_hyperparameters(model_type, pair, n_trials=50, visualization=True):
    """Enhanced hyperparameter tuning with better reporting."""
    logger.info(f"🔧 Starting enhanced hyperparameter tuning for {model_type} on {pair}")
    
    # Initialize the Forex Prediction system
    forex_pred = ForexPrediction()
    
    # Load and preprocess data
    forex_pred.load_data()
    forex_pred.preprocess_data()
    forex_pred.calculate_technical_indicators()
    forex_pred.select_features()
    
    # Create output directories
    output_dir = "output"
    tuning_dir = os.path.join(output_dir, "hyperparameter_tuning")
    os.makedirs(tuning_dir, exist_ok=True)
    
    # Enhanced objective functions with better error handling
    def enhanced_objective(trial):
        try:
            # Get hyperparameter space
            space = get_enhanced_hyperparameter_space(model_type)
            
            if model_type == 'cnn_lstm':
                hyperparams = {
                    'cnn_filters': trial.suggest_int('cnn_filters', space['cnn_filters'][0], space['cnn_filters'][1], step=32),
                    'cnn_kernel_size': trial.suggest_int('cnn_kernel_size', space['cnn_kernel_size'][0], space['cnn_kernel_size'][1]),
                    'lstm_units': trial.suggest_int('lstm_units', space['lstm_units'][0], space['lstm_units'][1], step=25),
                    'dropout_rate': trial.suggest_float('dropout_rate', space['dropout_rate'][0], space['dropout_rate'][1]),
                    'learning_rate': trial.suggest_float('learning_rate', space['learning_rate'][0], space['learning_rate'][1], log=True)
                }
            elif model_type == 'tft':
                hyperparams = {
                    'hidden_units': trial.suggest_int('hidden_units', space['hidden_units'][0], space['hidden_units'][1], step=32),
                    'num_heads': trial.suggest_int('num_heads', space['num_heads'][0], space['num_heads'][1]),
                    'dropout_rate': trial.suggest_float('dropout_rate', space['dropout_rate'][0], space['dropout_rate'][1]),
                    'learning_rate': trial.suggest_float('learning_rate', space['learning_rate'][0], space['learning_rate'][1], log=True)
                }
            elif model_type == 'xgboost':
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
            
            logger.info(f"Trial {trial.number}: Testing {model_type} with hyperparams: {hyperparams}")
            
            # Get training data
            if pair == 'Bagging':
                train_data = forex_pred.bagging_data['train']
            else:
                train_data = forex_pred.selected_features[pair]['train']
            
            # Prepare data based on model type
            is_lstm = model_type in ['cnn_lstm', 'tft']
            X_train, y_train, _, _ = forex_pred.prepare_model_data(train_data, is_lstm=is_lstm)
            
            # Handle NaN and Inf values
            if np.isnan(X_train).any() or np.isnan(y_train).any() or np.isinf(X_train).any() or np.isinf(y_train).any():
                logger.warning(f"NaN or Inf values detected in {pair} {model_type} training data. Fixing...")
                X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
                y_train = np.nan_to_num(y_train, nan=0.0, posinf=1.0, neginf=0.0)
            
            # Split training and validation sets
            val_size = int(len(X_train) * 0.2)
            X_val, y_val = X_train[-val_size:], y_train[-val_size:]
            X_train, y_train = X_train[:-val_size], y_train[:-val_size]
            
            # Train model based on type
            if model_type == 'cnn_lstm':
                input_shape = (X_train.shape[1], X_train.shape[2])
                model = forex_pred.build_cnn_lstm_model(input_shape, pair)
                
                # Override hyperparameters for this trial
                forex_pred.hyperparameters['cnn_lstm'][pair if pair != 'Bagging' else 'Bagging'] = hyperparams
                model = forex_pred.build_cnn_lstm_model(input_shape, pair)
                
                from tensorflow.keras.callbacks import EarlyStopping
                callbacks = [EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
                
                history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                                  epochs=50, batch_size=32, callbacks=callbacks, verbose=0)
                return history.history['val_accuracy'][-1]
                
            elif model_type == 'tft':
                input_dim = X_train.shape[2]
                
                # Override hyperparameters for this trial
                forex_pred.hyperparameters['tft'][pair if pair != 'Bagging' else 'Bagging'] = hyperparams
                model = forex_pred.build_tft_model(input_dim, pair)
                
                from tensorflow.keras.callbacks import EarlyStopping
                callbacks = [EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
                
                history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                                  epochs=50, batch_size=32, callbacks=callbacks, verbose=0)
                return history.history['val_accuracy'][-1]
                
            elif model_type == 'xgboost':
                # Override hyperparameters for this trial
                forex_pred.hyperparameters['xgboost'][pair if pair != 'Bagging' else 'Bagging'] = hyperparams
                model = forex_pred.build_xgboost_model(pair)
                
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
                return model.score(X_val, y_val)
                
        except Exception as e:
            logger.error(f"Error in trial {trial.number}: {e}")
            return 0.0
    
    # Create and run the Optuna study
    study = optuna.create_study(direction='maximize', 
                               study_name=f"enhanced_{model_type}_{pair}_tuning")
    study.optimize(enhanced_objective, n_trials=n_trials)
    
    # Get the best hyperparameters
    best_params = study.best_params
    best_value = study.best_value
    
    # Log the results
    logger.info(f"🎯 Best hyperparameters for {model_type} on {pair}: {best_params}")
    logger.info(f"🎯 Best validation accuracy: {best_value:.4f}")
    
    # Save the best hyperparameters to file
    output_file = os.path.join(tuning_dir, f"{pair}_{model_type}_best_hyperparams.json")
    with open(output_file, 'w') as f:
        json.dump(best_params, f, indent=2)
    
    # Create enhanced visualizations and report
    if visualization:
        report_path = create_hyperparameter_report(study, model_type, pair, output_dir)
        logger.info(f"📊 Enhanced hyperparameter report created: {report_path}")
    
    return best_params, study

def main():
    """Main function for enhanced hyperparameter tuning."""
    parser = argparse.ArgumentParser(description='Enhanced Hyperparameter Tuning for Forex Prediction Models')
    parser.add_argument('--model', type=str, choices=['cnn_lstm', 'tft', 'xgboost', 'all'], default='all',
                        help='Model type to tune (default: all)')
    parser.add_argument('--pair', type=str, choices=['EURUSD', 'GBPUSD', 'USDJPY', 'Bagging', 'all'], default='all',
                        help='Currency pair to tune (default: all)')
    parser.add_argument('--trials', type=int, default=50, help='Number of optimization trials (default: 50)')
    parser.add_argument('--no-visualization', action='store_true', help='Skip creating visualizations')
    
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
    
    # Run enhanced hyperparameter tuning
    results = {}
    total_combinations = len(model_types) * len(pairs)
    current_combination = 0
    
    for model_type in model_types:
        results[model_type] = {}
        for pair in pairs:
            current_combination += 1
            logger.info(f"🚀 Progress: {current_combination}/{total_combinations} - Tuning {model_type} for {pair} with {args.trials} trials")
            
            best_params, study = enhanced_tune_hyperparameters(
                model_type, pair, args.trials, not args.no_visualization
            )
            results[model_type][pair] = {
                'best_params': best_params,
                'best_value': study.best_value,
                'n_trials': len(study.trials)
            }
    
    # Save overall results
    output_dir = "output"
    tuning_dir = os.path.join(output_dir, "hyperparameter_tuning")
    with open(os.path.join(tuning_dir, "enhanced_tuning_results_summary.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("🎯 Enhanced hyperparameter tuning completed successfully!")
    
    # Print summary
    print("\n" + "="*80)
    print("🎯 ENHANCED HYPERPARAMETER TUNING COMPLETED!")
    print("="*80)
    for model_type, model_results in results.items():
        print(f"\n📊 {model_type.upper()} Results:")
        for pair, pair_results in model_results.items():
            print(f"  🎯 {pair}: Best accuracy = {pair_results['best_value']:.4f}")
    print("="*80)

if __name__ == "__main__":
    main()
"""
Enhanced Forex Trend Prediction Model
- Improved hyperparameter loading from JSON files
- Comprehensive visualization capabilities
- Models: CNN-LSTM, TFT, XGBoost
- Currency Pairs: EURUSD, GBPUSD, USDJPY
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time
import logging
import optuna
import json
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from functools import partial
from collections import defaultdict

# ML Libraries
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, roc_auc_score
from sklearn.feature_selection import SelectFromModel, mutual_info_regression, SelectKBest, VarianceThreshold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb

# Deep Learning Libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten, BatchNormalization, Input, Concatenate, MultiHeadAttention
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# Set up logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"forex_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create output directories
output_dir = "output"
models_dir = os.path.join(output_dir, "models")
results_dir = os.path.join(output_dir, "results")
plots_dir = os.path.join(output_dir, "plots")
feature_dir = os.path.join(output_dir, "features")
hyperparams_dir = os.path.join(output_dir, "hyperparameter_tuning")

for directory in [output_dir, models_dir, results_dir, plots_dir, feature_dir, hyperparams_dir]:
    os.makedirs(directory, exist_ok=True)

# Global constants
WINDOW_SIZE = 60  # 60 hours (lookback period)
PREDICTION_HORIZON = 1  # Predict direction 1 hour ahead
BATCH_SIZE = 32
EPOCHS = 100
PATIENCE = 20
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2
RANDOM_STATE = 42
TRAIN_START = '2020-01-01'
TRAIN_END = '2021-12-31'
TEST_START = '2022-01-01'
TEST_END = '2022-04-30'

# Enhanced plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Enable GPU growth if available
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"Using GPU: {len(gpus)} GPUs available")
    else:
        logger.info("No GPU available, using CPU")
except Exception as e:
    logger.warning(f"Error setting up GPU: {e}")

class ForexPrediction:
    def __init__(self, config=None):
        """Initialize the Forex Prediction system with configuration."""
        # Load default config if none provided
        if config is None:
            self.config = {
                'window_size': WINDOW_SIZE,
                'prediction_horizon': PREDICTION_HORIZON,
                'batch_size': BATCH_SIZE,
                'epochs': EPOCHS,
                'patience': PATIENCE,
                'test_size': TEST_SIZE,
                'validation_size': VALIDATION_SIZE,
                'random_state': RANDOM_STATE,
                'train_start': TRAIN_START,
                'train_end': TRAIN_END,
                'test_start': TEST_START,
                'test_end': TEST_END,
                'feature_selection_methods': ['random_forest', 'mutual_info', 'pca'],
                'n_features': 30,  # Number of features to select
                'models_to_train': ['cnn_lstm', 'tft', 'xgboost'],
                'currency_pairs': ['EURUSD', 'GBPUSD', 'USDJPY'],
                'use_bagging': True,
                'scaler_type': 'standard',  # 'standard', 'minmax', 'robust'
                'hyperparameter_tuning': False,
                'n_trials': 25,  # Number of optimization trials
                'evaluation_metrics': ['annual_return', 'win_rate', 'market_condition', 'buy_hold_comparison', 'single_bagging_comparison']
            }
        else:
            self.config = config
        
        # Initialize class variables
        self.data = {}  # Dict to store raw dataframes for each currency pair
        self.preprocessed_data = {}  # Dict to store preprocessed data
        self.enhanced_features = {}  # Dict to store data with enhanced features
        self.selected_features = {}  # Dict to store data with selected features
        self.models = {}  # Dict to store trained models
        self.scalers = {}  # Dict to store fitted scalers
        self.results = {}  # Dict to store evaluation results
        self.bagging_data = None  # Combined data for bagging approach
        self.hyperparameters = {}  # Dict to store loaded hyperparameters
        
        # Load hyperparameters at initialization
        self.load_hyperparameters()
        
        logger.info("Forex Prediction system initialized")
        logger.info(f"Configuration: {json.dumps(self.config, indent=2)}")
    
    def load_hyperparameters(self):
        """Load hyperparameters from JSON files."""
        logger.info("Loading hyperparameters from JSON files")
        
        # Define default hyperparameters as fallback - FIXED: Added 'objective' for XGBoost
        default_hyperparams = {
            'cnn_lstm': {
                'cnn_filters': 64,
                'cnn_kernel_size': 3,
                'lstm_units': 100,
                'dropout_rate': 0.3,
                'learning_rate': 0.001
            },
            'tft': {
                'hidden_units': 64,
                'num_heads': 4,
                'dropout_rate': 0.2,
                'learning_rate': 0.001
            },
            'xgboost': {
                'max_depth': 5,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 0,
                'reg_alpha': 0,
                'reg_lambda': 1,
                'objective': 'binary:logistic'  # FIXED: Added missing objective
            }
        }
        
        # Initialize hyperparameters dict
        self.hyperparameters = {}
        
        # Load hyperparameters for each model and currency pair combination
        for model_type in self.config['models_to_train']:
            self.hyperparameters[model_type] = {}
            
            # Load for individual currency pairs
            for pair in self.config['currency_pairs']:
                param_file = os.path.join(hyperparams_dir, f"{pair}_{model_type}_best_hyperparams.json")
                
                if os.path.exists(param_file):
                    try:
                        with open(param_file, 'r') as f:
                            params = json.load(f)
                        
                        # FIXED: Ensure XGBoost models have 'objective' parameter
                        if model_type == 'xgboost' and 'objective' not in params:
                            params['objective'] = 'binary:logistic'
                            logger.info(f"Added missing 'objective' parameter to {pair} {model_type}")
                        
                        self.hyperparameters[model_type][pair] = params
                        logger.info(f"Loaded hyperparameters for {pair} {model_type}: {params}")
                    except Exception as e:
                        logger.warning(f"Error loading hyperparameters for {pair} {model_type}: {e}")
                        self.hyperparameters[model_type][pair] = default_hyperparams[model_type].copy()
                else:
                    logger.warning(f"Hyperparameter file not found for {pair} {model_type}, using defaults")
                    self.hyperparameters[model_type][pair] = default_hyperparams[model_type].copy()
            
            # Load for bagging models
            if self.config['use_bagging']:
                bagging_param_file = os.path.join(hyperparams_dir, f"Bagging_{model_type}_best_hyperparams.json")
                
                if os.path.exists(bagging_param_file):
                    try:
                        with open(bagging_param_file, 'r') as f:
                            params = json.load(f)
                        
                        # FIXED: Ensure XGBoost models have 'objective' parameter
                        if model_type == 'xgboost' and 'objective' not in params:
                            params['objective'] = 'binary:logistic'
                            logger.info(f"Added missing 'objective' parameter to Bagging {model_type}")
                        
                        self.hyperparameters[model_type]['Bagging'] = params
                        logger.info(f"Loaded hyperparameters for Bagging {model_type}: {params}")
                    except Exception as e:
                        logger.warning(f"Error loading hyperparameters for Bagging {model_type}: {e}")
                        self.hyperparameters[model_type]['Bagging'] = default_hyperparams[model_type].copy()
                else:
                    logger.warning(f"Hyperparameter file not found for Bagging {model_type}, using defaults")
                    self.hyperparameters[model_type]['Bagging'] = default_hyperparams[model_type].copy()
    
    def get_hyperparameters(self, model_type, pair_or_bagging):
        """Get hyperparameters for a specific model and currency pair/bagging."""
        try:
            return self.hyperparameters[model_type][pair_or_bagging]
        except KeyError:
            logger.warning(f"Hyperparameters not found for {model_type} {pair_or_bagging}, using defaults")
            # FIXED: Return default hyperparameters with proper 'objective' for XGBoost
            default_hyperparams = {
                'cnn_lstm': {
                    'cnn_filters': 64,
                    'cnn_kernel_size': 3,
                    'lstm_units': 100,
                    'dropout_rate': 0.3,
                    'learning_rate': 0.001
                },
                'tft': {
                    'hidden_units': 64,
                    'num_heads': 4,
                    'dropout_rate': 0.2,
                    'learning_rate': 0.001
                },
                'xgboost': {
                    'max_depth': 5,
                    'learning_rate': 0.1,
                    'n_estimators': 100,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'gamma': 0,
                    'reg_alpha': 0,
                    'reg_lambda': 1,
                    'objective': 'binary:logistic'  # FIXED: Added missing objective
                }
            }
            return default_hyperparams.get(model_type, {})

    #####################################
    # VISUALIZATION FUNCTIONS
    #####################################
    
    def create_data_visualizations(self):
        """Create comprehensive data visualizations before training."""
        logger.info("Creating data visualizations")
        
        # Create subplots for time series plots
        fig, axes = plt.subplots(len(self.config['currency_pairs']), 1, figsize=(15, 4*len(self.config['currency_pairs'])))
        if len(self.config['currency_pairs']) == 1:
            axes = [axes]
        
        for idx, pair in enumerate(self.config['currency_pairs']):
            # Time Series Plot: Close Price during training period
            train_data = self.preprocessed_data[pair]['train']
            
            axes[idx].plot(train_data.index, train_data['Close'], linewidth=1, alpha=0.8)
            axes[idx].set_title(f'{pair} Close Price - Training Period ({self.config["train_start"]} to {self.config["train_end"]})', 
                               fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Date')
            axes[idx].set_ylabel('Price')
            axes[idx].grid(True, alpha=0.3)
            axes[idx].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "data_time_series_plots.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Correlation Heatmap for OHLCV features
        fig, axes = plt.subplots(1, len(self.config['currency_pairs']), figsize=(6*len(self.config['currency_pairs']), 5))
        if len(self.config['currency_pairs']) == 1:
            axes = [axes]
        
        for idx, pair in enumerate(self.config['currency_pairs']):
            train_data = self.preprocessed_data[pair]['train']
            # Calculate correlation matrix for OHLCV features
            ohlcv_data = train_data[['Open', 'High', 'Low', 'Close', 'Volume']]
            correlation_matrix = ohlcv_data.corr()
            
            # Create heatmap
            sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0, 
                       square=True, ax=axes[idx], cbar_kws={"shrink": .8})
            axes[idx].set_title(f'{pair} OHLCV Correlation', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "data_correlation_heatmaps.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Distribution plots for key features
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        feature_names = ['Returns', 'Log_Returns', 'Volume', 'Range']
        
        for idx, feature in enumerate(feature_names):
            if idx < len(axes):
                for pair in self.config['currency_pairs']:
                    train_data = self.preprocessed_data[pair]['train']
                    if feature == 'Range':
                        feature_data = train_data['High'] - train_data['Low']
                    else:
                        feature_data = train_data[feature]
                    
                    # Remove outliers for better visualization
                    q1, q3 = feature_data.quantile([0.25, 0.75])
                    iqr = q3 - q1
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    filtered_data = feature_data[(feature_data >= lower) & (feature_data <= upper)]
                    
                    axes[idx].hist(filtered_data.dropna(), bins=50, alpha=0.6, label=pair, density=True)
                
                axes[idx].set_title(f'Distribution of {feature}', fontsize=12, fontweight='bold')
                axes[idx].set_xlabel(feature)
                axes[idx].set_ylabel('Density')
                axes[idx].legend()
                axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "data_feature_distributions.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Data visualizations saved to plots directory")
    
    def create_training_visualizations(self, model_key, history):
        """Create training loss and accuracy plots."""
        if 'history' not in history:
            logger.warning(f"No training history found for {model_key}")
            return
        
        hist = history['history']
        
        # Create training plots
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        epochs = range(1, len(hist['loss']) + 1)
        axes[0].plot(epochs, hist['loss'], 'b-', label='Training Loss', linewidth=2)
        axes[0].plot(epochs, hist['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        axes[0].set_title(f'{model_key} Training and Validation Loss', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[1].plot(epochs, hist['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
        axes[1].plot(epochs, hist['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
        axes[1].set_title(f'{model_key} Training and Validation Accuracy', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{model_key}_training_history.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_prediction_visualizations(self, model_key, y_true, y_pred, y_pred_proba, test_data):
        """Create prediction result visualizations."""
        # Convert predictions to price predictions for visualization
        close_prices = test_data['Close'].values
        
        # Align close prices with predictions
        if len(close_prices) > len(y_pred) + self.config['window_size']:
            close_prices = close_prices[self.config['window_size']:len(y_pred) + self.config['window_size']]
        else:
            close_prices = close_prices[-len(y_pred):]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        
        # 1. Time series of actual vs predicted direction
        time_index = range(len(y_true))
        axes[0, 0].plot(time_index, y_true, 'b-', label='Actual Direction', alpha=0.7, linewidth=2)
        axes[0, 0].plot(time_index, y_pred, 'r-', label='Predicted Direction', alpha=0.7, linewidth=2)
        axes[0, 0].fill_between(time_index, y_true, y_pred, alpha=0.3, color='gray')
        axes[0, 0].set_title(f'{model_key} - Actual vs Predicted Direction', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Time Steps')
        axes[0, 0].set_ylabel('Direction (0=Down, 1=Up)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Price movement with prediction confidence
        axes[0, 1].plot(time_index, close_prices, 'k-', label='Close Price', linewidth=2)
        
        # Color-code based on prediction confidence
        for i in range(len(y_pred_proba)):
            confidence = abs(y_pred_proba[i] - 0.5) * 2  # Convert to 0-1 scale
            color = 'green' if y_pred[i] == 1 else 'red'
            alpha = max(0.1, confidence)
            axes[0, 1].scatter(i, close_prices[i], c=color, alpha=alpha, s=10)
        
        axes[0, 1].set_title(f'{model_key} - Price Movement with Prediction Confidence', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Time Steps')
        axes[0, 1].set_ylabel('Price')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Scatter plot: Actual vs Predicted probabilities
        axes[1, 0].scatter(y_true, y_pred_proba, alpha=0.6, s=20)
        axes[1, 0].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Prediction')
        axes[1, 0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
        axes[1, 0].axvline(x=0.5, color='gray', linestyle='--', alpha=0.7)
        axes[1, 0].set_title(f'{model_key} - Actual vs Predicted Probabilities', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Actual Direction')
        axes[1, 0].set_ylabel('Predicted Probability')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Prediction confidence distribution
        axes[1, 1].hist(y_pred_proba, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
        axes[1, 1].set_title(f'{model_key} - Prediction Confidence Distribution', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Predicted Probability')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{model_key}_prediction_analysis.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_evaluation_visualizations(self, results):
        """Create comprehensive evaluation metric visualizations."""
        logger.info("Creating evaluation metric visualizations")
        
        # Extract data for visualization
        model_names = []
        annual_returns = []
        win_rates = []
        max_drawdowns = []
        sharpe_ratios = []
        buy_hold_returns = []
        
        # Separate single and bagging models
        single_models = {}
        bagging_models = {}
        
        for model_key, model_results in results.items():
            if 'trading_performance' in model_results:
                perf = model_results['trading_performance']
                model_names.append(model_key)
                annual_returns.append(perf['annual_return'])
                win_rates.append(perf['win_rate'] * 100)
                max_drawdowns.append(perf['max_drawdown'])
                sharpe_ratios.append(perf['sharpe_ratio'])
                buy_hold_returns.append(perf['buy_hold_annual_return'])
                
                # Categorize models
                if 'Bagging' in model_key:
                    bagging_models[model_key] = model_results
                else:
                    single_models[model_key] = model_results
        
        # 1. Overall Performance Comparison
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        
        # Annual Return comparison
        bars = axes[0, 0].bar(model_names, annual_returns, color='skyblue', alpha=0.8)
        axes[0, 0].bar(model_names, buy_hold_returns, color='lightgray', alpha=0.5)
        axes[0, 0].set_title('Annual Return Comparison', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Annual Return (%)')
        axes[0, 0].tick_params(axis='x', rotation=90)
        axes[0, 0].grid(axis='y', linestyle='--', alpha=0.7)
        axes[0, 0].legend(['Model Return', 'Buy & Hold Return'])
        
        # Add value labels on bars
        for bar, value in zip(bars, annual_returns):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                           f'{value:.1f}%', ha='center', va='bottom', fontsize=8)
        
        # Win Rate comparison
        bars = axes[0, 1].bar(model_names, win_rates, color='lightgreen', alpha=0.8)
        axes[0, 1].axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Random (50%)')
        axes[0, 1].set_title('Win Rate Comparison', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Win Rate (%)')
        axes[0, 1].tick_params(axis='x', rotation=90)
        axes[0, 1].grid(axis='y', linestyle='--', alpha=0.7)
        axes[0, 1].legend()
        
        # Add value labels on bars
        for bar, value in zip(bars, win_rates):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                           f'{value:.1f}%', ha='center', va='bottom', fontsize=8)
        
        # Max Drawdown comparison
        bars = axes[1, 0].bar(model_names, max_drawdowns, color='salmon', alpha=0.8)
        axes[1, 0].set_title('Maximum Drawdown Comparison (Lower is Better)', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Max Drawdown (%)')
        axes[1, 0].tick_params(axis='x', rotation=90)
        axes[1, 0].grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels on bars
        for bar, value in zip(bars, max_drawdowns):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                           f'{value:.1f}%', ha='center', va='bottom', fontsize=8)
        
        # Sharpe Ratio comparison
        bars = axes[1, 1].bar(model_names, sharpe_ratios, color='purple', alpha=0.8)
        axes[1, 1].axhline(y=1, color='green', linestyle='--', alpha=0.7, label='Good (1.0)')
        axes[1, 1].axhline(y=2, color='orange', linestyle='--', alpha=0.7, label='Excellent (2.0)')
        axes[1, 1].set_title('Sharpe Ratio Comparison', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Sharpe Ratio')
        axes[1, 1].tick_params(axis='x', rotation=90)
        axes[1, 1].grid(axis='y', linestyle='--', alpha=0.7)
        axes[1, 1].legend()
        
        # Add value labels on bars
        for bar, value in zip(bars, sharpe_ratios):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                           f'{value:.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "evaluation_metrics_comprehensive.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Single vs Bagging Comparison
        if self.config['use_bagging']:
            self.create_single_vs_bagging_comparison(results)
        
        # 3. Market Condition Analysis
        self.create_market_condition_analysis(results)
        
        # 4. Risk-Return Scatter Plot
        self.create_risk_return_analysis(results)
        
        # 5. Performance Summary Table
        self.create_performance_table(results)
    
    def create_single_vs_bagging_comparison(self, results):
        """Create detailed single vs bagging comparison plots."""
        logger.info("Creating Single vs Bagging comparison visualizations")
        
        currencies = self.config['currency_pairs']
        model_types = self.config['models_to_train']
        
        fig, axes = plt.subplots(len(model_types), 4, figsize=(20, 5*len(model_types)))
        if len(model_types) == 1:
            axes = axes.reshape(1, -1)
        
        metrics = ['annual_return', 'win_rate', 'max_drawdown', 'sharpe_ratio']
        metric_titles = ['Annual Return (%)', 'Win Rate (%)', 'Max Drawdown (%)', 'Sharpe Ratio']
        
        for model_idx, model_type in enumerate(model_types):
            for metric_idx, (metric, title) in enumerate(zip(metrics, metric_titles)):
                single_values = []
                bagging_values = []
                pair_labels = []
                
                for pair in currencies:
                    single_key = f"{pair}_{model_type}"
                    bagging_key = f"Bagging_{model_type}_{pair}"
                    
                    if single_key in results and bagging_key in results:
                        if metric == 'win_rate':
                            single_val = results[single_key]['trading_performance'][metric] * 100
                            bagging_val = results[bagging_key]['trading_performance'][metric] * 100
                        else:
                            single_val = results[single_key]['trading_performance'][metric]
                            bagging_val = results[bagging_key]['trading_performance'][metric]
                        
                        single_values.append(single_val)
                        bagging_values.append(bagging_val)
                        pair_labels.append(pair)
                
                if single_values and bagging_values:
                    x = np.arange(len(pair_labels))
                    width = 0.35
                    
                    bars1 = axes[model_idx, metric_idx].bar(x - width/2, single_values, width, 
                                                           label='Single', alpha=0.8)
                    bars2 = axes[model_idx, metric_idx].bar(x + width/2, bagging_values, width, 
                                                           label='Bagging', alpha=0.8)
                    
                    axes[model_idx, metric_idx].set_title(f'{model_type} - {title}', 
                                                         fontsize=12, fontweight='bold')
                    axes[model_idx, metric_idx].set_ylabel(title)
                    axes[model_idx, metric_idx].set_xticks(x)
                    axes[model_idx, metric_idx].set_xticklabels(pair_labels)
                    axes[model_idx, metric_idx].legend()
                    axes[model_idx, metric_idx].grid(axis='y', linestyle='--', alpha=0.7)
                    
                    # Add improvement indicators
                    for i, (single, bagging) in enumerate(zip(single_values, bagging_values)):
                        improvement = bagging - single
                        color = 'green' if improvement > 0 else 'red'
                        y_pos = max(single, bagging) + (max(max(single_values), max(bagging_values)) * 0.05)
                        axes[model_idx, metric_idx].text(i, y_pos, f'{improvement:+.1f}', 
                                                        ha='center', va='bottom', color=color, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "single_vs_bagging_detailed_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_market_condition_analysis(self, results):
        """Analyze performance in different market conditions."""
        logger.info("Creating market condition analysis")
        
        # Categorize results by market condition
        up_market_results = {}
        down_market_results = {}
        
        for model_key, model_results in results.items():
            if 'trading_performance' in model_results:
                market_condition = model_results['trading_performance']['market_condition']
                if 'Up' in market_condition:
                    up_market_results[model_key] = model_results['trading_performance']
                else:
                    down_market_results[model_key] = model_results['trading_performance']
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        
        # Annual Return in different market conditions
        up_models = list(up_market_results.keys())
        down_models = list(down_market_results.keys())
        
        if up_models:
            up_returns = [up_market_results[model]['annual_return'] for model in up_models]
            axes[0, 0].bar(up_models, up_returns, color='green', alpha=0.7, label='Up Market')
            axes[0, 0].set_title('Annual Returns in Up Market', fontsize=12, fontweight='bold')
            axes[0, 0].tick_params(axis='x', rotation=90)
            axes[0, 0].set_ylabel('Annual Return (%)')
            axes[0, 0].grid(axis='y', alpha=0.3)
        
        if down_models:
            down_returns = [down_market_results[model]['annual_return'] for model in down_models]
            axes[0, 1].bar(down_models, down_returns, color='red', alpha=0.7, label='Down Market')
            axes[0, 1].set_title('Annual Returns in Down Market', fontsize=12, fontweight='bold')
            axes[0, 1].tick_params(axis='x', rotation=90)
            axes[0, 1].set_ylabel('Annual Return (%)')
            axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Win Rates in different market conditions
        if up_models:
            up_win_rates = [up_market_results[model]['win_rate'] * 100 for model in up_models]
            axes[1, 0].bar(up_models, up_win_rates, color='green', alpha=0.7)
            axes[1, 0].axhline(y=50, color='black', linestyle='--', alpha=0.7)
            axes[1, 0].set_title('Win Rates in Up Market', fontsize=12, fontweight='bold')
            axes[1, 0].tick_params(axis='x', rotation=90)
            axes[1, 0].set_ylabel('Win Rate (%)')
            axes[1, 0].grid(axis='y', alpha=0.3)
        
        if down_models:
            down_win_rates = [down_market_results[model]['win_rate'] * 100 for model in down_models]
            axes[1, 1].bar(down_models, down_win_rates, color='red', alpha=0.7)
            axes[1, 1].axhline(y=50, color='black', linestyle='--', alpha=0.7)
            axes[1, 1].set_title('Win Rates in Down Market', fontsize=12, fontweight='bold')
            axes[1, 1].tick_params(axis='x', rotation=90)
            axes[1, 1].set_ylabel('Win Rate (%)')
            axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "market_condition_analysis.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_risk_return_analysis(self, results):
        """Create risk-return scatter plot analysis."""
        logger.info("Creating risk-return analysis")
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Extract risk (max drawdown) and return data
        returns = []
        risks = []
        labels = []
        colors = []
        
        for model_key, model_results in results.items():
            if 'trading_performance' in model_results:
                perf = model_results['trading_performance']
                returns.append(perf['annual_return'])
                risks.append(perf['max_drawdown'])
                labels.append(model_key)
                
                # Color by model type
                if 'cnn_lstm' in model_key.lower():
                    colors.append('blue')
                elif 'tft' in model_key.lower():
                    colors.append('green')
                elif 'xgboost' in model_key.lower():
                    colors.append('red')
                else:
                    colors.append('purple')
        
        # Create scatter plot
        scatter = ax.scatter(risks, returns, c=colors, alpha=0.7, s=100)
        
        # Add labels for each point
        for i, label in enumerate(labels):
            ax.annotate(label, (risks[i], returns[i]), xytext=(5, 5), 
                       textcoords='offset points', fontsize=8, alpha=0.8)
        
        # Add quadrant lines
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax.axvline(x=np.median(risks), color='black', linestyle='--', alpha=0.5, 
                  label=f'Median Risk ({np.median(risks):.1f}%)')
        
        # Add ideal region (high return, low risk)
        ax.axhspan(0, max(returns), xmin=0, xmax=np.median(risks)/max(risks), 
                  alpha=0.1, color='green', label='Ideal Region')
        
        ax.set_xlabel('Risk (Max Drawdown %)', fontsize=12)
        ax.set_ylabel('Return (Annual Return %)', fontsize=12)
        ax.set_title('Risk-Return Analysis: All Models', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "risk_return_analysis.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_performance_table(self, results):
        """Create a comprehensive performance summary table."""
        logger.info("Creating performance summary table")
        
        # Prepare data for table
        table_data = []
        for model_key, model_results in results.items():
            if 'trading_performance' in model_results:
                perf = model_results['trading_performance']
                table_data.append([
                    model_key,
                    f"{perf['annual_return']:.2f}%",
                    f"{perf['win_rate']*100:.1f}%",
                    f"{perf['max_drawdown']:.2f}%",
                    f"{perf['sharpe_ratio']:.2f}",
                    f"{perf['trade_count']:d}",
                    perf['market_condition'],
                    f"{perf['buy_hold_annual_return']:.2f}%"
                ])
        
        # Sort by annual return (descending)
        table_data.sort(key=lambda x: float(x[1].replace('%', '')), reverse=True)
        
        # Create table plot
        fig, ax = plt.subplots(figsize=(16, max(8, len(table_data) * 0.5)))
        ax.axis('tight')
        ax.axis('off')
        
        columns = ['Model', 'Annual Return', 'Win Rate', 'Max Drawdown', 
                  'Sharpe Ratio', 'Trades', 'Market Condition', 'Buy & Hold Return']
        
        # Create table
        table = ax.table(cellText=table_data, colLabels=columns, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style the table
        for i in range(len(columns)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color code performance
        for i in range(1, len(table_data) + 1):
            annual_return = float(table_data[i-1][1].replace('%', ''))
            if annual_return > 10:
                table[(i, 1)].set_facecolor('#90EE90')  # Light green
            elif annual_return > 0:
                table[(i, 1)].set_facecolor('#FFFFE0')  # Light yellow
            else:
                table[(i, 1)].set_facecolor('#FFB6C1')  # Light red
        
        plt.title('Model Performance Summary Table\n(Sorted by Annual Return)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "performance_summary_table.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_feature_importance_visualizations(self):
        """Create feature importance visualizations for trained models."""
        logger.info("Creating feature importance visualizations")
        
        # Only XGBoost models have direct feature importance
        xgboost_models = {k: v for k, v in self.models.items() if 'xgboost' in k}
        
        if not xgboost_models:
            logger.warning("No XGBoost models found for feature importance analysis")
            return
        
        n_models = len(xgboost_models)
        fig, axes = plt.subplots(n_models, 1, figsize=(15, 6*n_models))
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_key, model_data) in enumerate(xgboost_models.items()):
            model = model_data['model']
            feature_names = model_data.get('feature_names', [])
            
            # Get feature importance
            importance = model.feature_importances_
            
            # Create feature importance DataFrame
            if feature_names:
                # If we have flattened features for XGBoost, we need to aggregate
                if len(feature_names) * self.config['window_size'] == len(importance):
                    # Aggregate importance by feature name across time steps
                    feature_importance = {}
                    for i, imp in enumerate(importance):
                        feature_idx = i % len(feature_names)
                        feature_name = feature_names[feature_idx]
                        if feature_name not in feature_importance:
                            feature_importance[feature_name] = 0
                        feature_importance[feature_name] += imp
                    
                    # Convert to DataFrame and sort
                    feature_df = pd.DataFrame(list(feature_importance.items()), 
                                            columns=['Feature', 'Importance'])
                else:
                    feature_df = pd.DataFrame({'Feature': feature_names[:len(importance)], 
                                             'Importance': importance})
            else:
                feature_df = pd.DataFrame({'Feature': [f'Feature_{i}' for i in range(len(importance))], 
                                         'Importance': importance})
            
            feature_df = feature_df.sort_values('Importance', ascending=False).head(20)
            
            # Plot
            bars = axes[idx].barh(range(len(feature_df)), feature_df['Importance'], 
                                 color='skyblue', alpha=0.8)
            axes[idx].set_yticks(range(len(feature_df)))
            axes[idx].set_yticklabels(feature_df['Feature'])
            axes[idx].set_xlabel('Importance')
            axes[idx].set_title(f'{model_key} - Top 20 Feature Importance', 
                               fontsize=12, fontweight='bold')
            axes[idx].grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, (bar, value) in enumerate(zip(bars, feature_df['Importance'])):
                axes[idx].text(value + 0.001, bar.get_y() + bar.get_height()/2, 
                              f'{value:.3f}', va='center', ha='left', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "feature_importance_analysis.png"), dpi=300, bbox_inches='tight')
        plt.close()

    #####################################
    # UPDATED MODEL BUILDING FUNCTIONS
    #####################################
    
    def build_cnn_lstm_model(self, input_shape, pair_or_bagging=None):
        """Build a CNN-LSTM hybrid model with hyperparameters loaded from JSON."""
        # Get hyperparameters from loaded JSON files
        if pair_or_bagging:
            hyperparams = self.get_hyperparameters('cnn_lstm', pair_or_bagging)
            logger.info(f"Using hyperparameters for {pair_or_bagging} CNN-LSTM: {hyperparams}")
        else:
            # Fallback to default hyperparameters
            hyperparams = {
                'cnn_filters': 64,
                'cnn_kernel_size': 3,
                'lstm_units': 100,
                'dropout_rate': 0.3,
                'learning_rate': 0.001
            }
            logger.info(f"Using default hyperparameters for CNN-LSTM: {hyperparams}")
        
        model = Sequential()
        
        # CNN layers
        model.add(Conv1D(filters=hyperparams['cnn_filters'], 
                        kernel_size=hyperparams['cnn_kernel_size'], 
                        activation='relu', 
                        input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(hyperparams['dropout_rate']))
        
        # Additional CNN layers
        model.add(Conv1D(filters=hyperparams['cnn_filters'] * 2, 
                        kernel_size=hyperparams['cnn_kernel_size'], 
                        activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(hyperparams['dropout_rate']))
        
        # LSTM layers
        model.add(LSTM(units=hyperparams['lstm_units'], return_sequences=True))
        model.add(Dropout(hyperparams['dropout_rate']))
        model.add(LSTM(units=hyperparams['lstm_units'] // 2))
        model.add(Dropout(hyperparams['dropout_rate']))
        
        # Output layer
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile the model with gradient clipping to prevent NaN losses
        optimizer = Adam(
            learning_rate=hyperparams['learning_rate'], 
            clipnorm=1.0  # Add gradient clipping
        )
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        
        return model
    
    def build_tft_model(self, input_dim, pair_or_bagging=None):
        """Build a simplified Temporal Fusion Transformer model with hyperparameters loaded from JSON."""
        # Get hyperparameters from loaded JSON files
        if pair_or_bagging:
            hyperparams = self.get_hyperparameters('tft', pair_or_bagging)
            logger.info(f"Using hyperparameters for {pair_or_bagging} TFT: {hyperparams}")
        else:
            # Fallback to default hyperparameters
            hyperparams = {
                'hidden_units': 64,
                'num_heads': 4,
                'dropout_rate': 0.2,
                'learning_rate': 0.001
            }
            logger.info(f"Using default hyperparameters for TFT: {hyperparams}")
        
        # Input
        inputs = Input(shape=(self.config['window_size'], input_dim))
        
        # Multi-head attention
        attention_output = MultiHeadAttention(
            num_heads=hyperparams['num_heads'], 
            key_dim=hyperparams['hidden_units'] // hyperparams['num_heads']
        )(inputs, inputs)
        
        # Residual connection and normalization
        x = tf.keras.layers.add([inputs, attention_output])
        x = BatchNormalization()(x)
        
        # Feed forward network
        ffn = Dense(hyperparams['hidden_units'], activation='relu', kernel_initializer='he_normal')(x)
        ffn = Dropout(hyperparams['dropout_rate'])(ffn)
        ffn = Dense(input_dim, kernel_initializer='he_normal')(ffn)
        
        # Residual connection and normalization
        x = tf.keras.layers.add([x, ffn])
        x = BatchNormalization()(x)
        
        # LSTM layers for sequence processing
        x = LSTM(units=hyperparams['hidden_units'], return_sequences=True)(x)
        x = Dropout(hyperparams['dropout_rate'])(x)
        x = LSTM(units=hyperparams['hidden_units'])(x)
        x = Dropout(hyperparams['dropout_rate'])(x)
        
        # Output
        outputs = Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform')(x)
        
        # Create and compile model with gradient clipping
        model = Model(inputs=inputs, outputs=outputs)
        optimizer = Adam(
            learning_rate=hyperparams['learning_rate'],
            clipnorm=1.0  # Add gradient clipping
        )
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        
        return model
    
    def build_xgboost_model(self, pair_or_bagging=None):
        """Build an XGBoost model with hyperparameters loaded from JSON."""
        # Get hyperparameters from loaded JSON files
        if pair_or_bagging:
            hyperparams = self.get_hyperparameters('xgboost', pair_or_bagging)
            logger.info(f"Using hyperparameters for {pair_or_bagging} XGBoost: {hyperparams}")
        else:
            # Fallback to default hyperparameters
            hyperparams = {
                'max_depth': 5,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 0,
                'reg_alpha': 0,
                'reg_lambda': 1,
                'objective': 'binary:logistic'  # FIXED: Added default objective
            }
            logger.info(f"Using default hyperparameters for XGBoost: {hyperparams}")
        
        # FIXED: Use .get() method to safely access 'objective' with fallback
        model = xgb.XGBClassifier(
            objective=hyperparams.get('objective', 'binary:logistic'),  # Safe access with fallback
            max_depth=hyperparams.get('max_depth', 5),
            learning_rate=hyperparams.get('learning_rate', 0.1),
            n_estimators=hyperparams.get('n_estimators', 100),
            subsample=hyperparams.get('subsample', 0.8),
            colsample_bytree=hyperparams.get('colsample_bytree', 0.8),
            gamma=hyperparams.get('gamma', 0),
            reg_alpha=hyperparams.get('reg_alpha', 0),
            reg_lambda=hyperparams.get('reg_lambda', 1),
            random_state=self.config['random_state']
        )
        
        return model
    
    def load_data(self, data_paths=None):
        """Load data from CSV files."""
        logger.info("Starting data loading process")
        
        if data_paths is None:
            # Default paths
            data_paths = {pair: f"{pair}_1H.csv" for pair in self.config['currency_pairs']}
        
        for pair, path in data_paths.items():
            try:
                # Load data
                df = pd.read_csv(path)
                
                # Convert time column to datetime
                df['Time'] = pd.to_datetime(df['Time'])
                
                # Set time as index
                df.set_index('Time', inplace=True)
                
                # Sort by time
                df.sort_index(inplace=True)
                
                self.data[pair] = df
                logger.info(f"Loaded {pair} data: {df.shape} rows")
                
            except Exception as e:
                logger.error(f"Error loading {pair} data from {path}: {e}")
                raise
        
        return self.data
    
    def preprocess_data(self):
        """Preprocess the loaded data."""
        logger.info("Starting data preprocessing")
        
        # Check if we have loaded data
        if not self.data:
            logger.error("No data loaded. Please run load_data() first.")
            return
        
        # Extract training and testing periods
        train_start = pd.to_datetime(self.config['train_start'])
        train_end = pd.to_datetime(self.config['train_end'])
        test_start = pd.to_datetime(self.config['test_start'])
        test_end = pd.to_datetime(self.config['test_end'])
        
        # Find common date range to ensure all pairs have the same rows
        common_dates = None
        for pair, df in self.data.items():
            date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='H')
            if common_dates is None:
                common_dates = set(date_range)
            else:
                common_dates = common_dates.intersection(set(date_range))
        
        common_dates = sorted(list(common_dates))
        logger.info(f"Common date range: {common_dates[0]} to {common_dates[-1]}, total: {len(common_dates)} hours")
        
        for pair, df in self.data.items():
            # Handle missing data
            df_reindexed = df.reindex(common_dates)
            
            # Fill missing values with the average of previous and next values
            df_reindexed.interpolate(method='linear', inplace=True)
            
            # Check if there are still any missing values at the start or end
            df_reindexed.fillna(method='ffill', inplace=True)  # Forward fill
            df_reindexed.fillna(method='bfill', inplace=True)  # Backward fill
            
            # Calculate returns and log returns
            df_reindexed['Returns'] = df_reindexed['Close'].pct_change()
            df_reindexed['Log_Returns'] = np.log(df_reindexed['Close'] / df_reindexed['Close'].shift(1))
            
            # Calculate target variable: price direction (1 if price goes up, 0 if down)
            df_reindexed['Target'] = (df_reindexed['Close'].shift(-self.config['prediction_horizon']) > df_reindexed['Close']).astype(int)
            
            # Extract training and testing sets
            train_data = df_reindexed[(df_reindexed.index >= train_start) & (df_reindexed.index <= train_end)]
            test_data = df_reindexed[(df_reindexed.index >= test_start) & (df_reindexed.index <= test_end)]
            
            # Store preprocessed data
            self.preprocessed_data[pair] = {
                'full': df_reindexed,
                'train': train_data,
                'test': test_data
            }
            
            logger.info(f"Preprocessed {pair} data: Full={df_reindexed.shape}, Train={train_data.shape}, Test={test_data.shape}")
            logger.info(f"Missing values in {pair} preprocessed data: {df_reindexed.isna().sum().sum()}")
        
        return self.preprocessed_data
    
    # Continue from the previous file...
# Adding the remaining methods to complete the implementation

    def calculate_technical_indicators(self):
        """Calculate technical indicators for each currency pair."""
        logger.info("Calculating technical indicators")
        
        for pair, data_dict in self.preprocessed_data.items():
            # Work with full dataset and then split later
            df = data_dict['full'].copy()
            
            # For readability, use price columns directly
            open_price = df['Open']
            high_price = df['High']
            low_price = df['Low']
            close_price = df['Close']
            volume = df['Volume']
            
            # Price features
            df['Price_Change'] = close_price - open_price
            df['Body_Size'] = abs(close_price - open_price)
            df['Upper_Shadow'] = high_price - np.maximum(close_price, open_price)
            df['Lower_Shadow'] = np.minimum(close_price, open_price) - low_price
            df['Range'] = high_price - low_price
            df['Mid_Point'] = (high_price + low_price) / 2
            df['HL_Ratio'] = high_price / (low_price + 1e-10)
            df['CO_Ratio'] = close_price / (open_price + 1e-10)
            
            # === Trend Indicators ===
            
            # Simple Moving Averages
            for window in [5, 10, 20, 50, 100]:
                df[f'SMA_{window}'] = close_price.rolling(window=window).mean()
                df[f'SMA_Distance_{window}'] = close_price / df[f'SMA_{window}'] - 1
            
            # Exponential Moving Averages
            for window in [5, 10, 20, 50, 100]:
                df[f'EMA_{window}'] = close_price.ewm(span=window, adjust=False).mean()
                df[f'EMA_Distance_{window}'] = close_price / df[f'EMA_{window}'] - 1
            
            # MACD
            df['EMA_12'] = close_price.ewm(span=12, adjust=False).mean()
            df['EMA_26'] = close_price.ewm(span=26, adjust=False).mean()
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
            
            # === Momentum Indicators ===
            
            # Relative Strength Index (RSI)
            delta = close_price.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-10)
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Stochastic Oscillator
            for k_period in [5, 14]:
                low_min = low_price.rolling(window=k_period).min()
                high_max = high_price.rolling(window=k_period).max()
                df[f'Stochastic_{k_period}'] = 100 * (close_price - low_min) / (high_max - low_min + 1e-10)
                df[f'Stochastic_Signal_{k_period}'] = df[f'Stochastic_{k_period}'].rolling(window=3).mean()
            
            # Rate of Change (ROC)
            for period in [5, 10, 20]:
                df[f'ROC_{period}'] = (close_price / close_price.shift(period) - 1) * 100
            
            # Momentum
            for period in [5, 10, 20]:
                df[f'Momentum_{period}'] = close_price / close_price.shift(period)
            
            # === Volatility Indicators ===
            
            # Bollinger Bands
            for period in [20]:
                df[f'BB_Middle_{period}'] = close_price.rolling(window=period).mean()
                df[f'BB_Std_{period}'] = close_price.rolling(window=period).std()
                df[f'BB_Upper_{period}'] = df[f'BB_Middle_{period}'] + 2 * df[f'BB_Std_{period}']
                df[f'BB_Lower_{period}'] = df[f'BB_Middle_{period}'] - 2 * df[f'BB_Std_{period}']
                df[f'BB_Width_{period}'] = (df[f'BB_Upper_{period}'] - df[f'BB_Lower_{period}']) / df[f'BB_Middle_{period}']
                df[f'BB_Position_{period}'] = (close_price - df[f'BB_Lower_{period}']) / (df[f'BB_Upper_{period}'] - df[f'BB_Lower_{period}'] + 1e-10)
            
            # Average True Range (ATR)
            tr1 = high_price - low_price
            tr2 = abs(high_price - close_price.shift(1))
            tr3 = abs(low_price - close_price.shift(1))
            true_range = pd.DataFrame({'TR1': tr1, 'TR2': tr2, 'TR3': tr3}).max(axis=1)
            df['ATR_14'] = true_range.rolling(window=14).mean()
            df['ATR_14_Ratio'] = df['ATR_14'] / close_price
            
            # Chaikin Volatility
            for period in [10]:
                df[f'Chaikin_Volatility_{period}'] = ((high_price - low_price).rolling(window=period).mean() / 
                                                      (high_price - low_price).rolling(window=period).mean().shift(period) - 1) * 100
            
            # === Volume Indicators ===
            
            # Volume Moving Average
            for window in [5, 10, 20]:
                df[f'Volume_SMA_{window}'] = volume.rolling(window=window).mean()
                df[f'Volume_Ratio_{window}'] = volume / df[f'Volume_SMA_{window}']
            
            # On-Balance Volume (OBV)
            df['OBV'] = (np.sign(close_price.diff()) * volume).fillna(0).cumsum()
            for window in [10, 20]:
                df[f'OBV_SMA_{window}'] = df['OBV'].rolling(window=window).mean()
                df[f'OBV_Ratio_{window}'] = df['OBV'] / df[f'OBV_SMA_{window}'] - 1
            
            # Chaikin Money Flow
            money_flow_multiplier = ((close_price - low_price) - (high_price - close_price)) / (high_price - low_price + 1e-10)
            money_flow_volume = money_flow_multiplier * volume
            df['CMF_20'] = money_flow_volume.rolling(window=20).sum() / volume.rolling(window=20).sum()
            
            # === Additional Features ===
            
            # Price Channels
            for period in [20, 50]:
                df[f'Upper_Channel_{period}'] = high_price.rolling(window=period).max()
                df[f'Lower_Channel_{period}'] = low_price.rolling(window=period).min()
                df[f'Channel_Width_{period}'] = (df[f'Upper_Channel_{period}'] - df[f'Lower_Channel_{period}']) / close_price
                df[f'Channel_Position_{period}'] = (close_price - df[f'Lower_Channel_{period}']) / (df[f'Upper_Channel_{period}'] - df[f'Lower_Channel_{period}'] + 1e-10)
            
            # Time-based features
            df['Hour'] = df.index.hour
            df['Day'] = df.index.day
            df['Month'] = df.index.month
            df['Day_of_Week'] = df.index.dayofweek
            df['Is_Weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)
            
            # Cyclical encoding of time features
            df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
            df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
            df['Day_Sin'] = np.sin(2 * np.pi * df['Day'] / 31)
            df['Day_Cos'] = np.cos(2 * np.pi * df['Day'] / 31)
            df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
            df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
            df['Day_of_Week_Sin'] = np.sin(2 * np.pi * df['Day_of_Week'] / 7)
            df['Day_of_Week_Cos'] = np.cos(2 * np.pi * df['Day_of_Week'] / 7)
            
            # Replace inf and NaN values
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.fillna(method='ffill', inplace=True)
            df.fillna(method='bfill', inplace=True)
            df.fillna(0, inplace=True)
            
            # Update train and test sets
            train_start = pd.to_datetime(self.config['train_start'])
            train_end = pd.to_datetime(self.config['train_end'])
            test_start = pd.to_datetime(self.config['test_start'])
            test_end = pd.to_datetime(self.config['test_end'])
            
            train_data = df[(df.index >= train_start) & (df.index <= train_end)]
            test_data = df[(df.index >= test_start) & (df.index <= test_end)]
            
            # Store enhanced features
            self.enhanced_features[pair] = {
                'full': df,
                'train': train_data,
                'test': test_data
            }
            
            logger.info(f"Enhanced features for {pair}: {df.shape[1]} features created")
        
        # Save feature CSV for analysis
        for pair, data_dict in self.enhanced_features.items():
            data_dict['full'].to_csv(os.path.join(feature_dir, f"{pair}_enhanced_features.csv"))
        
        return self.enhanced_features
    
    def select_features(self):
        """Select the most important features using multiple methods."""
        logger.info("Selecting important features")
        
        feature_importance_results = {}
        
        for pair, data_dict in self.enhanced_features.items():
            train_data = data_dict['train'].copy()
            
            # Remove non-feature columns and target
            features = train_data.drop(['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 'Log_Returns', 'Target'], axis=1, errors='ignore')
            target = train_data['Target']
            
            feature_rankings = {}
            selected_columns = {}
            
            # Method 1: Random Forest Importance
            logger.info(f"Feature selection for {pair} using Random Forest Importance")
            
            rf_model = RandomForestRegressor(n_estimators=100, random_state=self.config['random_state'])
            rf_model.fit(features, target)
            rf_importances = pd.Series(rf_model.feature_importances_, index=features.columns)
            rf_importances = rf_importances.sort_values(ascending=False)
            feature_rankings['random_forest'] = rf_importances
            selected_columns['random_forest'] = rf_importances.nlargest(self.config['n_features']).index.tolist()
            
            # Method 2: Mutual Information
            logger.info(f"Feature selection for {pair} using Mutual Information")
            
            mi_values = mutual_info_regression(features, target, random_state=self.config['random_state'])
            mi_importances = pd.Series(mi_values, index=features.columns)
            mi_importances = mi_importances.sort_values(ascending=False)
            feature_rankings['mutual_info'] = mi_importances
            selected_columns['mutual_info'] = mi_importances.nlargest(self.config['n_features']).index.tolist()
            
            # Method 3: Principal Component Analysis (PCA)
            logger.info(f"Feature selection for {pair} using PCA")
            
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            
            pca = PCA(n_components=min(self.config['n_features'], len(features.columns)))
            pca.fit(scaled_features)
            
            # Calculate feature importance based on loadings and explained variance
            loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
            pca_importance = np.sum(loadings**2, axis=1)
            pca_importances = pd.Series(pca_importance, index=features.columns)
            pca_importances = pca_importances.sort_values(ascending=False)
            feature_rankings['pca'] = pca_importances
            selected_columns['pca'] = pca_importances.nlargest(self.config['n_features']).index.tolist()
            
            # Combine the selected features from all methods
            all_selected_features = set()
            for method, selected_features in selected_columns.items():
                all_selected_features.update(selected_features)
            
            # Log the feature selection results
            logger.info(f"Selected features for {pair}: {len(all_selected_features)} unique features selected")
            
            # Create DataFrames with selected features
            self.selected_features[pair] = {
                'full': self.enhanced_features[pair]['full'][list(all_selected_features) + ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 'Log_Returns', 'Target']],
                'train': self.enhanced_features[pair]['train'][list(all_selected_features) + ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 'Log_Returns', 'Target']],
                'test': self.enhanced_features[pair]['test'][list(all_selected_features) + ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 'Log_Returns', 'Target']]
            }
            
            # Store feature importance results
            feature_importance_results[pair] = {
                'feature_rankings': feature_rankings,
                'selected_columns': selected_columns,
                'all_selected_features': list(all_selected_features)
            }
            
            # Plot feature importance for each method
            for method, importances in feature_rankings.items():
                plt.figure(figsize=(12, 8))
                importances.nlargest(20).plot(kind='barh')
                plt.title(f'Top 20 Feature Importance for {pair} using {method}')
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f"{pair}_{method}_feature_importance.png"))
                plt.close()
        
        # Save feature importance results
        with open(os.path.join(feature_dir, 'feature_importance_results.json'), 'w') as f:
            json.dump({
                pair: {
                    'selected_columns': {method: list(cols) for method, cols in data['selected_columns'].items()},
                    'all_selected_features': list(data['all_selected_features'])
                } 
                for pair, data in feature_importance_results.items()
            }, f, indent=2)
        
        # Create bagging dataset by combining all pairs
        self._create_bagging_dataset()
        
        return self.selected_features
    
    def _create_bagging_dataset(self):
        """Create a combined dataset for the bagging approach."""
        logger.info("Creating bagging dataset")
        
        # Combine the selected features from all currency pairs
        combined_train_dfs = []
        combined_test_dfs = []
        
        for pair, data_dict in self.selected_features.items():
            # Add a column to identify the currency pair
            train_df = data_dict['train'].copy()
            train_df['CurrencyPair'] = pair
            
            test_df = data_dict['test'].copy()
            test_df['CurrencyPair'] = pair
            
            combined_train_dfs.append(train_df)
            combined_test_dfs.append(test_df)
        
        # Concatenate all pairs
        bagging_train = pd.concat(combined_train_dfs, axis=0)
        bagging_test = pd.concat(combined_test_dfs, axis=0)
        
        # Sort by date
        bagging_train.sort_index(inplace=True)
        bagging_test.sort_index(inplace=True)
        
        self.bagging_data = {
            'train': bagging_train,
            'test': bagging_test
        }
        
        logger.info(f"Bagging dataset created: Train={bagging_train.shape}, Test={bagging_test.shape}")
        
        return self.bagging_data
    
    def prepare_model_data(self, data, is_lstm=True):
        """Prepare data for model training."""
        # Extract features and target
        X = data.drop(['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 'Log_Returns', 'Target', 'CurrencyPair'], axis=1, errors='ignore')
        y = data['Target']
        
        # Scale the features
        if self.config['scaler_type'] == 'standard':
            scaler = StandardScaler()
        elif self.config['scaler_type'] == 'minmax':
            scaler = MinMaxScaler()
        elif self.config['scaler_type'] == 'robust':
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
        
        X_scaled = scaler.fit_transform(X)
        
        if is_lstm:
            # Create sequences for LSTM models
            X_seq, y_seq = [], []
            for i in range(len(X_scaled) - self.config['window_size']):
                X_seq.append(X_scaled[i:i + self.config['window_size']])
                y_seq.append(y.iloc[i + self.config['window_size']])
            
            return np.array(X_seq), np.array(y_seq), scaler, X.columns.tolist()
        else:
            # For non-LSTM models, we need to include lagged features
            X_lagged = np.zeros((X_scaled.shape[0] - self.config['window_size'], X_scaled.shape[1] * self.config['window_size']))
            y_lagged = y.iloc[self.config['window_size']:].values
            
            for i in range(len(X_scaled) - self.config['window_size']):
                X_lagged[i] = X_scaled[i:i + self.config['window_size']].flatten()
            
            return X_lagged, y_lagged, scaler, X.columns.tolist()
    
    def train_models(self):
        """Train all models on all currency pairs with enhanced visualization."""
        logger.info("Starting model training")
        
        # Create data visualizations before training
        self.create_data_visualizations()
        
        # Define callbacks for deep learning models
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=self.config['patience'], restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        # Train Single-Pair Models
        for pair in self.config['currency_pairs']:
            logger.info(f"Training models for {pair}")
            
            train_data = self.selected_features[pair]['train']
            test_data = self.selected_features[pair]['test']
            
            # Train CNN-LSTM model
            if 'cnn_lstm' in self.config['models_to_train']:
                logger.info(f"Training CNN-LSTM model for {pair}")
                
                # Prepare data for LSTM
                X_train, y_train, scaler, feature_names = self.prepare_model_data(train_data, is_lstm=True)
                
                # Check for NaN values
                if np.isnan(X_train).any() or np.isnan(y_train).any():
                    logger.warning(f"NaN values detected in {pair} CNN-LSTM training data. Fixing...")
                    X_train = np.nan_to_num(X_train)
                    y_train = np.nan_to_num(y_train)
                
                # Split training and validation sets
                val_size = int(len(X_train) * self.config['validation_size'])
                X_val, y_val = X_train[-val_size:], y_train[-val_size:]
                X_train, y_train = X_train[:-val_size], y_train[:-val_size]
                
                input_shape = (X_train.shape[1], X_train.shape[2])
                
                # Initialize model with pair-specific hyperparameters
                model = self.build_cnn_lstm_model(input_shape, pair)
                
                # Create model checkpoint callback
                checkpoint_path = os.path.join(models_dir, f"{pair}_cnn_lstm_best.keras")
                model_checkpoint = ModelCheckpoint(
                    checkpoint_path,
                    save_best_only=True,
                    monitor='val_loss',
                    mode='min',
                    verbose=1
                )
                
                # Train the model
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=self.config['epochs'],
                    batch_size=self.config['batch_size'],
                    callbacks=callbacks + [model_checkpoint],
                    verbose=1
                )
                
                # Save the final model if checkpoint didn't work
                if not os.path.exists(checkpoint_path):
                    logger.info(f"Saving final model for {pair} CNN-LSTM")
                    model.save(checkpoint_path)
                
                # Store the model and scaler
                self.models[f"{pair}_cnn_lstm"] = {
                    'model': model,
                    'scaler': scaler,
                    'feature_names': feature_names,
                    'history': history.history
                }
                
                # Create training visualizations
                self.create_training_visualizations(f"{pair}_cnn_lstm", {'history': history.history})
                
                logger.info(f"CNN-LSTM model for {pair} trained successfully")
            
            # Train TFT model
            if 'tft' in self.config['models_to_train']:
                logger.info(f"Training TFT model for {pair}")
                
                # Prepare data for TFT
                X_train, y_train, scaler, feature_names = self.prepare_model_data(train_data, is_lstm=True)
                
                # Check for NaN values
                if np.isnan(X_train).any() or np.isnan(y_train).any():
                    logger.warning(f"NaN values detected in {pair} TFT training data. Fixing...")
                    X_train = np.nan_to_num(X_train)
                    y_train = np.nan_to_num(y_train)
                
                # Split training and validation sets
                val_size = int(len(X_train) * self.config['validation_size'])
                X_val, y_val = X_train[-val_size:], y_train[-val_size:]
                X_train, y_train = X_train[:-val_size], y_train[:-val_size]
                
                input_dim = X_train.shape[2]
                
                # Initialize model with pair-specific hyperparameters
                model = self.build_tft_model(input_dim, pair)
                
                # Create model checkpoint callback
                checkpoint_path = os.path.join(models_dir, f"{pair}_tft_best.keras")
                model_checkpoint = ModelCheckpoint(
                    checkpoint_path,
                    save_best_only=True,
                    monitor='val_loss',
                    mode='min',
                    verbose=1
                )
                
                # Train the model
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=self.config['epochs'],
                    batch_size=self.config['batch_size'],
                    callbacks=callbacks + [model_checkpoint],
                    verbose=1
                )
                
                # Save the final model if checkpoint didn't work
                if not os.path.exists(checkpoint_path):
                    logger.info(f"Saving final model for {pair} TFT")
                    model.save(checkpoint_path)
                
                # Store the model and scaler
                self.models[f"{pair}_tft"] = {
                    'model': model,
                    'scaler': scaler,
                    'feature_names': feature_names,
                    'history': history.history
                }
                
                # Create training visualizations
                self.create_training_visualizations(f"{pair}_tft", {'history': history.history})
                
                logger.info(f"TFT model for {pair} trained successfully")
            
            # Train XGBoost model
            if 'xgboost' in self.config['models_to_train']:
                logger.info(f"Training XGBoost model for {pair}")
                
                # Prepare data for XGBoost
                X_train, y_train, scaler, feature_names = self.prepare_model_data(train_data, is_lstm=False)
                
                # Check for NaN values
                if np.isnan(X_train).any() or np.isnan(y_train).any():
                    logger.warning(f"NaN values detected in {pair} XGBoost training data. Fixing...")
                    X_train = np.nan_to_num(X_train)
                    y_train = np.nan_to_num(y_train)
                
                # Split training and validation sets
                val_size = int(len(X_train) * self.config['validation_size'])
                X_val, y_val = X_train[-val_size:], y_train[-val_size:]
                X_train, y_train = X_train[:-val_size], y_train[:-val_size]
                
                # Initialize model with pair-specific hyperparameters
                model = self.build_xgboost_model(pair)
                
                # Train the model
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=1
                )
                
                # Save the model
                model_path = os.path.join(models_dir, f"{pair}_xgboost.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                
                # Store the model and scaler
                self.models[f"{pair}_xgboost"] = {
                    'model': model,
                    'scaler': scaler,
                    'feature_names': feature_names
                }
                
                logger.info(f"XGBoost model for {pair} trained successfully")
        
        # Train Bagging Models
        if self.config['use_bagging']:
            logger.info("Training bagging models (combined currency pairs)")
            
            train_data = self.bagging_data['train']
            
            # Train CNN-LSTM model
            if 'cnn_lstm' in self.config['models_to_train']:
                logger.info("Training Bagging CNN-LSTM model")
                
                # Prepare data for LSTM
                X_train, y_train, scaler, feature_names = self.prepare_model_data(train_data, is_lstm=True)
                
                # Check for NaN values
                if np.isnan(X_train).any() or np.isnan(y_train).any():
                    logger.warning("NaN values detected in Bagging CNN-LSTM training data. Fixing...")
                    X_train = np.nan_to_num(X_train)
                    y_train = np.nan_to_num(y_train)
                
                # Split training and validation sets
                val_size = int(len(X_train) * self.config['validation_size'])
                X_val, y_val = X_train[-val_size:], y_train[-val_size:]
                X_train, y_train = X_train[:-val_size], y_train[:-val_size]
                
                input_shape = (X_train.shape[1], X_train.shape[2])
                
                # Initialize model with bagging-specific hyperparameters
                model = self.build_cnn_lstm_model(input_shape, 'Bagging')
                
                # Create model checkpoint callback
                checkpoint_path = os.path.join(models_dir, "Bagging_cnn_lstm_best.keras")
                model_checkpoint = ModelCheckpoint(
                    checkpoint_path,
                    save_best_only=True,
                    monitor='val_loss',
                    mode='min',
                    verbose=1
                )
                
                # Train the model
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=self.config['epochs'],
                    batch_size=self.config['batch_size'],
                    callbacks=callbacks + [model_checkpoint],
                    verbose=1
                )
                
                # Save the final model if checkpoint didn't work
                if not os.path.exists(checkpoint_path):
                    logger.info("Saving final Bagging CNN-LSTM model")
                    model.save(checkpoint_path)
                
                # Store the model and scaler
                self.models["Bagging_cnn_lstm"] = {
                    'model': model,
                    'scaler': scaler,
                    'feature_names': feature_names,
                    'history': history.history
                }
                
                # Create training visualizations
                self.create_training_visualizations("Bagging_cnn_lstm", {'history': history.history})
                
                logger.info("Bagging CNN-LSTM model trained successfully")
            
            # Train TFT model
            if 'tft' in self.config['models_to_train']:
                logger.info("Training Bagging TFT model")
                
                # Prepare data for TFT
                X_train, y_train, scaler, feature_names = self.prepare_model_data(train_data, is_lstm=True)
                
                # Check for NaN values
                if np.isnan(X_train).any() or np.isnan(y_train).any():
                    logger.warning("NaN values detected in Bagging TFT training data. Fixing...")
                    X_train = np.nan_to_num(X_train)
                    y_train = np.nan_to_num(y_train)
                
                # Split training and validation sets
                val_size = int(len(X_train) * self.config['validation_size'])
                X_val, y_val = X_train[-val_size:], y_train[-val_size:]
                X_train, y_train = X_train[:-val_size], y_train[:-val_size]
                
                input_dim = X_train.shape[2]
                
                # Initialize model with bagging-specific hyperparameters
                model = self.build_tft_model(input_dim, 'Bagging')
                
                # Create model checkpoint callback
                checkpoint_path = os.path.join(models_dir, "Bagging_tft_best.keras")
                model_checkpoint = ModelCheckpoint(
                    checkpoint_path,
                    save_best_only=True,
                    monitor='val_loss',
                    mode='min',
                    verbose=1
                )
                
                # Train the model
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=self.config['epochs'],
                    batch_size=self.config['batch_size'],
                    callbacks=callbacks + [model_checkpoint],
                    verbose=1
                )
                
                # Save the final model regardless of checkpoint
                logger.info("Saving final Bagging TFT model")
                model.save(checkpoint_path)
                
                # Store the model and scaler
                self.models["Bagging_tft"] = {
                    'model': model,
                    'scaler': scaler,
                    'feature_names': feature_names,
                    'history': history.history
                }
                
                # Create training visualizations
                self.create_training_visualizations("Bagging_tft", {'history': history.history})
                
                logger.info("Bagging TFT model trained successfully")
            
            # Train XGBoost model
            if 'xgboost' in self.config['models_to_train']:
                logger.info("Training Bagging XGBoost model")
                
                # Prepare data for XGBoost
                X_train, y_train, scaler, feature_names = self.prepare_model_data(train_data, is_lstm=False)
                
                # Check for NaN values
                if np.isnan(X_train).any() or np.isnan(y_train).any():
                    logger.warning("NaN values detected in Bagging XGBoost training data. Fixing...")
                    X_train = np.nan_to_num(X_train)
                    y_train = np.nan_to_num(y_train)
                
                # Split training and validation sets
                val_size = int(len(X_train) * self.config['validation_size'])
                X_val, y_val = X_train[-val_size:], y_train[-val_size:]
                X_train, y_train = X_train[:-val_size], y_train[:-val_size]
                
                # Initialize model with bagging-specific hyperparameters
                model = self.build_xgboost_model('Bagging')
                
                # Train the model
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=1
                )
                
                # Save the model
                model_path = os.path.join(models_dir, "Bagging_xgboost.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                
                # Store the model and scaler
                self.models["Bagging_xgboost"] = {
                    'model': model,
                    'scaler': scaler,
                    'feature_names': feature_names
                }
                
                logger.info("Bagging XGBoost model trained successfully")
        
        return self.models
    
    def evaluate_models(self):
        """Evaluate all trained models on test data with comprehensive visualization."""
        logger.info("Starting model evaluation")
        
        results = {}
        
        # Evaluate Single-Pair Models
        for pair in self.config['currency_pairs']:
            test_data = self.selected_features[pair]['test']
            
            # Evaluate each model type
            for model_type in self.config['models_to_train']:
                model_key = f"{pair}_{model_type}"
                if model_key not in self.models:
                    logger.warning(f"Model {model_key} not found in trained models")
                    continue
                
                model_data = self.models[model_key]
                model = model_data['model']
                scaler = model_data['scaler']
                feature_names = model_data['feature_names']
                
                # Generate predictions
                if model_type in ['cnn_lstm', 'tft']:
                    # Prepare data for LSTM models
                    X_test, y_test, _, _ = self.prepare_model_data(test_data, is_lstm=True)
                    
                    # Check and handle NaN values
                    if np.isnan(X_test).any() or np.isnan(y_test).any():
                        logger.warning(f"NaN values detected in {model_key} test data. Fixing...")
                        X_test = np.nan_to_num(X_test)
                        y_test = np.nan_to_num(y_test)
                    
                    # Get predictions
                    y_pred_proba = model.predict(X_test)
                    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
                else:  # XGBoost
                    # Prepare data for XGBoost
                    X_test, y_test, _, _ = self.prepare_model_data(test_data, is_lstm=False)
                    
                    # Check and handle NaN values
                    if np.isnan(X_test).any() or np.isnan(y_test).any():
                        logger.warning(f"NaN values detected in {model_key} test data. Fixing...")
                        X_test = np.nan_to_num(X_test)
                        y_test = np.nan_to_num(y_test)
                    
                    # Get predictions
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    y_pred = (y_pred_proba > 0.5).astype(int)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                
                logger.info(f"Model {model_key} test metrics: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
                
                # Create prediction visualizations
                self.create_prediction_visualizations(model_key, y_test, y_pred, y_pred_proba, test_data)
                
                # Generate trading signals and evaluate performance
                trading_performance = self.evaluate_trading_performance(pair, model_type, y_test, y_pred, test_data)
                
                # Store results
                results[model_key] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'trading_performance': trading_performance
                }
        
        # Evaluate Bagging Models (similar structure as original, but with visualization calls)
        if self.config['use_bagging']:
            bagging_test_data = self.bagging_data['test']
            
            # Extract data for each currency pair
            for pair in self.config['currency_pairs']:
                # Filter bagging test data for this pair
                pair_test_data = bagging_test_data[bagging_test_data['CurrencyPair'] == pair].copy()
                
                if pair_test_data.empty:
                    logger.warning(f"No test data available for {pair} in the bagging dataset")
                    continue
                
                # Evaluate each model type
                for model_type in self.config['models_to_train']:
                    model_key = f"Bagging_{model_type}"
                    if model_key not in self.models:
                        logger.warning(f"Model {model_key} not found in trained models")
                        continue
                    
                    model_data = self.models[model_key]
                    model = model_data['model']
                    scaler = model_data['scaler']
                    feature_names = model_data['feature_names']
                    
                    # Generate predictions
                    if model_type in ['cnn_lstm', 'tft']:
                        # Prepare data for LSTM models
                        X_test, y_test, _, _ = self.prepare_model_data(pair_test_data, is_lstm=True)
                        
                        # Check and handle NaN values
                        if np.isnan(X_test).any() or np.isnan(y_test).any():
                            logger.warning(f"NaN values detected in {model_key} test data for {pair}. Fixing...")
                            X_test = np.nan_to_num(X_test)
                            y_test = np.nan_to_num(y_test)
                        
                        # Log shapes for debugging
                        logger.info(f"Bagging {model_type} evaluation for {pair}: X_test shape = {X_test.shape}")
                        
                        # Get predictions
                        try:
                            y_pred_proba = model.predict(X_test)
                            y_pred = (y_pred_proba > 0.5).astype(int).flatten()
                        except Exception as e:
                            logger.error(f"Error making predictions with {model_key} on {pair}: {e}")
                            continue
                            
                    else:  # XGBoost
                        # Prepare data for XGBoost
                        X_test, y_test, _, _ = self.prepare_model_data(pair_test_data, is_lstm=False)
                        
                        # Check and handle NaN values
                        if np.isnan(X_test).any() or np.isnan(y_test).any():
                            logger.warning(f"NaN values detected in {model_key} test data for {pair}. Fixing...")
                            X_test = np.nan_to_num(X_test)
                            y_test = np.nan_to_num(y_test)
                        
                        # Get predictions
                        try:
                            y_pred_proba = model.predict_proba(X_test)[:, 1]
                            y_pred = (y_pred_proba > 0.5).astype(int)
                        except Exception as e:
                            logger.error(f"Error making predictions with {model_key} on {pair}: {e}")
                            continue
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred)
                    recall = recall_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)
                    
                    logger.info(f"Model {model_key} on {pair} test metrics: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
                    
                    # Create prediction visualizations
                    self.create_prediction_visualizations(f"{model_key}_{pair}", y_test, y_pred, y_pred_proba, pair_test_data)
                    
                    # Generate trading signals and evaluate performance
                    trading_performance = self.evaluate_trading_performance(pair, f"Bagging_{model_type}", y_test, y_pred, pair_test_data)
                    
                    # Store results
                    results[f"{model_key}_{pair}"] = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'trading_performance': trading_performance
                    }
        
        # Save results to file
        with open(os.path.join(results_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create comprehensive evaluation visualizations
        self.create_evaluation_visualizations(results)
        
        # Create feature importance visualizations
        self.create_feature_importance_visualizations()
        
        self.results = results
        return results
    
    def evaluate_trading_performance(self, pair, model_type, y_true, y_pred, test_data):
        """Evaluate trading performance based on predictions."""
        logger.info(f"Evaluating trading performance for {pair} using {model_type}")
        
        # Get price data
        close_prices = test_data['Close'].values
        
        # Align close prices with predictions
        if len(close_prices) > len(y_pred) + self.config['window_size']:
            close_prices = close_prices[self.config['window_size']:len(y_pred) + self.config['window_size']]
        else:
            # Adjust if we have fewer prices than expected
            close_prices = close_prices[-len(y_pred):]
        
        # Ensure we have the same number of prices as predictions
        assert len(close_prices) == len(y_pred), f"Close prices length ({len(close_prices)}) != predictions length ({len(y_pred)})"
        
        # Initialize performance metrics
        initial_balance = 10000  # Initial investment amount
        balance = initial_balance
        trades = []
        position = 0  # 0: no position, 1: long position
        entry_price = 0
        trade_count = 0
        winning_trades = 0
        
        # Simulate trading
        for i in range(len(y_pred)):
            signal = y_pred[i]
            price = close_prices[i]
            
            # Enter long position
            if signal == 1 and position == 0:
                position = 1
                entry_price = price
                trades.append({'type': 'buy', 'price': price, 'balance': balance})
            
            # Exit long position
            elif signal == 0 and position == 1:
                position = 0
                exit_price = price
                profit_pct = (exit_price / entry_price - 1) * 100
                balance *= (1 + profit_pct / 100)
                
                trade_count += 1
                if exit_price > entry_price:
                    winning_trades += 1
                
                trades.append({'type': 'sell', 'price': price, 'balance': balance, 'profit_pct': profit_pct})
        
        # Close any open position at the end
        if position == 1:
            exit_price = close_prices[-1]
            profit_pct = (exit_price / entry_price - 1) * 100
            balance *= (1 + profit_pct / 100)
            
            trade_count += 1
            if exit_price > entry_price:
                winning_trades += 1
            
            trades.append({'type': 'sell', 'price': exit_price, 'balance': balance, 'profit_pct': profit_pct})
        
        # Calculate performance metrics
        win_rate = winning_trades / trade_count if trade_count > 0 else 0
        total_return = (balance / initial_balance - 1) * 100
        
        # Calculate annualized return
        # Assuming 252 trading days per year
        n_days = len(y_pred) / 24  # Convert hours to days
        annual_return = ((1 + total_return / 100) ** (252 / n_days) - 1) * 100
        
        # Calculate Buy & Hold return
        buy_hold_return = (close_prices[-1] / close_prices[0] - 1) * 100
        
        # Calculate Buy & Hold annualized return
        buy_hold_annual_return = ((1 + buy_hold_return / 100) ** (252 / n_days) - 1) * 100
        
        # Determine market conditions
        up_market = close_prices[-1] > close_prices[0]
        market_condition = "Up Market" if up_market else "Down Market"
        
        # Calculate average profit per trade
        if trade_count > 0:
            avg_profit_per_trade = total_return / trade_count
        else:
            avg_profit_per_trade = 0
        
        # Calculate maximum drawdown
        cumulative_returns = np.array([trade.get('balance', initial_balance) / initial_balance for trade in trades])
        if len(cumulative_returns) > 0:
            peak = np.maximum.accumulate(cumulative_returns)
            drawdown = (peak - cumulative_returns) / peak
            max_drawdown = drawdown.max() * 100
        else:
            max_drawdown = 0
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0)
        if len(trades) > 1:
            returns = np.diff([trade.get('balance', initial_balance) for trade in trades]) / np.array([trade.get('balance', initial_balance) for trade in trades[:-1]])
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            # Annualize Sharpe ratio
            sharpe_ratio *= np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Create performance plot
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(close_prices)
        plt.title(f'{pair} {model_type} - Price Chart')
        plt.xlabel('Time')
        plt.ylabel('Price')
        
        plt.subplot(2, 1, 2)
        balance_history = [initial_balance]
        for trade in trades:
            if 'balance' in trade:
                balance_history.append(trade['balance'])
        plt.plot(balance_history)
        plt.title(f'{pair} {model_type} - Account Balance')
        plt.xlabel('Trades')
        plt.ylabel('Balance')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{pair}_{model_type}_trading_performance.png"))
        plt.close()
        
        # Store performance metrics
        performance = {
            'annual_return': annual_return,
            'total_return': total_return,
            'win_rate': win_rate,
            'trade_count': trade_count,
            'avg_profit_per_trade': avg_profit_per_trade,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'market_condition': market_condition,
            'buy_hold_return': buy_hold_return,
            'buy_hold_annual_return': buy_hold_annual_return
        }
        
        logger.info(f"Trading performance for {pair} {model_type}: Annual Return={annual_return:.2f}%, Win Rate={win_rate:.2f}, Max Drawdown={max_drawdown:.2f}%")
        
        return performance
    
    def run_pipeline(self):
        """Run the complete enhanced pipeline with comprehensive visualization."""
        start_time = time.time()
        logger.info("Starting the Enhanced Forex Prediction pipeline")
        
        # Step 1: Data Collection & Preprocessing
        logger.info("STEP 1: Data Collection & Preprocessing")
        self.load_data()
        self.preprocess_data()
        
        # Step 2: Feature Enhancement
        logger.info("STEP 2: Feature Enhancement")
        self.calculate_technical_indicators()
        self.select_features()
        
        # Step 3: Model Development
        logger.info("STEP 3: Model Development")
        self.train_models()
        
        # Step 4: Model Evaluation & Performance Analysis
        logger.info("STEP 4: Model Evaluation & Performance Analysis")
        results = self.evaluate_models()
        
        elapsed_time = time.time() - start_time
        logger.info(f"Enhanced pipeline completed in {elapsed_time / 60:.2f} minutes")
        
        return results
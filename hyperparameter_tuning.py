"""
Enhanced Hyperparameter Tuning for Master's Thesis
Version: 2.0 - Advanced Optimization with Bayesian and Multi-Objective Methods
Author: Master's Thesis Student
Date: 2024

This module provides comprehensive hyperparameter optimization including:
- Bayesian Optimization with multiple acquisition functions
- Multi-objective optimization (return vs risk)
- Parallel optimization support
- Advanced pruning strategies
- Comprehensive visualization and reporting
"""

import os
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import multiprocessing as mp
from functools import partial
from dataclasses import dataclass
import pickle
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Optimization libraries
import optuna
from optuna.samplers import (TPESampler, CmaEsSampler, NSGAIISampler, 
                           RandomSampler, GridSampler)
from optuna.pruners import (MedianPruner, SuccessiveHalvingPruner, 
                          HyperbandPruner, ThresholdPruner)
from optuna.visualization import (plot_optimization_history, plot_param_importances,
                                plot_parallel_coordinate, plot_slice, plot_contour,
                                plot_edf, plot_intermediate_values, plot_pareto_front)

from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, STATUS_FAIL
from hyperopt.pyll import scope

import skopt
from skopt import gp_minimize, forest_minimize, gbrt_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt.plots import plot_convergence, plot_objective, plot_evaluations

# Machine Learning
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import make_scorer
import xgboost as xgb
import lightgbm as lgb

# Deep Learning
import tensorflow as tf
from tensorflow.keras import backend as K

# Import main forex prediction module
from forex_prediction import (
    TradingConfig, EnhancedForexPredictor, AdvancedFeatureEngineer,
    AdvancedCNNLSTM, AdvancedTransformer, AdvancedTradingStrategy
)

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for hyperparameter optimization"""
    method: str = 'bayesian'  # 'bayesian', 'multi_objective', 'grid', 'random'
    n_trials: int = 100
    n_jobs: int = 1
    timeout: int = 3600  # 1 hour
    
    # Bayesian optimization
    sampler: str = 'tpe'  # 'tpe', 'cmaes', 'random', 'grid'
    pruner: str = 'median'  # 'median', 'hyperband', 'successive_halving', 'threshold'
    
    # Multi-objective
    objectives: List[str] = None  # ['return', 'risk', 'sharpe']
    
    # Early stopping
    early_stopping_rounds: int = 20
    min_improvement: float = 0.001
    
    # Cross-validation
    cv_folds: int = 3
    cv_metric: str = 'sharpe_ratio'
    
    # Parallel optimization
    use_parallel: bool = True
    parallel_backend: str = 'multiprocessing'  # 'multiprocessing', 'threading'
    
    def __post_init__(self):
        if self.objectives is None:
            self.objectives = ['return', 'risk']


class HyperparameterSpace:
    """Define hyperparameter search spaces for different models"""
    
    @staticmethod
    def get_cnn_lstm_space(trial: optuna.Trial = None, method: str = 'optuna') -> Dict:
        """Get CNN-LSTM hyperparameter space"""
        if method == 'optuna':
            return {
                # CNN parameters
                'cnn_filters_1': trial.suggest_int('cnn_filters_1', 32, 256, step=32),
                'cnn_filters_2': trial.suggest_int('cnn_filters_2', 64, 512, step=64),
                'cnn_kernel_size': trial.suggest_int('cnn_kernel_size', 2, 5),
                'cnn_activation': trial.suggest_categorical('cnn_activation', ['relu', 'elu', 'tanh']),
                
                # LSTM parameters
                'lstm_units_1': trial.suggest_int('lstm_units_1', 64, 256, step=32),
                'lstm_units_2': trial.suggest_int('lstm_units_2', 32, 128, step=16),
                'lstm_dropout': trial.suggest_float('lstm_dropout', 0.1, 0.5),
                'use_bidirectional': trial.suggest_categorical('use_bidirectional', [True, False]),
                
                # Attention parameters
                'use_attention': trial.suggest_categorical('use_attention', [True, False]),
                'attention_heads': trial.suggest_int('attention_heads', 4, 16, step=4),
                
                # Training parameters
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
                'optimizer': trial.suggest_categorical('optimizer', ['adam', 'sgd', 'rmsprop']),
                
                # Regularization
                'l1_reg': trial.suggest_float('l1_reg', 1e-6, 1e-3, log=True),
                'l2_reg': trial.suggest_float('l2_reg', 1e-6, 1e-3, log=True),
                'dropout_rate': trial.suggest_float('dropout_rate', 0.2, 0.6),
            }
        
        elif method == 'hyperopt':
            return {
                'cnn_filters_1': hp.choice('cnn_filters_1', [32, 64, 96, 128, 160, 192, 224, 256]),
                'cnn_filters_2': hp.choice('cnn_filters_2', [64, 128, 192, 256, 320, 384, 448, 512]),
                'cnn_kernel_size': hp.choice('cnn_kernel_size', [2, 3, 4, 5]),
                'cnn_activation': hp.choice('cnn_activation', ['relu', 'elu', 'tanh']),
                
                'lstm_units_1': hp.choice('lstm_units_1', [64, 96, 128, 160, 192, 224, 256]),
                'lstm_units_2': hp.choice('lstm_units_2', [32, 48, 64, 80, 96, 112, 128]),
                'lstm_dropout': hp.uniform('lstm_dropout', 0.1, 0.5),
                'use_bidirectional': hp.choice('use_bidirectional', [True, False]),
                
                'use_attention': hp.choice('use_attention', [True, False]),
                'attention_heads': hp.choice('attention_heads', [4, 8, 12, 16]),
                
                'learning_rate': hp.loguniform('learning_rate', np.log(1e-5), np.log(1e-2)),
                'batch_size': hp.choice('batch_size', [16, 32, 64]),
                'optimizer': hp.choice('optimizer', ['adam', 'sgd', 'rmsprop']),
                
                'l1_reg': hp.loguniform('l1_reg', np.log(1e-6), np.log(1e-3)),
                'l2_reg': hp.loguniform('l2_reg', np.log(1e-6), np.log(1e-3)),
                'dropout_rate': hp.uniform('dropout_rate', 0.2, 0.6),
            }
        
        elif method == 'skopt':
            return [
                Integer(32, 256, name='cnn_filters_1'),
                Integer(64, 512, name='cnn_filters_2'),
                Integer(2, 5, name='cnn_kernel_size'),
                Categorical(['relu', 'elu', 'tanh'], name='cnn_activation'),
                
                Integer(64, 256, name='lstm_units_1'),
                Integer(32, 128, name='lstm_units_2'),
                Real(0.1, 0.5, name='lstm_dropout'),
                Categorical([True, False], name='use_bidirectional'),
                
                Categorical([True, False], name='use_attention'),
                Integer(4, 16, name='attention_heads'),
                
                Real(1e-5, 1e-2, prior='log-uniform', name='learning_rate'),
                Categorical([16, 32, 64], name='batch_size'),
                Categorical(['adam', 'sgd', 'rmsprop'], name='optimizer'),
                
                Real(1e-6, 1e-3, prior='log-uniform', name='l1_reg'),
                Real(1e-6, 1e-3, prior='log-uniform', name='l2_reg'),
                Real(0.2, 0.6, name='dropout_rate'),
            ]
    
    @staticmethod
    def get_transformer_space(trial: optuna.Trial = None, method: str = 'optuna') -> Dict:
        """Get Transformer hyperparameter space"""
        if method == 'optuna':
            return {
                # Architecture
                'num_heads': trial.suggest_int('num_heads', 4, 16, step=2),
                'head_size': trial.suggest_int('head_size', 32, 128, step=16),
                'num_blocks': trial.suggest_int('num_blocks', 2, 6),
                'ff_dim': trial.suggest_int('ff_dim', 128, 512, step=64),
                
                # Embeddings
                'embedding_dim': trial.suggest_int('embedding_dim', 64, 256, step=32),
                'positional_encoding': trial.suggest_categorical('positional_encoding', 
                                                                ['sinusoidal', 'learned']),
                
                # Regularization
                'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
                'attention_dropout': trial.suggest_float('attention_dropout', 0.0, 0.3),
                
                # Training
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                'warmup_steps': trial.suggest_int('warmup_steps', 100, 1000, step=100),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
                
                # Optimization
                'optimizer': trial.suggest_categorical('optimizer', ['adam', 'adamw', 'sgd']),
                'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
                'gradient_clip': trial.suggest_float('gradient_clip', 0.5, 2.0),
            }
        else:
            # Similar structure for other methods
            raise NotImplementedError(f"Transformer space for {method} not implemented")
    
    @staticmethod
    def get_xgboost_space(trial: optuna.Trial = None, method: str = 'optuna') -> Dict:
        """Get XGBoost hyperparameter space"""
        if method == 'optuna':
            return {
                # Tree parameters
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
                'max_leaves': trial.suggest_int('max_leaves', 0, 200),
                
                # Learning parameters
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
                'colsample_bynode': trial.suggest_float('colsample_bynode', 0.5, 1.0),
                
                # Regularization
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 5),
                
                # Advanced parameters
                'tree_method': trial.suggest_categorical('tree_method', ['auto', 'hist', 'approx']),
                'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
                'max_delta_step': trial.suggest_int('max_delta_step', 0, 10),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 2.0),
            }
        
        elif method == 'hyperopt':
            return {
                'n_estimators': hp.choice('n_estimators', range(100, 1001, 50)),
                'max_depth': hp.choice('max_depth', range(3, 16)),
                'min_child_weight': hp.choice('min_child_weight', range(1, 21)),
                'max_leaves': hp.choice('max_leaves', range(0, 201, 10)),
                
                'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(0.3)),
                'subsample': hp.uniform('subsample', 0.5, 1.0),
                'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
                'colsample_bylevel': hp.uniform('colsample_bylevel', 0.5, 1.0),
                'colsample_bynode': hp.uniform('colsample_bynode', 0.5, 1.0),
                
                'gamma': hp.uniform('gamma', 0, 5),
                'reg_alpha': hp.uniform('reg_alpha', 0, 5),
                'reg_lambda': hp.uniform('reg_lambda', 0.1, 5),
                
                'tree_method': hp.choice('tree_method', ['auto', 'hist', 'approx']),
                'grow_policy': hp.choice('grow_policy', ['depthwise', 'lossguide']),
                'max_delta_step': hp.choice('max_delta_step', range(0, 11)),
                'scale_pos_weight': hp.uniform('scale_pos_weight', 0.5, 2.0),
            }
        
        elif method == 'skopt':
            return [
                Integer(100, 1000, name='n_estimators'),
                Integer(3, 15, name='max_depth'),
                Integer(1, 20, name='min_child_weight'),
                Integer(0, 200, name='max_leaves'),
                
                Real(0.001, 0.3, prior='log-uniform', name='learning_rate'),
                Real(0.5, 1.0, name='subsample'),
                Real(0.5, 1.0, name='colsample_bytree'),
                Real(0.5, 1.0, name='colsample_bylevel'),
                Real(0.5, 1.0, name='colsample_bynode'),
                
                Real(0, 5, name='gamma'),
                Real(0, 5, name='reg_alpha'),
                Real(0.1, 5, name='reg_lambda'),
                
                Categorical(['auto', 'hist', 'approx'], name='tree_method'),
                Categorical(['depthwise', 'lossguide'], name='grow_policy'),
                Integer(0, 10, name='max_delta_step'),
                Real(0.5, 2.0, name='scale_pos_weight'),
            ]
    
    @staticmethod
    def get_lightgbm_space(trial: optuna.Trial = None, method: str = 'optuna') -> Dict:
        """Get LightGBM hyperparameter space"""
        if method == 'optuna':
            return {
                # Core parameters
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
                'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                'max_depth': trial.suggest_int('max_depth', -1, 20),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
                
                # Learning parameters
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 0, 10),
                
                # Regularization
                'lambda_l1': trial.suggest_float('lambda_l1', 0, 5),
                'lambda_l2': trial.suggest_float('lambda_l2', 0, 5),
                'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0, 1),
                
                # Advanced
                'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss']),
                'objective': trial.suggest_categorical('objective', ['binary', 'cross_entropy']),
                'metric': trial.suggest_categorical('metric', ['binary_logloss', 'auc']),
            }
        else:
            raise NotImplementedError(f"LightGBM space for {method} not implemented")


class ModelEvaluator:
    """Evaluate models with various metrics"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def evaluate_model(self, model, X_train, y_train, X_val, y_val, 
                      model_type: str, params: Dict) -> Dict[str, float]:
        """Evaluate model and return metrics"""
        try:
            # Train model with given parameters
            if model_type in ['cnn_lstm', 'transformer']:
                # Deep learning model
                trained_model = self._train_deep_model(
                    model_type, X_train, y_train, X_val, y_val, params
                )
                predictions = trained_model.predict(X_val)
            else:
                # Tree-based model
                trained_model = self._train_tree_model(
                    model_type, X_train, y_train, X_val, y_val, params
                )
                predictions = trained_model.predict_proba(X_val)[:, 1]
            
            # Calculate metrics
            metrics = self._calculate_metrics(predictions, y_val)
            
            # Trading simulation for financial metrics
            trading_metrics = self._simulate_trading(predictions, X_val, y_val)
            metrics.update(trading_metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating model: {str(e)}")
            return {'error': 1.0}
    
    def _train_deep_model(self, model_type: str, X_train, y_train, 
                         X_val, y_val, params: Dict):
        """Train deep learning model"""
        # Clear previous model to free memory
        K.clear_session()
        
        if model_type == 'cnn_lstm':
            # Build model with hyperparameters
            model = self._build_cnn_lstm(X_train.shape[1:], params)
        elif model_type == 'transformer':
            model = self._build_transformer(X_train.shape[1:], params)
        else:
            raise ValueError(f"Unknown deep model type: {model_type}")
        
        # Compile model
        optimizer = self._get_optimizer(params)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,  # Reduced for hyperparameter search
            batch_size=params.get('batch_size', 32),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
            ],
            verbose=0
        )
        
        return model
    
    def _build_cnn_lstm(self, input_shape: Tuple, params: Dict):
        """Build CNN-LSTM model with given parameters"""
        inputs = tf.keras.layers.Input(shape=input_shape)
        
        # CNN layers
        x = tf.keras.layers.Conv1D(
            params['cnn_filters_1'], 
            params['cnn_kernel_size'],
            activation=params['cnn_activation'],
            padding='same'
        )(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        x = tf.keras.layers.Dropout(params['dropout_rate'])(x)
        
        x = tf.keras.layers.Conv1D(
            params['cnn_filters_2'],
            params['cnn_kernel_size'],
            activation=params['cnn_activation'],
            padding='same'
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(params['dropout_rate'])(x)
        
        # LSTM layers
        if params.get('use_bidirectional', False):
            x = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(params['lstm_units_1'], return_sequences=True)
            )(x)
        else:
            x = tf.keras.layers.LSTM(params['lstm_units_1'], return_sequences=True)(x)
        
        x = tf.keras.layers.Dropout(params['lstm_dropout'])(x)
        
        if params.get('use_bidirectional', False):
            x = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(params['lstm_units_2'], return_sequences=True)
            )(x)
        else:
            x = tf.keras.layers.LSTM(params['lstm_units_2'], return_sequences=True)(x)
        
        # Attention if enabled
        if params.get('use_attention', False):
            attention = tf.keras.layers.MultiHeadAttention(
                num_heads=params['attention_heads'],
                key_dim=params['lstm_units_2'] // params['attention_heads']
            )(x, x)
            x = tf.keras.layers.Add()([x, attention])
            x = tf.keras.layers.LayerNormalization()(x)
        
        # Global pooling and output
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Add regularization
        if params.get('l1_reg', 0) > 0 or params.get('l2_reg', 0) > 0:
            for layer in model.layers:
                if hasattr(layer, 'kernel_regularizer'):
                    layer.kernel_regularizer = tf.keras.regularizers.l1_l2(
                        l1=params.get('l1_reg', 0),
                        l2=params.get('l2_reg', 0)
                    )
        
        return model
    
    def _build_transformer(self, input_shape: Tuple, params: Dict):
        """Build Transformer model with given parameters"""
        # Simplified transformer for hyperparameter search
        inputs = tf.keras.layers.Input(shape=input_shape)
        
        # Embedding
        x = tf.keras.layers.Dense(params['embedding_dim'])(inputs)
        
        # Positional encoding
        if params['positional_encoding'] == 'sinusoidal':
            # Add sinusoidal positional encoding
            positions = tf.range(start=0, limit=input_shape[0], delta=1)
            position_embedding = tf.keras.layers.Embedding(
                input_dim=input_shape[0],
                output_dim=params['embedding_dim']
            )(positions)
            x = x + position_embedding
        
        # Transformer blocks
        for _ in range(params['num_blocks']):
            # Multi-head attention
            attn_output = tf.keras.layers.MultiHeadAttention(
                num_heads=params['num_heads'],
                key_dim=params['head_size'],
                dropout=params['attention_dropout']
            )(x, x)
            
            # Skip connection and normalization
            x = tf.keras.layers.Add()([x, attn_output])
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
            
            # Feed-forward network
            ffn = tf.keras.Sequential([
                tf.keras.layers.Dense(params['ff_dim'], activation='relu'),
                tf.keras.layers.Dropout(params['dropout_rate']),
                tf.keras.layers.Dense(params['embedding_dim'])
            ])
            ffn_output = ffn(x)
            
            # Skip connection and normalization
            x = tf.keras.layers.Add()([x, ffn_output])
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Output
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        
        return tf.keras.Model(inputs=inputs, outputs=outputs)
    
    def _train_tree_model(self, model_type: str, X_train, y_train, 
                         X_val, y_val, params: Dict):
        """Train tree-based model"""
        # Flatten inputs for tree models
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_val_flat = X_val.reshape(X_val.shape[0], -1)
        
        if model_type == 'xgboost':
            model = xgb.XGBClassifier(
                **{k: v for k, v in params.items() if k in xgb.XGBClassifier().get_params()},
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=42
            )
            model.fit(
                X_train_flat, y_train,
                eval_set=[(X_val_flat, y_val)],
                early_stopping_rounds=20,
                verbose=False
            )
            
        elif model_type == 'lightgbm':
            model = lgb.LGBMClassifier(
                **{k: v for k, v in params.items() if k in lgb.LGBMClassifier().get_params()},
                random_state=42,
                n_jobs=-1
            )
            model.fit(
                X_train_flat, y_train,
                eval_set=[(X_val_flat, y_val)],
                callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
            )
        else:
            raise ValueError(f"Unknown tree model type: {model_type}")
        
        return model
    
    def _get_optimizer(self, params: Dict):
        """Get optimizer based on parameters"""
        lr = params.get('learning_rate', 0.001)
        optimizer_name = params.get('optimizer', 'adam')
        
        if optimizer_name == 'adam':
            return tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
        elif optimizer_name == 'adamw':
            return tf.keras.optimizers.AdamW(
                learning_rate=lr, 
                weight_decay=params.get('weight_decay', 0.01)
            )
        elif optimizer_name == 'sgd':
            return tf.keras.optimizers.SGD(
                learning_rate=lr, 
                momentum=0.9,
                clipnorm=params.get('gradient_clip', 1.0)
            )
        elif optimizer_name == 'rmsprop':
            return tf.keras.optimizers.RMSprop(learning_rate=lr, clipnorm=1.0)
        else:
            return tf.keras.optimizers.Adam(learning_rate=lr)
    
    def _calculate_metrics(self, predictions: np.ndarray, y_true: np.ndarray) -> Dict:
        """Calculate classification metrics"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        y_pred = (predictions > 0.5).astype(int)
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'auc': roc_auc_score(y_true, predictions) if len(np.unique(y_true)) > 1 else 0.5
        }
    
    def _simulate_trading(self, predictions: np.ndarray, X_val: np.ndarray, 
                         y_val: np.ndarray) -> Dict:
        """Simulate trading to get financial metrics"""
        # Simplified trading simulation for hyperparameter search
        returns = []
        
        for i in range(len(predictions)):
            if predictions[i] > 0.5:
                # Simulated return based on actual direction
                returns.append(0.01 if y_val[i] == 1 else -0.01)
            else:
                returns.append(-0.01 if y_val[i] == 1 else 0.01)
        
        returns = np.array(returns)
        
        # Calculate metrics
        total_return = np.sum(returns) * 100
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
        
        # Calculate drawdown
        cumulative_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (running_max - cumulative_returns) / (running_max + 1e-10)
        max_drawdown = np.max(drawdown) * 100
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': np.mean(returns > 0)
        }


class BayesianOptimizer:
    """Bayesian optimization using Optuna"""
    
    def __init__(self, config: OptimizationConfig, trading_config: TradingConfig):
        self.config = config
        self.trading_config = trading_config
        self.evaluator = ModelEvaluator(trading_config)
        self.logger = logging.getLogger(__name__)
        
    def optimize(self, model_type: str, X_train, y_train, X_val, y_val,
                storage_url: Optional[str] = None) -> Tuple[Dict, optuna.Study]:
        """Run Bayesian optimization"""
        # Create study
        study_name = f"{model_type}_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Select sampler
        sampler = self._get_sampler()
        pruner = self._get_pruner()
        
        study = optuna.create_study(
            study_name=study_name,
            direction='maximize',
            sampler=sampler,
            pruner=pruner,
            storage=storage_url,
            load_if_exists=True
        )
        
        # Define objective function
        def objective(trial):
            # Get hyperparameter space
            space_func = getattr(HyperparameterSpace, f"get_{model_type}_space")
            params = space_func(trial, 'optuna')
            
            # Evaluate model
            metrics = self.evaluator.evaluate_model(
                None, X_train, y_train, X_val, y_val, model_type, params
            )
            
            # Handle errors
            if 'error' in metrics:
                return 0.0
            
            # Select metric to optimize
            if self.config.cv_metric == 'sharpe_ratio':
                return metrics.get('sharpe_ratio', 0.0)
            elif self.config.cv_metric == 'total_return':
                return metrics.get('total_return', 0.0)
            elif self.config.cv_metric == 'accuracy':
                return metrics.get('accuracy', 0.0)
            else:
                # Combined metric
                return (metrics.get('sharpe_ratio', 0.0) * 0.5 + 
                       metrics.get('accuracy', 0.0) * 0.5)
        
        # Optimize
        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            n_jobs=self.config.n_jobs if self.config.use_parallel else 1,
            show_progress_bar=True
        )
        
        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value
        
        self.logger.info(f"Best parameters for {model_type}: {best_params}")
        self.logger.info(f"Best {self.config.cv_metric}: {best_value}")
        
        return best_params, study
    
    def _get_sampler(self):
        """Get Optuna sampler based on configuration"""
        if self.config.sampler == 'tpe':
            return TPESampler(seed=42)
        elif self.config.sampler == 'cmaes':
            return CmaEsSampler(seed=42)
        elif self.config.sampler == 'random':
            return RandomSampler(seed=42)
        elif self.config.sampler == 'grid':
            # Grid sampler requires search space
            return GridSampler()
        else:
            return TPESampler(seed=42)
    
    def _get_pruner(self):
        """Get Optuna pruner based on configuration"""
        if self.config.pruner == 'median':
            return MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        elif self.config.pruner == 'hyperband':
            return HyperbandPruner()
        elif self.config.pruner == 'successive_halving':
            return SuccessiveHalvingPruner()
        elif self.config.pruner == 'threshold':
            return ThresholdPruner(lower=0.0)
        else:
            return MedianPruner()


class MultiObjectiveOptimizer:
    """Multi-objective optimization for trading strategies"""
    
    def __init__(self, config: OptimizationConfig, trading_config: TradingConfig):
        self.config = config
        self.trading_config = trading_config
        self.evaluator = ModelEvaluator(trading_config)
        self.logger = logging.getLogger(__name__)
        
    def optimize(self, model_type: str, X_train, y_train, X_val, y_val) -> Tuple[List[Dict], optuna.Study]:
        """Run multi-objective optimization"""
        # Create multi-objective study
        study = optuna.create_study(
            directions=['maximize', 'minimize'],  # Return, Risk
            sampler=NSGAIISampler(seed=42)
        )
        
        def objective(trial):
            # Get hyperparameters
            space_func = getattr(HyperparameterSpace, f"get_{model_type}_space")
            params = space_func(trial, 'optuna')
            
            # Evaluate model
            metrics = self.evaluator.evaluate_model(
                None, X_train, y_train, X_val, y_val, model_type, params
            )
            
            # Return multiple objectives
            total_return = metrics.get('total_return', 0.0)
            max_drawdown = metrics.get('max_drawdown', 100.0)
            
            return total_return, max_drawdown
        
        # Optimize
        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            n_jobs=self.config.n_jobs if self.config.use_parallel else 1,
            show_progress_bar=True
        )
        
        # Get Pareto front solutions
        pareto_front = self._get_pareto_front(study)
        
        self.logger.info(f"Found {len(pareto_front)} Pareto optimal solutions")
        
        return pareto_front, study
    
    def _get_pareto_front(self, study: optuna.Study) -> List[Dict]:
        """Extract Pareto front solutions"""
        pareto_front = []
        
        for trial in study.best_trials:
            solution = {
                'params': trial.params,
                'values': {
                    'return': trial.values[0],
                    'risk': trial.values[1]
                },
                'trial_number': trial.number
            }
            pareto_front.append(solution)
        
        return pareto_front


class HyperparameterTuner:
    """Main hyperparameter tuning orchestrator"""
    
    def __init__(self, config: OptimizationConfig, trading_config: TradingConfig):
        self.config = config
        self.trading_config = trading_config
        self.logger = logging.getLogger(__name__)
        self.results = {}
        
    def tune_all_models(self, X_train, y_train, X_val, y_val, 
                       models: List[str] = None) -> Dict:
        """Tune hyperparameters for all models"""
        if models is None:
            models = ['cnn_lstm', 'transformer', 'xgboost', 'lightgbm']
        
        self.logger.info(f"Starting hyperparameter tuning for {len(models)} models")
        
        for model_type in models:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Tuning {model_type}")
            self.logger.info(f"{'='*60}")
            
            try:
                if self.config.method == 'bayesian':
                    optimizer = BayesianOptimizer(self.config, self.trading_config)
                    best_params, study = optimizer.optimize(
                        model_type, X_train, y_train, X_val, y_val
                    )
                    
                    self.results[model_type] = {
                        'best_params': best_params,
                        'best_value': study.best_value,
                        'study': study
                    }
                    
                elif self.config.method == 'multi_objective':
                    optimizer = MultiObjectiveOptimizer(self.config, self.trading_config)
                    pareto_front, study = optimizer.optimize(
                        model_type, X_train, y_train, X_val, y_val
                    )
                    
                    self.results[model_type] = {
                        'pareto_front': pareto_front,
                        'study': study
                    }
                    
                else:
                    raise ValueError(f"Unknown optimization method: {self.config.method}")
                    
            except Exception as e:
                self.logger.error(f"Error tuning {model_type}: {str(e)}")
                import traceback
                traceback.print_exc()
                
        return self.results
    
    def save_results(self, output_dir: str = './output/hyperparameter_tuning'):
        """Save tuning results"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save parameters
        for model_type, results in self.results.items():
            # Save best parameters
            if 'best_params' in results:
                params_file = Path(output_dir) / f"{model_type}_best_params.json"
                with open(params_file, 'w') as f:
                    json.dump(results['best_params'], f, indent=4)
                    
            # Save Pareto front
            if 'pareto_front' in results:
                pareto_file = Path(output_dir) / f"{model_type}_pareto_front.json"
                with open(pareto_file, 'w') as f:
                    json.dump(results['pareto_front'], f, indent=4, default=str)
                    
            # Save study
            if 'study' in results:
                study_file = Path(output_dir) / f"{model_type}_study.pkl"
                with open(study_file, 'wb') as f:
                    pickle.dump(results['study'], f)
                    
        self.logger.info(f"Results saved to {output_dir}")
    
    def visualize_results(self, output_dir: str = './output/hyperparameter_tuning'):
        """Create visualizations of optimization results"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for model_type, results in self.results.items():
            if 'study' not in results:
                continue
                
            study = results['study']
            
            # Create visualizations
            try:
                # Optimization history
                fig = plot_optimization_history(study)
                fig.write_image(Path(output_dir) / f"{model_type}_optimization_history.png")
                
                # Parameter importance
                if len(study.trials) > 10:
                    fig = plot_param_importances(study)
                    fig.write_image(Path(output_dir) / f"{model_type}_param_importance.png")
                
                # Parallel coordinate plot
                fig = plot_parallel_coordinate(study)
                fig.write_image(Path(output_dir) / f"{model_type}_parallel_coordinate.png")
                
                # Slice plot
                fig = plot_slice(study)
                fig.write_image(Path(output_dir) / f"{model_type}_slice.png")
                
                # For multi-objective
                if self.config.method == 'multi_objective':
                    fig = plot_pareto_front(study)
                    fig.write_image(Path(output_dir) / f"{model_type}_pareto_front.png")
                    
            except Exception as e:
                self.logger.warning(f"Could not create visualization for {model_type}: {str(e)}")
                
        self.logger.info(f"Visualizations saved to {output_dir}")
    
    def generate_report(self, output_dir: str = './output/hyperparameter_tuning'):
        """Generate comprehensive tuning report"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        report_path = Path(output_dir) / f"tuning_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Hyperparameter Tuning Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                .section {{ margin: 30px 0; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; 
                          background-color: #f0f0f0; border-radius: 5px; }}
                pre {{ background-color: #f5f5f5; padding: 10px; overflow-x: auto; }}
            </style>
        </head>
        <body>
            <h1>Hyperparameter Tuning Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="section">
                <h2>Configuration</h2>
                <pre>{json.dumps(self.config.__dict__, indent=2)}</pre>
            </div>
            
            <div class="section">
                <h2>Results Summary</h2>
                {self._generate_results_summary()}
            </div>
            
            <div class="section">
                <h2>Best Parameters</h2>
                {self._generate_params_table()}
            </div>
            
            <div class="section">
                <h2>Optimization Visualizations</h2>
                {self._generate_visualization_section(output_dir)}
            </div>
        </body>
        </html>
        """
        
        with open(report_path, 'w') as f:
            f.write(html_content)
            
        self.logger.info(f"Report generated: {report_path}")
        
        return report_path
    
    def _generate_results_summary(self) -> str:
        """Generate results summary HTML"""
        summary_html = ""
        
        for model_type, results in self.results.items():
            if 'best_value' in results:
                summary_html += f"""
                <div class="metric">
                    <h3>{model_type}</h3>
                    <p>Best {self.config.cv_metric}: {results['best_value']:.4f}</p>
                    <p>Trials: {len(results.get('study', {}).trials)}</p>
                </div>
                """
            elif 'pareto_front' in results:
                summary_html += f"""
                <div class="metric">
                    <h3>{model_type}</h3>
                    <p>Pareto solutions: {len(results['pareto_front'])}</p>
                </div>
                """
                
        return summary_html
    
    def _generate_params_table(self) -> str:
        """Generate best parameters table HTML"""
        table_html = "<table><tr><th>Model</th><th>Parameter</th><th>Value</th></tr>"
        
        for model_type, results in self.results.items():
            if 'best_params' in results:
                for param, value in results['best_params'].items():
                    table_html += f"""
                    <tr>
                        <td>{model_type}</td>
                        <td>{param}</td>
                        <td>{value}</td>
                    </tr>
                    """
                    
        table_html += "</table>"
        return table_html
    
    def _generate_visualization_section(self, output_dir: str) -> str:
        """Generate visualization section HTML"""
        viz_html = ""
        
        for model_type in self.results.keys():
            viz_html += f"<h3>{model_type}</h3>"
            
            # Check for visualization files
            viz_files = [
                f"{model_type}_optimization_history.png",
                f"{model_type}_param_importance.png",
                f"{model_type}_parallel_coordinate.png",
                f"{model_type}_slice.png"
            ]
            
            if self.config.method == 'multi_objective':
                viz_files.append(f"{model_type}_pareto_front.png")
                
            for viz_file in viz_files:
                viz_path = Path(output_dir) / viz_file
                if viz_path.exists():
                    viz_html += f'<img src="{viz_file}" style="max-width: 800px; margin: 10px;"><br>'
                    
        return viz_html


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Advanced Hyperparameter Tuning for Forex Prediction'
    )
    
    # Optimization method
    parser.add_argument('--method', choices=['bayesian', 'multi_objective', 'grid', 'random'],
                       default='bayesian', help='Optimization method')
    
    # Models to tune
    parser.add_argument('--models', nargs='+', 
                       default=['cnn_lstm', 'transformer', 'xgboost', 'lightgbm'],
                       help='Models to tune')
    
    # Optimization parameters
    parser.add_argument('--n-trials', type=int, default=100,
                       help='Number of optimization trials')
    parser.add_argument('--timeout', type=int, default=3600,
                       help='Timeout in seconds')
    parser.add_argument('--metric', choices=['sharpe_ratio', 'total_return', 'accuracy'],
                       default='sharpe_ratio', help='Metric to optimize')
    
    # Sampler and pruner
    parser.add_argument('--sampler', choices=['tpe', 'cmaes', 'random', 'grid'],
                       default='tpe', help='Optuna sampler')
    parser.add_argument('--pruner', choices=['median', 'hyperband', 'successive_halving'],
                       default='median', help='Optuna pruner')
    
    # Parallel processing
    parser.add_argument('--n-jobs', type=int, default=1,
                       help='Number of parallel jobs')
    parser.add_argument('--no-parallel', action='store_true',
                       help='Disable parallel processing')
    
    # Data parameters
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Data directory')
    parser.add_argument('--currency-pair', type=str, default='EURUSD',
                       help='Currency pair to optimize')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='./output/hyperparameter_tuning',
                       help='Output directory')
    parser.add_argument('--no-visualization', action='store_true',
                       help='Skip visualization generation')
    parser.add_argument('--no-report', action='store_true',
                       help='Skip report generation')
    
    args = parser.parse_args()
    
    # Create configurations
    opt_config = OptimizationConfig(
        method=args.method,
        n_trials=args.n_trials,
        timeout=args.timeout,
        cv_metric=args.metric,
        sampler=args.sampler,
        pruner=args.pruner,
        n_jobs=args.n_jobs,
        use_parallel=not args.no_parallel
    )
    
    trading_config = TradingConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    # Load and prepare data
    logger.info("Loading data...")
    predictor = EnhancedForexPredictor(trading_config)
    raw_data = predictor.load_data(args.currency_pair)
    data_splits = predictor.prepare_data(raw_data)
    
    # Feature engineering
    logger.info("Engineering features...")
    feature_engineer = AdvancedFeatureEngineer(trading_config)
    train_features = feature_engineer.engineer_features(data_splits['train'])
    val_features = feature_engineer.engineer_features(data_splits['validation'])
    
    # Create sequences
    X_train, y_train = predictor.create_sequences(train_features)
    X_val, y_val = predictor.create_sequences(val_features)
    
    logger.info(f"Data shapes - Train: {X_train.shape}, Validation: {X_val.shape}")
    
    # Run hyperparameter tuning
    tuner = HyperparameterTuner(opt_config, trading_config)
    results = tuner.tune_all_models(X_train, y_train, X_val, y_val, args.models)
    
    # Save results
    tuner.save_results(args.output_dir)
    
    # Generate visualizations
    if not args.no_visualization:
        tuner.visualize_results(args.output_dir)
    
    # Generate report
    if not args.no_report:
        report_path = tuner.generate_report(args.output_dir)
        logger.info(f"Report generated: {report_path}")
    
    logger.info("Hyperparameter tuning completed!")
    
    return 0


if __name__ == "__main__":
    main()
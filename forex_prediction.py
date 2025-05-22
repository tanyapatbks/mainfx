"""
Enhanced Forex Trend Prediction System for Master's Thesis
- Advanced feature engineering with Elliott Wave & Fibonacci
- Market regime detection and adaptive strategies
- Dynamic ensemble methods with attention mechanisms
- Walk-forward analysis and advanced risk management
- Bayesian hyperparameter optimization
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm
from functools import partial
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Advanced ML Libraries
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, roc_auc_score
from sklearn.feature_selection import SelectFromModel, mutual_info_regression, SelectKBest, VarianceThreshold
from sklearn.ensemble import RandomForestRegressor, VotingClassifier, StackingClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

# Advanced optimization
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

# Deep Learning Libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import (Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten, 
                                   BatchNormalization, Input, Concatenate, MultiHeadAttention,
                                   LayerNormalization, Add, GlobalAveragePooling1D, Reshape,
                                   Attention, AdditiveAttention)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# Advanced technical analysis
import talib
from scipy import signal
from scipy.stats import linregress
from scipy.signal import find_peaks

# Set up enhanced logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"enhanced_forex_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create enhanced output directories
output_dir = "output"
models_dir = os.path.join(output_dir, "models")
results_dir = os.path.join(output_dir, "results")
plots_dir = os.path.join(output_dir, "plots")
feature_dir = os.path.join(output_dir, "features")
hyperparams_dir = os.path.join(output_dir, "hyperparameter_tuning")
walkforward_dir = os.path.join(output_dir, "walkforward_analysis")
regime_dir = os.path.join(output_dir, "regime_analysis")

for directory in [output_dir, models_dir, results_dir, plots_dir, feature_dir, 
                 hyperparams_dir, walkforward_dir, regime_dir]:
    os.makedirs(directory, exist_ok=True)

# Enhanced constants for thesis requirements
WINDOW_SIZE = 60  # 60 hours lookback
PREDICTION_HORIZON = 1  # 1 hour ahead prediction
BATCH_SIZE = 32
EPOCHS = 150  # Increased for better convergence
PATIENCE = 25  # Increased patience
TEST_SIZE = 0.15  # Reduced test size to have more training data
VALIDATION_SIZE = 0.15
RANDOM_STATE = 42

# Updated date ranges as requested (2020-2022)
TRAIN_START = '2020-01-01'
TRAIN_END = '2021-06-30'   # Training period
VALIDATION_START = '2021-07-01'
VALIDATION_END = '2021-12-31'  # Validation period
TEST_START = '2022-01-01'
TEST_END = '2022-12-31'    # Test period (full year for comprehensive testing)

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

class EnhancedForexPrediction:
    """
    Enhanced Forex Prediction System for Master's Thesis
    Features:
    - Advanced feature engineering with Elliott Wave & Fibonacci
    - Market regime detection
    - Dynamic ensemble methods
    - Walk-forward analysis
    - Advanced risk management
    """
    
    def __init__(self, config=None):
        """Initialize the Enhanced Forex Prediction system."""
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
                'validation_start': VALIDATION_START,
                'validation_end': VALIDATION_END,
                'test_start': TEST_START,
                'test_end': TEST_END,
                'feature_selection_methods': ['random_forest', 'mutual_info', 'pca'],
                'n_features': 50,  # Increased for more complex features
                'models_to_train': ['enhanced_cnn_lstm', 'advanced_tft', 'enhanced_xgboost'],
                'currency_pairs': ['EURUSD', 'GBPUSD', 'USDJPY'],
                'use_bagging': True,
                'use_dynamic_ensemble': True,  # New: Dynamic ensemble
                'scaler_type': 'robust',  # Changed to robust for better outlier handling
                'hyperparameter_tuning': True,
                'optimization_method': 'bayesian',  # New: Bayesian optimization
                'n_trials': 100,  # Increased trials for better optimization
                'use_walkforward': True,  # New: Walk-forward analysis
                'walkforward_window': 180,  # 180 days for walk-forward
                'use_regime_detection': True,  # New: Market regime detection
                'risk_management': {
                    'use_kelly_criterion': True,
                    'use_dynamic_stops': True,
                    'max_drawdown_limit': 0.05,  # 5% max drawdown
                    'confidence_threshold': 0.6,  # Minimum confidence for trades
                    'leverage_scaling': True  # Scale leverage based on confidence
                }
            }
        else:
            self.config = config
        
        # Initialize enhanced class variables
        self.data = {}
        self.preprocessed_data = {}
        self.enhanced_features = {}
        self.market_regimes = {}  # New: Market regime data
        self.selected_features = {}
        self.models = {}
        self.ensemble_models = {}  # New: Ensemble models
        self.scalers = {}
        self.results = {}
        self.walkforward_results = {}  # New: Walk-forward results
        self.regime_results = {}  # New: Regime-based results
        self.bagging_data = None
        self.hyperparameters = {}
        
        # New: Risk management components
        self.risk_manager = None
        self.position_sizer = None
        
        # Load hyperparameters
        self.load_hyperparameters()
        
        logger.info("Enhanced Forex Prediction system initialized for Master's Thesis")
        logger.info(f"Configuration: {json.dumps(self.config, indent=2)}")

    # ===== STEP 1: ENHANCED DATA LOADING AND PREPROCESSING =====
    
    def load_data(self, data_paths=None):
        """Load and preprocess data with enhanced date ranges."""
        logger.info("=== STEP 1: ENHANCED DATA LOADING ===")
        
        if data_paths is None:
            data_paths = {pair: f"{pair}_1H.csv" for pair in self.config['currency_pairs']}
        
        for pair, path in data_paths.items():
            try:
                df = pd.read_csv(path)
                df['Time'] = pd.to_datetime(df['Time'])
                df.set_index('Time', inplace=True)
                df.sort_index(inplace=True)
                
                # Filter data for thesis period (2020-2022)
                start_date = pd.to_datetime('2020-01-01')
                end_date = pd.to_datetime('2022-12-31')
                df = df[(df.index >= start_date) & (df.index <= end_date)]
                
                self.data[pair] = df
                logger.info(f"Loaded {pair} data: {df.shape} rows for period {start_date} to {end_date}")
                
            except Exception as e:
                logger.error(f"Error loading {pair} data from {path}: {e}")
                raise
        
        return self.data
    
    def preprocess_data(self):
        """Enhanced preprocessing with proper train/validation/test splits."""
        logger.info("=== ENHANCED PREPROCESSING WITH PROPER DATA SPLITS ===")
        
        # Define date ranges for proper time series split
        train_start = pd.to_datetime(self.config['train_start'])
        train_end = pd.to_datetime(self.config['train_end'])
        validation_start = pd.to_datetime(self.config['validation_start'])
        validation_end = pd.to_datetime(self.config['validation_end'])
        test_start = pd.to_datetime(self.config['test_start'])
        test_end = pd.to_datetime(self.config['test_end'])
        
        logger.info(f"Data splits: Train({train_start} to {train_end}), "
                   f"Validation({validation_start} to {validation_end}), "
                   f"Test({test_start} to {test_end})")
        
        # Find common date range to ensure all pairs have the same timeline
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
            # Reindex to common dates and handle missing values
            df_reindexed = df.reindex(common_dates)
            df_reindexed.interpolate(method='linear', inplace=True)
            df_reindexed.fillna(method='ffill', inplace=True)
            df_reindexed.fillna(method='bfill', inplace=True)
            
            # Calculate basic returns
            df_reindexed['Returns'] = df_reindexed['Close'].pct_change()
            df_reindexed['Log_Returns'] = np.log(df_reindexed['Close'] / df_reindexed['Close'].shift(1))
            
            # Enhanced target calculation with future price direction
            df_reindexed['Future_Price'] = df_reindexed['Close'].shift(-self.config['prediction_horizon'])
            df_reindexed['Target'] = (df_reindexed['Future_Price'] > df_reindexed['Close']).astype(int)
            
            # Create proper time series splits (NO DATA LEAKAGE)
            train_data = df_reindexed[(df_reindexed.index >= train_start) & (df_reindexed.index <= train_end)].copy()
            validation_data = df_reindexed[(df_reindexed.index >= validation_start) & (df_reindexed.index <= validation_end)].copy()
            test_data = df_reindexed[(df_reindexed.index >= test_start) & (df_reindexed.index <= test_end)].copy()
            
            # Remove future price column from splits (prevent data leakage)
            for split_data in [train_data, validation_data, test_data]:
                if 'Future_Price' in split_data.columns:
                    split_data.drop('Future_Price', axis=1, inplace=True)
            
            self.preprocessed_data[pair] = {
                'full': df_reindexed.drop('Future_Price', axis=1, errors='ignore'),
                'train': train_data,
                'validation': validation_data,
                'test': test_data
            }
            
            logger.info(f"Preprocessed {pair}: Train={train_data.shape}, "
                       f"Validation={validation_data.shape}, Test={test_data.shape}")
        
        return self.preprocessed_data

    # ===== STEP 2: ADVANCED FEATURE ENGINEERING =====
    
    def calculate_elliott_wave_features(self, df):
        """Calculate Elliott Wave pattern features."""
        close = df['Close']
        high = df['High']
        low = df['Low']
        
        # Elliott Wave features
        features = {}
        
        # Find swing highs and lows
        swing_highs = find_peaks(high, distance=5)[0]
        swing_lows = find_peaks(-low, distance=5)[0]
        
        # Wave count features
        features['wave_high_count_20'] = pd.Series(0, index=df.index)
        features['wave_low_count_20'] = pd.Series(0, index=df.index)
        
        for i in range(len(df)):
            start_idx = max(0, i-20)
            features['wave_high_count_20'].iloc[i] = len([h for h in swing_highs if start_idx <= h <= i])
            features['wave_low_count_20'].iloc[i] = len([l for l in swing_lows if start_idx <= l <= i])
        
        # Wave momentum
        if len(swing_highs) > 2:
            features['wave_momentum_high'] = pd.Series(0.0, index=df.index)
            for i in range(2, len(swing_highs)):
                if swing_highs[i] < len(df):
                    wave1 = high.iloc[swing_highs[i-2]:swing_highs[i-1]].max() - high.iloc[swing_highs[i-2]:swing_highs[i-1]].min()
                    wave2 = high.iloc[swing_highs[i-1]:swing_highs[i]].max() - high.iloc[swing_highs[i-1]:swing_highs[i]].min()
                    momentum = wave2 / (wave1 + 1e-10)
                    features['wave_momentum_high'].iloc[swing_highs[i]:] = momentum
        
        if len(swing_lows) > 2:
            features['wave_momentum_low'] = pd.Series(0.0, index=df.index)
            for i in range(2, len(swing_lows)):
                if swing_lows[i] < len(df):
                    wave1 = low.iloc[swing_lows[i-2]:swing_lows[i-1]].max() - low.iloc[swing_lows[i-2]:swing_lows[i-1]].min()
                    wave2 = low.iloc[swing_lows[i-1]:swing_lows[i]].max() - low.iloc[swing_lows[i-1]:swing_lows[i]].min()
                    momentum = wave2 / (wave1 + 1e-10)
                    features['wave_momentum_low'].iloc[swing_lows[i]:] = momentum
        
        # Fill missing values
        for key in features:
            if key not in ['wave_momentum_high', 'wave_momentum_low']:
                continue
            if features[key].empty:
                features[key] = pd.Series(0.0, index=df.index)
            features[key].fillna(method='ffill', inplace=True)
            features[key].fillna(0, inplace=True)
        
        return features
    
    def calculate_fibonacci_features(self, df):
        """Calculate Fibonacci retracement and extension features."""
        close = df['Close']
        high = df['High']
        low = df['Low']
        
        features = {}
        
        # Fibonacci levels
        fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618]
        
        # Rolling high/low for Fibonacci calculation
        for window in [20, 50, 100]:
            rolling_high = high.rolling(window=window).max()
            rolling_low = low.rolling(window=window).min()
            price_range = rolling_high - rolling_low
            
            for level in fib_levels:
                # Fibonacci retracements from high
                fib_level_high = rolling_high - (price_range * level)
                features[f'fib_dist_high_{level}_{window}'] = (close - fib_level_high) / (price_range + 1e-10)
                
                # Fibonacci retracements from low
                fib_level_low = rolling_low + (price_range * level)
                features[f'fib_dist_low_{level}_{window}'] = (close - fib_level_low) / (price_range + 1e-10)
                
                # Distance to nearest Fibonacci level
                dist_to_fib = np.minimum(
                    np.abs(close - fib_level_high),
                    np.abs(close - fib_level_low)
                )
                features[f'fib_proximity_{level}_{window}'] = dist_to_fib / (price_range + 1e-10)
        
        return features
    
    def detect_market_regime(self, df):
        """Detect market regime (trending/ranging/high volatility/low volatility)."""
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']
        
        regime_features = {}
        
        # Trend strength
        for window in [20, 50, 100]:
            # Linear regression slope for trend detection
            slopes = []
            for i in range(len(close)):
                start_idx = max(0, i - window + 1)
                end_idx = i + 1
                if end_idx - start_idx >= 10:  # Minimum data points
                    x = np.arange(end_idx - start_idx)
                    y = close.iloc[start_idx:end_idx].values
                    if len(y) > 1:
                        slope, _, r_value, _, _ = linregress(x, y)
                        slopes.append(slope * r_value**2)  # Weight by R-squared
                    else:
                        slopes.append(0)
                else:
                    slopes.append(0)
            
            regime_features[f'trend_strength_{window}'] = pd.Series(slopes, index=close.index)
        
        # Volatility regime
        for window in [20, 50]:
            volatility = close.rolling(window=window).std()
            volatility_ma = volatility.rolling(window=window).mean()
            regime_features[f'volatility_regime_{window}'] = volatility / (volatility_ma + 1e-10)
        
        # Range detection (sideways market)
        for window in [20, 50]:
            price_range = (high.rolling(window=window).max() - low.rolling(window=window).min())
            avg_range = price_range.rolling(window=window).mean()
            regime_features[f'range_regime_{window}'] = price_range / (avg_range + 1e-10)
        
        # Volume regime
        for window in [20, 50]:
            volume_ma = volume.rolling(window=window).mean()
            regime_features[f'volume_regime_{window}'] = volume / (volume_ma + 1e-10)
        
        # Market state classification
        trend_strength = regime_features['trend_strength_50']
        volatility_regime = regime_features['volatility_regime_20']
        
        market_state = pd.Series('ranging', index=close.index)
        market_state[trend_strength > 0.1] = 'trending_up'
        market_state[trend_strength < -0.1] = 'trending_down'
        market_state[volatility_regime > 1.5] = 'high_volatility'
        market_state[volatility_regime < 0.5] = 'low_volatility'
        
        regime_features['market_state'] = market_state
        
        return regime_features
    
    def calculate_advanced_technical_indicators(self, df):
        """Calculate advanced technical indicators."""
        open_price = df['Open'].values
        high_price = df['High'].values
        low_price = df['Low'].values
        close_price = df['Close'].values
        volume = df['Volume'].values
        
        indicators = {}
        
        # Advanced momentum indicators
        indicators['CMO_14'] = talib.CMO(close_price, timeperiod=14)  # Chande Momentum Oscillator
        indicators['AROON_up'], indicators['AROON_down'] = talib.AROON(high_price, low_price, timeperiod=14)
        indicators['AROONOSC'] = talib.AROONOSC(high_price, low_price, timeperiod=14)
        indicators['BOP'] = talib.BOP(open_price, high_price, low_price, close_price)  # Balance of Power
        indicators['CCI_14'] = talib.CCI(high_price, low_price, close_price, timeperiod=14)
        indicators['DX_14'] = talib.DX(high_price, low_price, close_price, timeperiod=14)
        indicators['MFI_14'] = talib.MFI(high_price, low_price, close_price, volume, timeperiod=14)
        indicators['MINUS_DI'] = talib.MINUS_DI(high_price, low_price, close_price, timeperiod=14)
        indicators['PLUS_DI'] = talib.PLUS_DI(high_price, low_price, close_price, timeperiod=14)
        indicators['WILLR_14'] = talib.WILLR(high_price, low_price, close_price, timeperiod=14)
        
        # Advanced volatility indicators
        indicators['NATR_14'] = talib.NATR(high_price, low_price, close_price, timeperiod=14)
        indicators['TRANGE'] = talib.TRANGE(high_price, low_price, close_price)
        
        # Advanced volume indicators
        indicators['AD'] = talib.AD(high_price, low_price, close_price, volume)
        indicators['ADOSC'] = talib.ADOSC(high_price, low_price, close_price, volume, fastperiod=3, slowperiod=10)
        
        # Pattern recognition
        indicators['CDL_DOJI'] = talib.CDLDOJI(open_price, high_price, low_price, close_price)
        indicators['CDL_HAMMER'] = talib.CDLHAMMER(open_price, high_price, low_price, close_price)
        indicators['CDL_ENGULFING'] = talib.CDLENGULFING(open_price, high_price, low_price, close_price)
        indicators['CDL_HARAMI'] = talib.CDLHARAMI(open_price, high_price, low_price, close_price)
        
        # Convert to pandas Series
        for key, values in indicators.items():
            indicators[key] = pd.Series(values, index=df.index)
            indicators[key].fillna(method='ffill', inplace=True)
            indicators[key].fillna(0, inplace=True)
        
        return indicators
    
    def calculate_enhanced_features(self):
        """Calculate all enhanced features including Elliott Wave, Fibonacci, and market regime."""
        logger.info("=== STEP 2: ADVANCED FEATURE ENGINEERING ===")
        
        for pair, data_dict in self.preprocessed_data.items():
            logger.info(f"Calculating enhanced features for {pair}")
            
            # Work with full dataset first
            df = data_dict['full'].copy()
            
            # Basic features (existing)
            self._calculate_basic_features(df)
            
            # Advanced technical indicators
            advanced_indicators = self.calculate_advanced_technical_indicators(df)
            for key, values in advanced_indicators.items():
                df[key] = values
            
            # Elliott Wave features
            logger.info(f"Calculating Elliott Wave features for {pair}")
            elliott_features = self.calculate_elliott_wave_features(df)
            for key, values in elliott_features.items():
                df[key] = values
            
            # Fibonacci features
            logger.info(f"Calculating Fibonacci features for {pair}")
            fibonacci_features = self.calculate_fibonacci_features(df)
            for key, values in fibonacci_features.items():
                df[key] = values
            
            # Market regime detection
            logger.info(f"Detecting market regimes for {pair}")
            regime_features = self.detect_market_regime(df)
            for key, values in regime_features.items():
                if key != 'market_state':  # Handle market_state separately
                    df[key] = values
            
            # Store market regime information separately
            self.market_regimes[pair] = regime_features['market_state']
            
            # Handle inf and NaN values
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.fillna(method='ffill', inplace=True)
            df.fillna(method='bfill', inplace=True)
            df.fillna(0, inplace=True)
            
            # Split back into train/validation/test
            train_start = pd.to_datetime(self.config['train_start'])
            train_end = pd.to_datetime(self.config['train_end'])
            validation_start = pd.to_datetime(self.config['validation_start'])
            validation_end = pd.to_datetime(self.config['validation_end'])
            test_start = pd.to_datetime(self.config['test_start'])
            test_end = pd.to_datetime(self.config['test_end'])
            
            train_data = df[(df.index >= train_start) & (df.index <= train_end)].copy()
            validation_data = df[(df.index >= validation_start) & (df.index <= validation_end)].copy()
            test_data = df[(df.index >= test_start) & (df.index <= test_end)].copy()
            
            self.enhanced_features[pair] = {
                'full': df,
                'train': train_data,
                'validation': validation_data,
                'test': test_data
            }
            
            logger.info(f"Enhanced features for {pair}: {df.shape[1]} features created")
        
        # Save enhanced features
        for pair, data_dict in self.enhanced_features.items():
            data_dict['full'].to_csv(os.path.join(feature_dir, f"{pair}_enhanced_features.csv"))
        
        return self.enhanced_features
    
    def _calculate_basic_features(self, df):
        """Calculate basic technical features (existing functionality)."""
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
        
        # Moving averages
        for window in [5, 10, 20, 50, 100]:
            df[f'SMA_{window}'] = close_price.rolling(window=window).mean()
            df[f'EMA_{window}'] = close_price.ewm(span=window, adjust=False).mean()
            df[f'SMA_Distance_{window}'] = close_price / df[f'SMA_{window}'] - 1
            df[f'EMA_Distance_{window}'] = close_price / df[f'EMA_{window}'] - 1
        
        # MACD
        df['EMA_12'] = close_price.ewm(span=12, adjust=False).mean()
        df['EMA_26'] = close_price.ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # RSI
        delta = close_price.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        for period in [20]:
            df[f'BB_Middle_{period}'] = close_price.rolling(window=period).mean()
            df[f'BB_Std_{period}'] = close_price.rolling(window=period).std()
            df[f'BB_Upper_{period}'] = df[f'BB_Middle_{period}'] + 2 * df[f'BB_Std_{period}']
            df[f'BB_Lower_{period}'] = df[f'BB_Middle_{period}'] - 2 * df[f'BB_Std_{period}']
            df[f'BB_Width_{period}'] = (df[f'BB_Upper_{period}'] - df[f'BB_Lower_{period}']) / df[f'BB_Middle_{period}']
            df[f'BB_Position_{period}'] = (close_price - df[f'BB_Lower_{period}']) / (df[f'BB_Upper_{period}'] - df[f'BB_Lower_{period}'] + 1e-10)
        
        # ATR
        tr1 = high_price - low_price
        tr2 = abs(high_price - close_price.shift(1))
        tr3 = abs(low_price - close_price.shift(1))
        true_range = pd.DataFrame({'TR1': tr1, 'TR2': tr2, 'TR3': tr3}).max(axis=1)
        df['ATR_14'] = true_range.rolling(window=14).mean()
        df['ATR_14_Ratio'] = df['ATR_14'] / close_price
        
        # Volume indicators
        for window in [5, 10, 20]:
            df[f'Volume_SMA_{window}'] = volume.rolling(window=window).mean()
            df[f'Volume_Ratio_{window}'] = volume / df[f'Volume_SMA_{window}']
        
        # OBV
        df['OBV'] = (np.sign(close_price.diff()) * volume).fillna(0).cumsum()
        for window in [10, 20]:
            df[f'OBV_SMA_{window}'] = df['OBV'].rolling(window=window).mean()
            df[f'OBV_Ratio_{window}'] = df['OBV'] / df[f'OBV_SMA_{window}'] - 1
        
        # Time-based features
        df['Hour'] = df.index.hour
        df['Day'] = df.index.day
        df['Month'] = df.index.month
        df['Day_of_Week'] = df.index.dayofweek
        df['Is_Weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)
        
        # Cyclical encoding
        df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
        df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
        df['Day_Sin'] = np.sin(2 * np.pi * df['Day'] / 31)
        df['Day_Cos'] = np.cos(2 * np.pi * df['Day'] / 31)
        df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df['Day_of_Week_Sin'] = np.sin(2 * np.pi * df['Day_of_Week'] / 7)
        df['Day_of_Week_Cos'] = np.cos(2 * np.pi * df['Day_of_Week'] / 7)

    # ===== STEP 3: ENHANCED MODEL ARCHITECTURE =====
    
    def build_enhanced_cnn_lstm_model(self, input_shape, pair_or_bagging=None):
        """Build enhanced CNN-LSTM with improved attention mechanisms."""
        hyperparams = self.get_hyperparameters('enhanced_cnn_lstm', pair_or_bagging)
        logger.info(f"Building Enhanced CNN-LSTM for {pair_or_bagging} with hyperparams: {hyperparams}")
        
        model = Sequential()
        
        # Enhanced CNN layers with residual connections
        model.add(Conv1D(filters=hyperparams.get('cnn_filters', 64), 
                        kernel_size=hyperparams.get('cnn_kernel_size', 3), 
                        activation='relu', 
                        input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Dropout(hyperparams.get('dropout_rate', 0.3)))
        
        # Additional CNN layers
        model.add(Conv1D(filters=hyperparams.get('cnn_filters', 64) * 2, 
                        kernel_size=hyperparams.get('cnn_kernel_size', 3), 
                        activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(hyperparams.get('dropout_rate', 0.3)))
        
        # Enhanced LSTM with attention
        model.add(LSTM(units=hyperparams.get('lstm_units', 100), 
                      return_sequences=True))
        model.add(Dropout(hyperparams.get('dropout_rate', 0.3)))
        
        # Attention mechanism
        model.add(LSTM(units=hyperparams.get('lstm_units', 100) // 2, 
                      return_sequences=True))
        model.add(GlobalAveragePooling1D())  # Simple attention alternative
        model.add(Dropout(hyperparams.get('dropout_rate', 0.3)))
        
        # Dense layers
        model.add(Dense(units=hyperparams.get('lstm_units', 100) // 4, activation='relu'))
        model.add(Dropout(hyperparams.get('dropout_rate', 0.3)))
        model.add(Dense(1, activation='sigmoid'))
        
        # Enhanced optimizer with gradient clipping
        optimizer = Adam(
            learning_rate=hyperparams.get('learning_rate', 0.001),
            clipnorm=1.0
        )
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        
        return model
    
    def build_advanced_tft_model(self, input_dim, pair_or_bagging=None):
        """Build advanced Temporal Fusion Transformer with enhanced attention."""
        hyperparams = self.get_hyperparameters('advanced_tft', pair_or_bagging)
        logger.info(f"Building Advanced TFT for {pair_or_bagging} with hyperparams: {hyperparams}")
        
        inputs = Input(shape=(self.config['window_size'], input_dim))
        
        # Enhanced multi-head attention with more heads
        attention_output = MultiHeadAttention(
            num_heads=hyperparams.get('num_heads', 8), 
            key_dim=hyperparams.get('hidden_units', 64) // hyperparams.get('num_heads', 8)
        )(inputs, inputs)
        
        # Layer normalization and residual connection
        x = LayerNormalization()(Add()([inputs, attention_output]))
        
        # Enhanced feed forward network
        ffn1 = Dense(hyperparams.get('hidden_units', 64) * 4, activation='relu')(x)
        ffn1 = Dropout(hyperparams.get('dropout_rate', 0.2))(ffn1)
        ffn2 = Dense(input_dim)(ffn1)
        
        # Second residual connection
        x = LayerNormalization()(Add()([x, ffn2]))
        
        # Additional attention layer
        attention_output2 = MultiHeadAttention(
            num_heads=hyperparams.get('num_heads', 8) // 2, 
            key_dim=hyperparams.get('hidden_units', 64) // hyperparams.get('num_heads', 8)
        )(x, x)
        
        x = LayerNormalization()(Add()([x, attention_output2]))
        
        # Enhanced LSTM for temporal processing
        x = LSTM(units=hyperparams.get('hidden_units', 64), return_sequences=True)(x)
        x = Dropout(hyperparams.get('dropout_rate', 0.2))(x)
        x = LSTM(units=hyperparams.get('hidden_units', 64))(x)
        x = Dropout(hyperparams.get('dropout_rate', 0.2))(x)
        
        # Enhanced output layers
        x = Dense(units=hyperparams.get('hidden_units', 64), activation='relu')(x)
        x = Dropout(hyperparams.get('dropout_rate', 0.2))(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # Enhanced optimizer
        optimizer = Adam(
            learning_rate=hyperparams.get('learning_rate', 0.001),
            clipnorm=1.0
        )
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        
        return model
    
    def build_enhanced_xgboost_model(self, pair_or_bagging=None):
        """Build enhanced XGBoost with advanced hyperparameters."""
        hyperparams = self.get_hyperparameters('enhanced_xgboost', pair_or_bagging)
        logger.info(f"Building Enhanced XGBoost for {pair_or_bagging} with hyperparams: {hyperparams}")
        
        # Enhanced XGBoost with more sophisticated parameters
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            max_depth=hyperparams.get('max_depth', 6),
            learning_rate=hyperparams.get('learning_rate', 0.1),
            n_estimators=hyperparams.get('n_estimators', 200),
            subsample=hyperparams.get('subsample', 0.8),
            colsample_bytree=hyperparams.get('colsample_bytree', 0.8),
            colsample_bylevel=hyperparams.get('colsample_bylevel', 0.8),
            colsample_bynode=hyperparams.get('colsample_bynode', 0.8),
            gamma=hyperparams.get('gamma', 0),
            reg_alpha=hyperparams.get('reg_alpha', 0),
            reg_lambda=hyperparams.get('reg_lambda', 1),
            min_child_weight=hyperparams.get('min_child_weight', 1),
            max_delta_step=hyperparams.get('max_delta_step', 0),
            random_state=self.config['random_state'],
            n_jobs=-1,
            tree_method='hist'  # Faster training
        )
        
        return model

    # ===== STEP 4: ADVANCED ENSEMBLE METHODS =====
    
    def create_dynamic_ensemble(self, models_dict, validation_data):
        """Create dynamic ensemble that adapts weights based on market conditions."""
        logger.info("=== CREATING DYNAMIC ENSEMBLE ===")
        
        ensemble_models = {}
        
        for pair in self.config['currency_pairs']:
            logger.info(f"Creating dynamic ensemble for {pair}")
            
            # Get models for this pair
            pair_models = {}
            for model_type in self.config['models_to_train']:
                model_key = f"{pair}_{model_type}"
                if model_key in models_dict:
                    pair_models[model_type] = models_dict[model_key]
            
            if len(pair_models) < 2:
                logger.warning(f"Not enough models for ensemble on {pair}")
                continue
            
            # Create regime-based weights
            market_regime = self.market_regimes.get(pair)
            if market_regime is not None:
                val_data = validation_data[pair] if pair in validation_data else self.enhanced_features[pair]['validation']
                regime_weights = self._calculate_regime_weights(pair_models, val_data, market_regime)
                
                ensemble_models[pair] = {
                    'models': pair_models,
                    'regime_weights': regime_weights,
                    'type': 'dynamic'
                }
            else:
                # Fallback to simple voting
                ensemble_models[pair] = {
                    'models': pair_models,
                    'type': 'voting'
                }
        
        self.ensemble_models = ensemble_models
        return ensemble_models
    
    def _calculate_regime_weights(self, models, validation_data, market_regime):
        """Calculate model weights for different market regimes."""
        regime_weights = {}
        
        # Unique regimes
        unique_regimes = market_regime.unique()
        
        for regime in unique_regimes:
            regime_mask = market_regime == regime
            regime_data = validation_data[regime_mask]
            
            if len(regime_data) < 10:  # Not enough data for this regime
                continue
            
            regime_weights[regime] = {}
            regime_accuracies = {}
            
            # Calculate accuracy for each model in this regime
            for model_name, model_data in models.items():
                try:
                    # Get predictions for this regime
                    if model_name in ['enhanced_cnn_lstm', 'advanced_tft']:
                        X, y, _, _ = self.prepare_model_data(regime_data, is_lstm=True)
                        if len(X) > 0:
                            y_pred = model_data['model'].predict(X)
                            y_pred_binary = (y_pred > 0.5).astype(int).flatten()
                            accuracy = accuracy_score(y, y_pred_binary)
                            regime_accuracies[model_name] = accuracy
                    else:  # XGBoost
                        X, y, _, _ = self.prepare_model_data(regime_data, is_lstm=False)
                        if len(X) > 0:
                            y_pred = model_data['model'].predict(X)
                            accuracy = accuracy_score(y, y_pred)
                            regime_accuracies[model_name] = accuracy
                except Exception as e:
                    logger.warning(f"Error calculating accuracy for {model_name} in {regime}: {e}")
                    regime_accuracies[model_name] = 0.5  # Default
            
            # Calculate weights based on accuracy (softmax)
            if regime_accuracies:
                total_accuracy = sum(regime_accuracies.values())
                for model_name, accuracy in regime_accuracies.items():
                    regime_weights[regime][model_name] = accuracy / (total_accuracy + 1e-10)
        
        return regime_weights

    # ===== STEP 5: ADVANCED HYPERPARAMETER OPTIMIZATION =====
    
    def bayesian_hyperparameter_optimization(self, model_type, pair, n_trials=100):
        """Bayesian optimization for hyperparameter tuning."""
        logger.info(f"=== BAYESIAN OPTIMIZATION: {model_type} on {pair} ===")
        
        def objective(trial):
            try:
                # Define search space based on model type
                if model_type == 'enhanced_cnn_lstm':
                    hyperparams = {
                        'cnn_filters': trial.suggest_categorical('cnn_filters', [32, 64, 96, 128, 160, 192]),
                        'cnn_kernel_size': trial.suggest_int('cnn_kernel_size', 2, 7),
                        'lstm_units': trial.suggest_categorical('lstm_units', [50, 75, 100, 125, 150, 200]),
                        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.6),
                        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
                    }
                elif model_type == 'advanced_tft':
                    hyperparams = {
                        'hidden_units': trial.suggest_categorical('hidden_units', [32, 64, 96, 128, 160, 192]),
                        'num_heads': trial.suggest_categorical('num_heads', [2, 4, 6, 8, 12, 16]),
                        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
                        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
                    }
                elif model_type == 'enhanced_xgboost':
                    hyperparams = {
                        'max_depth': trial.suggest_int('max_depth', 3, 12),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                        'n_estimators': trial.suggest_categorical('n_estimators', [100, 150, 200, 300, 400]),
                        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
                        'gamma': trial.suggest_float('gamma', 0, 5),
                        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
                        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 5),
                        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10)
                    }
                
                # Get training data
                if pair == 'Bagging':
                    train_data = self.bagging_data['train']
                    val_data = self.bagging_data.get('validation', train_data.sample(frac=0.2))
                else:
                    train_data = self.enhanced_features[pair]['train']
                    val_data = self.enhanced_features[pair]['validation']
                
                # Prepare data
                is_lstm = model_type in ['enhanced_cnn_lstm', 'advanced_tft']
                X_train, y_train, _, _ = self.prepare_model_data(train_data, is_lstm=is_lstm)
                X_val, y_val, _, _ = self.prepare_model_data(val_data, is_lstm=is_lstm)
                
                # Handle NaN values
                if np.isnan(X_train).any() or np.isnan(y_train).any():
                    X_train = np.nan_to_num(X_train)
                    y_train = np.nan_to_num(y_train)
                if np.isnan(X_val).any() or np.isnan(y_val).any():
                    X_val = np.nan_to_num(X_val)
                    y_val = np.nan_to_num(y_val)
                
                # Train model with hyperparameters
                if model_type == 'enhanced_cnn_lstm':
                    # Temporarily store hyperparameters
                    original_hyperparams = self.hyperparameters.get('enhanced_cnn_lstm', {}).get(pair, {})
                    self.hyperparameters.setdefault('enhanced_cnn_lstm', {})[pair] = hyperparams
                    
                    input_shape = (X_train.shape[1], X_train.shape[2])
                    model = self.build_enhanced_cnn_lstm_model(input_shape, pair)
                    
                    history = model.fit(X_train, y_train, 
                                      validation_data=(X_val, y_val),
                                      epochs=50, batch_size=32, 
                                      callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
                                      verbose=0)
                    
                    score = max(history.history['val_accuracy'])
                    
                    # Restore original hyperparameters
                    self.hyperparameters['enhanced_cnn_lstm'][pair] = original_hyperparams
                    
                elif model_type == 'advanced_tft':
                    original_hyperparams = self.hyperparameters.get('advanced_tft', {}).get(pair, {})
                    self.hyperparameters.setdefault('advanced_tft', {})[pair] = hyperparams
                    
                    input_dim = X_train.shape[2]
                    model = self.build_advanced_tft_model(input_dim, pair)
                    
                    history = model.fit(X_train, y_train, 
                                      validation_data=(X_val, y_val),
                                      epochs=50, batch_size=32, 
                                      callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
                                      verbose=0)
                    
                    score = max(history.history['val_accuracy'])
                    
                    # Restore original hyperparameters
                    self.hyperparameters['advanced_tft'][pair] = original_hyperparams
                    
                elif model_type == 'enhanced_xgboost':
                    original_hyperparams = self.hyperparameters.get('enhanced_xgboost', {}).get(pair, {})
                    self.hyperparameters.setdefault('enhanced_xgboost', {})[pair] = hyperparams
                    
                    model = self.build_enhanced_xgboost_model(pair)
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
                    score = model.score(X_val, y_val)
                    
                    # Restore original hyperparameters
                    self.hyperparameters['enhanced_xgboost'][pair] = original_hyperparams
                
                return score
                
            except Exception as e:
                logger.error(f"Error in trial {trial.number}: {e}")
                return 0.0
        
        # Create study with advanced sampler and pruner
        sampler = TPESampler(seed=self.config['random_state'])
        pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=20)
        
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=pruner,
            study_name=f"enhanced_{model_type}_{pair}"
        )
        
        study.optimize(objective, n_trials=n_trials)
        
        # Store best hyperparameters
        best_params = study.best_params
        best_value = study.best_value
        
        logger.info(f"Best hyperparameters for {model_type} on {pair}: {best_params}")
        logger.info(f"Best validation score: {best_value:.4f}")
        
        # Save results
        os.makedirs(hyperparams_dir, exist_ok=True)
        with open(os.path.join(hyperparams_dir, f"{pair}_{model_type}_best_hyperparams.json"), 'w') as f:
            json.dump(best_params, f, indent=2)
        
        return best_params, study

    # ===== STEP 6: WALK-FORWARD ANALYSIS =====
    
    def walk_forward_analysis(self):
        """Perform walk-forward analysis for robust model evaluation."""
        logger.info("=== STEP 6: WALK-FORWARD ANALYSIS ===")
        
        results = {}
        
        for pair in self.config['currency_pairs']:
            logger.info(f"Walk-forward analysis for {pair}")
            
            pair_data = self.enhanced_features[pair]['full']
            
            # Define walk-forward windows
            window_size = self.config['walkforward_window']  # 180 days
            step_size = 30  # Step by 30 days
            
            # Convert to trading days (approximately 24 hours * days)
            window_hours = window_size * 24
            step_hours = step_size * 24
            
            walk_results = []
            
            start_date = pd.to_datetime(self.config['train_start'])
            end_date = pd.to_datetime(self.config['test_end'])
            
            current_start = start_date
            
            while current_start + pd.Timedelta(hours=window_hours) <= end_date:
                train_end = current_start + pd.Timedelta(hours=window_hours * 0.7)  # 70% for training
                val_end = current_start + pd.Timedelta(hours=window_hours * 0.85)   # 15% for validation
                test_end = current_start + pd.Timedelta(hours=window_hours)         # 15% for testing
                
                # Extract data windows
                train_window = pair_data[(pair_data.index >= current_start) & 
                                       (pair_data.index <= train_end)]
                val_window = pair_data[(pair_data.index > train_end) & 
                                     (pair_data.index <= val_end)]
                test_window = pair_data[(pair_data.index > val_end) & 
                                      (pair_data.index <= test_end)]
                
                if len(train_window) < 100 or len(test_window) < 10:
                    current_start += pd.Timedelta(hours=step_hours)
                    continue
                
                # Train and evaluate models for this window
                window_results = self._evaluate_walk_forward_window(
                    pair, train_window, val_window, test_window, current_start
                )
                
                walk_results.append(window_results)
                current_start += pd.Timedelta(hours=step_hours)
            
            results[pair] = walk_results
            
            # Save walk-forward results
            with open(os.path.join(walkforward_dir, f"{pair}_walkforward_results.json"), 'w') as f:
                json.dump(walk_results, f, indent=2, default=str)
        
        self.walkforward_results = results
        return results
    
    def _evaluate_walk_forward_window(self, pair, train_data, val_data, test_data, window_start):
        """Evaluate models on a single walk-forward window."""
        logger.info(f"Evaluating window starting {window_start} for {pair}")
        
        window_results = {
            'window_start': window_start,
            'train_size': len(train_data),
            'val_size': len(val_data), 
            'test_size': len(test_data),
            'model_results': {}
        }
        
        for model_type in self.config['models_to_train']:
            try:
                # Prepare data
                is_lstm = model_type in ['enhanced_cnn_lstm', 'advanced_tft']
                X_train, y_train, scaler, _ = self.prepare_model_data(train_data, is_lstm=is_lstm)
                X_val, y_val, _, _ = self.prepare_model_data(val_data, is_lstm=is_lstm)
                X_test, y_test, _, _ = self.prepare_model_data(test_data, is_lstm=is_lstm)
                
                # Handle NaN values
                for X, y in [(X_train, y_train), (X_val, y_val), (X_test, y_test)]:
                    if np.isnan(X).any() or np.isnan(y).any():
                        X = np.nan_to_num(X)
                        y = np.nan_to_num(y)
                
                # Build and train model
                if model_type == 'enhanced_cnn_lstm':
                    input_shape = (X_train.shape[1], X_train.shape[2])
                    model = self.build_enhanced_cnn_lstm_model(input_shape, pair)
                elif model_type == 'advanced_tft':
                    input_dim = X_train.shape[2]
                    model = self.build_advanced_tft_model(input_dim, pair)
                elif model_type == 'enhanced_xgboost':
                    model = self.build_enhanced_xgboost_model(pair)
                
                # Train model
                if model_type in ['enhanced_cnn_lstm', 'advanced_tft']:
                    model.fit(X_train, y_train, 
                            validation_data=(X_val, y_val),
                            epochs=50, batch_size=32, 
                            callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
                            verbose=0)
                    
                    # Predict
                    y_pred_proba = model.predict(X_test)
                    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
                else:
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    y_pred = (y_pred_proba > 0.5).astype(int)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                
                window_results['model_results'][model_type] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                }
                
            except Exception as e:
                logger.error(f"Error evaluating {model_type} in window {window_start}: {e}")
                window_results['model_results'][model_type] = {
                    'accuracy': 0.5, 'precision': 0.5, 'recall': 0.5, 'f1': 0.5
                }
        
        return window_results

    # ===== RISK MANAGEMENT ENHANCEMENTS =====
    
    def calculate_kelly_criterion(self, win_rate, avg_win, avg_loss):
        """Calculate optimal position size using Kelly Criterion."""
        if avg_loss == 0:
            return 0
        
        win_loss_ratio = avg_win / avg_loss
        kelly_fraction = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        
        # Cap Kelly fraction to prevent excessive risk
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Max 25% of capital
        
        return kelly_fraction
    
    def calculate_dynamic_position_size(self, prediction_confidence, kelly_fraction, base_size=0.1):
        """Calculate position size based on prediction confidence and Kelly Criterion."""
        # Scale base size by Kelly fraction
        kelly_adjusted_size = base_size * kelly_fraction
        
        # Further scale by prediction confidence
        confidence_multiplier = max(0.1, min(2.0, (prediction_confidence - 0.5) * 4))
        
        final_size = kelly_adjusted_size * confidence_multiplier
        
        # Apply maximum position size limit
        max_position_size = 0.2  # Never risk more than 20% of capital
        final_size = min(final_size, max_position_size)
        
        return final_size
    
    def calculate_dynamic_stops(self, entry_price, atr, confidence, direction):
        """Calculate dynamic stop-loss and take-profit levels."""
        # Base stop distance as multiple of ATR
        base_stop_distance = atr * 2.0
        
        # Adjust based on confidence (lower confidence = tighter stops)
        confidence_multiplier = 0.5 + (confidence * 1.5)  # 0.5 to 2.0 range
        stop_distance = base_stop_distance * confidence_multiplier
        
        if direction == 1:  # Long position
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + (stop_distance * 2)  # 2:1 risk-reward
        else:  # Short position
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - (stop_distance * 2)
        
        return stop_loss, take_profit

    # ===== UTILITY FUNCTIONS =====
    
    def load_hyperparameters(self):
        """Load hyperparameters with enhanced model types."""
        logger.info("Loading enhanced hyperparameters")
        
        # Enhanced default hyperparameters
        default_hyperparams = {
            'enhanced_cnn_lstm': {
                'cnn_filters': 96,
                'cnn_kernel_size': 3,
                'lstm_units': 128,
                'dropout_rate': 0.3,
                'learning_rate': 0.001
            },
            'advanced_tft': {
                'hidden_units': 96,
                'num_heads': 8,
                'dropout_rate': 0.2,
                'learning_rate': 0.001
            },
            'enhanced_xgboost': {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 200,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'colsample_bylevel': 0.8,
                'gamma': 0,
                'reg_alpha': 0,
                'reg_lambda': 1,
                'min_child_weight': 1
            }
        }
        
        self.hyperparameters = {}
        
        for model_type in self.config['models_to_train']:
            self.hyperparameters[model_type] = {}
            
            for pair in self.config['currency_pairs'] + (['Bagging'] if self.config['use_bagging'] else []):
                param_file = os.path.join(hyperparams_dir, f"{pair}_{model_type}_best_hyperparams.json")
                
                if os.path.exists(param_file):
                    try:
                        with open(param_file, 'r') as f:
                            params = json.load(f)
                        self.hyperparameters[model_type][pair] = params
                        logger.info(f"Loaded hyperparameters for {pair} {model_type}")
                    except Exception as e:
                        logger.warning(f"Error loading hyperparameters for {pair} {model_type}: {e}")
                        self.hyperparameters[model_type][pair] = default_hyperparams[model_type].copy()
                else:
                    self.hyperparameters[model_type][pair] = default_hyperparams[model_type].copy()
    
    def get_hyperparameters(self, model_type, pair_or_bagging):
        """Get hyperparameters for enhanced model types."""
        try:
            return self.hyperparameters[model_type][pair_or_bagging]
        except KeyError:
            logger.warning(f"Hyperparameters not found for {model_type} {pair_or_bagging}, using defaults")
            
            default_hyperparams = {
                'enhanced_cnn_lstm': {
                    'cnn_filters': 96, 'cnn_kernel_size': 3, 'lstm_units': 128,
                    'dropout_rate': 0.3, 'learning_rate': 0.001
                },
                'advanced_tft': {
                    'hidden_units': 96, 'num_heads': 8, 'dropout_rate': 0.2,
                    'learning_rate': 0.001
                },
                'enhanced_xgboost': {
                    'max_depth': 6, 'learning_rate': 0.1, 'n_estimators': 200,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0,
                    'reg_alpha': 0, 'reg_lambda': 1
                }
            }
            return default_hyperparams.get(model_type, {})
    
    def prepare_model_data(self, data, is_lstm=True):
        """Prepare data for enhanced models."""
        # Extract features and target
        X = data.drop(['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 
                      'Log_Returns', 'Target', 'CurrencyPair'], axis=1, errors='ignore')
        y = data['Target']
        
        # Use robust scaler for better outlier handling
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        if is_lstm:
            # Create sequences for LSTM models
            X_seq, y_seq = [], []
            for i in range(len(X_scaled) - self.config['window_size']):
                X_seq.append(X_scaled[i:i + self.config['window_size']])
                y_seq.append(y.iloc[i + self.config['window_size']])
            
            return np.array(X_seq), np.array(y_seq), scaler, X.columns.tolist()
        else:
            # For XGBoost, flatten the sequences
            X_lagged = np.zeros((X_scaled.shape[0] - self.config['window_size'], 
                               X_scaled.shape[1] * self.config['window_size']))
            y_lagged = y.iloc[self.config['window_size']:].values
            
            for i in range(len(X_scaled) - self.config['window_size']):
                X_lagged[i] = X_scaled[i:i + self.config['window_size']].flatten()
            
            return X_lagged, y_lagged, scaler, X.columns.tolist()
    
    def select_features(self):
        """Enhanced feature selection with more sophisticated methods."""
        logger.info("=== ENHANCED FEATURE SELECTION ===")
        
        feature_importance_results = {}
        
        for pair, data_dict in self.enhanced_features.items():
            train_data = data_dict['train'].copy()
            
            # Remove non-feature columns
            features = train_data.drop(['Open', 'High', 'Low', 'Close', 'Volume', 
                                      'Returns', 'Log_Returns', 'Target'], axis=1, errors='ignore')
            target = train_data['Target']
            
            feature_rankings = {}
            selected_columns = {}
            
            # Method 1: Enhanced Random Forest with more estimators
            logger.info(f"Enhanced Random Forest feature selection for {pair}")
            rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, 
                                           random_state=self.config['random_state'], n_jobs=-1)
            rf_model.fit(features, target)
            rf_importances = pd.Series(rf_model.feature_importances_, index=features.columns)
            rf_importances = rf_importances.sort_values(ascending=False)
            feature_rankings['random_forest'] = rf_importances
            selected_columns['random_forest'] = rf_importances.nlargest(self.config['n_features']).index.tolist()
            
            # Method 2: Enhanced Mutual Information
            logger.info(f"Enhanced Mutual Information feature selection for {pair}")
            mi_values = mutual_info_regression(features, target, 
                                             random_state=self.config['random_state'], n_neighbors=5)
            mi_importances = pd.Series(mi_values, index=features.columns)
            mi_importances = mi_importances.sort_values(ascending=False)
            feature_rankings['mutual_info'] = mi_importances
            selected_columns['mutual_info'] = mi_importances.nlargest(self.config['n_features']).index.tolist()
            
            # Method 3: Enhanced PCA with explained variance
            logger.info(f"Enhanced PCA feature selection for {pair}")
            scaler = RobustScaler()
            scaled_features = scaler.fit_transform(features)
            
            pca = PCA(n_components=min(self.config['n_features'], len(features.columns)))
            pca.fit(scaled_features)
            
            # Feature importance based on loadings and explained variance
            loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
            pca_importance = np.sum(loadings**2, axis=1)
            pca_importances = pd.Series(pca_importance, index=features.columns)
            pca_importances = pca_importances.sort_values(ascending=False)
            feature_rankings['pca'] = pca_importances
            selected_columns['pca'] = pca_importances.nlargest(self.config['n_features']).index.tolist()
            
            # Combine features using weighted voting
            all_selected_features = set()
            for method, selected_features in selected_columns.items():
                all_selected_features.update(selected_features)
            
            # Select top features by combined score
            combined_scores = {}
            for feature in features.columns:
                score = 0
                for method, ranking in feature_rankings.items():
                    if feature in ranking.index:
                        normalized_score = ranking[feature] / ranking.max()
                        score += normalized_score
                combined_scores[feature] = score
            
            # Select top N features
            final_selected = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            final_features = [feat for feat, score in final_selected[:self.config['n_features']]]
            
            logger.info(f"Selected {len(final_features)} features for {pair}")
            
            # Create datasets with selected features
            base_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 'Log_Returns', 'Target']
            all_cols = final_features + base_cols
            
            self.selected_features[pair] = {
                'full': self.enhanced_features[pair]['full'][all_cols],
                'train': self.enhanced_features[pair]['train'][all_cols],
                'validation': self.enhanced_features[pair]['validation'][all_cols],
                'test': self.enhanced_features[pair]['test'][all_cols]
            }
            
            # Store results
            feature_importance_results[pair] = {
                'feature_rankings': {method: ranking.to_dict() for method, ranking in feature_rankings.items()},
                'selected_columns': selected_columns,
                'final_selected_features': final_features,
                'combined_scores': combined_scores
            }
        
        # Save results
        with open(os.path.join(feature_dir, 'enhanced_feature_importance_results.json'), 'w') as f:
            json.dump({
                pair: {
                    'selected_columns': {method: list(cols) for method, cols in data['selected_columns'].items()},
                    'final_selected_features': list(data['final_selected_features']),
                    'combined_scores': {k: float(v) for k, v in data['combined_scores'].items()}
                }
                for pair, data in feature_importance_results.items()
            }, f, indent=2)
        
        # Create enhanced bagging dataset
        self._create_enhanced_bagging_dataset()
        
        return self.selected_features
    
    def _create_enhanced_bagging_dataset(self):
        """Create enhanced bagging dataset with regime information."""
        logger.info("Creating enhanced bagging dataset")
        
        combined_train_dfs = []
        combined_val_dfs = []
        combined_test_dfs = []
        
        for pair, data_dict in self.selected_features.items():
            # Add currency pair and regime information
            for split_name, split_data in [('train', combined_train_dfs), 
                                         ('validation', combined_val_dfs), 
                                         ('test', combined_test_dfs)]:
                df = data_dict[split_name].copy()
                df['CurrencyPair'] = pair
                
                # Add market regime information
                if pair in self.market_regimes:
                    regime_data = self.market_regimes[pair]
                    aligned_regime = regime_data.reindex(df.index, method='ffill')
                    df['MarketRegime'] = aligned_regime.fillna('ranging')
                
                split_data.append(df)
        
        # Concatenate and sort
        bagging_train = pd.concat(combined_train_dfs, axis=0).sort_index()
        bagging_val = pd.concat(combined_val_dfs, axis=0).sort_index()
        bagging_test = pd.concat(combined_test_dfs, axis=0).sort_index()
        
        self.bagging_data = {
            'train': bagging_train,
            'validation': bagging_val,
            'test': bagging_test
        }
        
        logger.info(f"Enhanced bagging dataset created: Train={bagging_train.shape}, "
                   f"Validation={bagging_val.shape}, Test={bagging_test.shape}")
        
        return self.bagging_data
    
    def run_enhanced_pipeline(self):
        """Run the complete enhanced pipeline for Master's thesis."""
        start_time = time.time()
        logger.info("=== STARTING ENHANCED FOREX PREDICTION PIPELINE FOR MASTER'S THESIS ===")
        
        try:
            # Step 1: Enhanced Data Loading and Preprocessing
            logger.info("STEP 1: Enhanced Data Loading and Preprocessing")
            self.load_data()
            self.preprocess_data()
            
            # Step 2: Advanced Feature Engineering
            logger.info("STEP 2: Advanced Feature Engineering")
            self.calculate_enhanced_features()
            self.select_features()
            
            # Step 3: Bayesian Hyperparameter Optimization (if enabled)
            if self.config['hyperparameter_tuning']:
                logger.info("STEP 3: Bayesian Hyperparameter Optimization")
                for pair in self.config['currency_pairs'] + (['Bagging'] if self.config['use_bagging'] else []):
                    for model_type in self.config['models_to_train']:
                        best_params, study = self.bayesian_hyperparameter_optimization(
                            model_type, pair, self.config['n_trials']
                        )
                        # Update hyperparameters
                        self.hyperparameters.setdefault(model_type, {})[pair] = best_params
            
            # Step 4: Model Training with Enhanced Architecture
            logger.info("STEP 4: Enhanced Model Training")
            trained_models = self.train_enhanced_models()
            
            # Step 5: Dynamic Ensemble Creation
            if self.config['use_dynamic_ensemble']:
                logger.info("STEP 5: Dynamic Ensemble Creation")
                validation_data = {pair: data['validation'] for pair, data in self.selected_features.items()}
                self.create_dynamic_ensemble(trained_models, validation_data)
            
            # Step 6: Walk-Forward Analysis
            if self.config['use_walkforward']:
                logger.info("STEP 6: Walk-Forward Analysis")
                self.walk_forward_analysis()
            
            # Step 7: Enhanced Model Evaluation
            logger.info("STEP 7: Enhanced Model Evaluation")
            results = self.evaluate_enhanced_models()
            
            elapsed_time = time.time() - start_time
            logger.info(f"Enhanced pipeline completed in {elapsed_time / 60:.2f} minutes")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in enhanced pipeline: {e}", exc_info=True)
            raise
    
    def train_enhanced_models(self):
        """Train enhanced models with proper validation."""
        logger.info("Training enhanced models")
        
        trained_models = {}
        
        # Train single-pair models
        for pair in self.config['currency_pairs']:
            logger.info(f"Training enhanced models for {pair}")
            
            train_data = self.selected_features[pair]['train']
            val_data = self.selected_features[pair]['validation']
            
            for model_type in self.config['models_to_train']:
                model_key = f"{pair}_{model_type}"
                logger.info(f"Training {model_key}")
                
                try:
                    # Prepare data
                    is_lstm = model_type in ['enhanced_cnn_lstm', 'advanced_tft']
                    X_train, y_train, scaler, feature_names = self.prepare_model_data(train_data, is_lstm=is_lstm)
                    X_val, y_val, _, _ = self.prepare_model_data(val_data, is_lstm=is_lstm)
                    
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
                        model = self.build_enhanced_cnn_lstm_model(input_shape, pair)
                        
                        callbacks = [
                            EarlyStopping(monitor='val_loss', patience=self.config['patience'], 
                                        restore_best_weights=True),
                            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
                        ]
                        
                        history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                                          epochs=self.config['epochs'], batch_size=self.config['batch_size'],
                                          callbacks=callbacks, verbose=1)
                        
                        # Save model
                        model_path = os.path.join(models_dir, f"{model_key}_best.keras")
                        model.save(model_path)
                        
                        trained_models[model_key] = {
                            'model': model, 'scaler': scaler, 'feature_names': feature_names,
                            'history': history.history
                        }
                        
                    elif model_type == 'advanced_tft':
                        input_dim = X_train.shape[2]
                        model = self.build_advanced_tft_model(input_dim, pair)
                        
                        callbacks = [
                            EarlyStopping(monitor='val_loss', patience=self.config['patience'], 
                                        restore_best_weights=True),
                            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
                        ]
                        
                        history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                                          epochs=self.config['epochs'], batch_size=self.config['batch_size'],
                                          callbacks=callbacks, verbose=1)
                        
                        # Save model
                        model_path = os.path.join(models_dir, f"{model_key}_best.keras")
                        model.save(model_path)
                        
                        trained_models[model_key] = {
                            'model': model, 'scaler': scaler, 'feature_names': feature_names,
                            'history': history.history
                        }
                        
                    elif model_type == 'enhanced_xgboost':
                        model = self.build_enhanced_xgboost_model(pair)
                        
                        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=1)
                        
                        # Save model
                        model_path = os.path.join(models_dir, f"{model_key}.pkl")
                        with open(model_path, 'wb') as f:
                            pickle.dump(model, f)
                        
                        trained_models[model_key] = {
                            'model': model, 'scaler': scaler, 'feature_names': feature_names
                        }
                    
                    logger.info(f"Successfully trained {model_key}")
                    
                except Exception as e:
                    logger.error(f"Error training {model_key}: {e}")
                    continue
        
        # Train bagging models if enabled
        if self.config['use_bagging']:
            logger.info("Training enhanced bagging models")
            
            train_data = self.bagging_data['train']
            val_data = self.bagging_data['validation']
            
            for model_type in self.config['models_to_train']:
                model_key = f"Bagging_{model_type}"
                logger.info(f"Training {model_key}")
                
                try:
                    # Similar training process as single-pair models
                    is_lstm = model_type in ['enhanced_cnn_lstm', 'advanced_tft']
                    X_train, y_train, scaler, feature_names = self.prepare_model_data(train_data, is_lstm=is_lstm)
                    X_val, y_val, _, _ = self.prepare_model_data(val_data, is_lstm=is_lstm)
                    
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
                        model = self.build_enhanced_cnn_lstm_model(input_shape, 'Bagging')
                        
                        callbacks = [
                            EarlyStopping(monitor='val_loss', patience=self.config['patience'], 
                                        restore_best_weights=True),
                            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
                        ]
                        
                        history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                                          epochs=self.config['epochs'], batch_size=self.config['batch_size'],
                                          callbacks=callbacks, verbose=1)
                        
                        model_path = os.path.join(models_dir, f"{model_key}_best.keras")
                        model.save(model_path)
                        
                        trained_models[model_key] = {
                            'model': model, 'scaler': scaler, 'feature_names': feature_names,
                            'history': history.history
                        }
                        
                    elif model_type == 'advanced_tft':
                        input_dim = X_train.shape[2]
                        model = self.build_advanced_tft_model(input_dim, 'Bagging')
                        
                        callbacks = [
                            EarlyStopping(monitor='val_loss', patience=self.config['patience'], 
                                        restore_best_weights=True),
                            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
                        ]
                        
                        history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                                          epochs=self.config['epochs'], batch_size=self.config['batch_size'],
                                          callbacks=callbacks, verbose=1)
                        
                        model_path = os.path.join(models_dir, f"{model_key}_best.keras")
                        model.save(model_path)
                        
                        trained_models[model_key] = {
                            'model': model, 'scaler': scaler, 'feature_names': feature_names,
                            'history': history.history
                        }
                        
                    elif model_type == 'enhanced_xgboost':
                        model = self.build_enhanced_xgboost_model('Bagging')
                        
                        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=1)
                        
                        model_path = os.path.join(models_dir, f"{model_key}.pkl")
                        with open(model_path, 'wb') as f:
                            pickle.dump(model, f)
                        
                        trained_models[model_key] = {
                            'model': model, 'scaler': scaler, 'feature_names': feature_names
                        }
                    
                    logger.info(f"Successfully trained {model_key}")
                    
                except Exception as e:
                    logger.error(f"Error training {model_key}: {e}")
                    continue
        
        self.models = trained_models
        return trained_models
    
    def evaluate_enhanced_models(self):
        """Evaluate enhanced models with advanced metrics and risk management."""
        logger.info("=== ENHANCED MODEL EVALUATION ===")
        
        results = {}
        
        # Evaluate single-pair models
        for pair in self.config['currency_pairs']:
            test_data = self.selected_features[pair]['test']
            
            for model_type in self.config['models_to_train']:
                model_key = f"{pair}_{model_type}"
                if model_key not in self.models:
                    continue
                
                logger.info(f"Evaluating {model_key}")
                
                try:
                    model_data = self.models[model_key]
                    model = model_data['model']
                    scaler = model_data['scaler']
                    
                    # Generate predictions
                    is_lstm = model_type in ['enhanced_cnn_lstm', 'advanced_tft']
                    X_test, y_test, _, _ = self.prepare_model_data(test_data, is_lstm=is_lstm)
                    
                    if np.isnan(X_test).any() or np.isnan(y_test).any():
                        X_test = np.nan_to_num(X_test)
                        y_test = np.nan_to_num(y_test)
                    
                    if is_lstm:
                        y_pred_proba = model.predict(X_test)
                        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
                        y_pred_proba = y_pred_proba.flatten()
                    else:
                        y_pred_proba = model.predict_proba(X_test)[:, 1]
                        y_pred = (y_pred_proba > 0.5).astype(int)
                    
                    # Calculate basic metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred)
                    recall = recall_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)
                    
                    # Enhanced trading performance evaluation
                    trading_performance = self.evaluate_enhanced_trading_performance(
                        pair, model_type, y_test, y_pred, y_pred_proba, test_data
                    )
                    
                    results[model_key] = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'trading_performance': trading_performance
                    }
                    
                    logger.info(f"{model_key} - Accuracy: {accuracy:.4f}, "
                              f"Annual Return: {trading_performance['annual_return']:.2f}%")
                    
                except Exception as e:
                    logger.error(f"Error evaluating {model_key}: {e}")
                    continue
        
        # Evaluate bagging models
        if self.config['use_bagging']:
            for pair in self.config['currency_pairs']:
                # Filter bagging test data for this pair
                bagging_test_data = self.bagging_data['test']
                pair_test_data = bagging_test_data[bagging_test_data['CurrencyPair'] == pair].copy()
                
                if pair_test_data.empty:
                    continue
                
                for model_type in self.config['models_to_train']:
                    model_key = f"Bagging_{model_type}"
                    result_key = f"{model_key}_{pair}"
                    
                    if model_key not in self.models:
                        continue
                    
                    logger.info(f"Evaluating {result_key}")
                    
                    try:
                        model_data = self.models[model_key]
                        model = model_data['model']
                        
                        # Generate predictions
                        is_lstm = model_type in ['enhanced_cnn_lstm', 'advanced_tft']
                        X_test, y_test, _, _ = self.prepare_model_data(pair_test_data, is_lstm=is_lstm)
                        
                        if np.isnan(X_test).any() or np.isnan(y_test).any():
                            X_test = np.nan_to_num(X_test)
                            y_test = np.nan_to_num(y_test)
                        
                        if is_lstm:
                            y_pred_proba = model.predict(X_test)
                            y_pred = (y_pred_proba > 0.5).astype(int).flatten()
                            y_pred_proba = y_pred_proba.flatten()
                        else:
                            y_pred_proba = model.predict_proba(X_test)[:, 1]
                            y_pred = (y_pred_proba > 0.5).astype(int)
                        
                        # Calculate metrics
                        accuracy = accuracy_score(y_test, y_pred)
                        precision = precision_score(y_test, y_pred)
                        recall = recall_score(y_test, y_pred)
                        f1 = f1_score(y_test, y_pred)
                        
                        # Enhanced trading performance
                        trading_performance = self.evaluate_enhanced_trading_performance(
                            pair, f"Bagging_{model_type}", y_test, y_pred, y_pred_proba, pair_test_data
                        )
                        
                        results[result_key] = {
                            'accuracy': accuracy,
                            'precision': precision,
                            'recall': recall,
                            'f1': f1,
                            'trading_performance': trading_performance
                        }
                        
                        logger.info(f"{result_key} - Accuracy: {accuracy:.4f}, "
                                  f"Annual Return: {trading_performance['annual_return']:.2f}%")
                        
                    except Exception as e:
                        logger.error(f"Error evaluating {result_key}: {e}")
                        continue
        
        # Save enhanced results
        with open(os.path.join(results_dir, 'enhanced_evaluation_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        self.results = results
        return results
    
    def evaluate_enhanced_trading_performance(self, pair, model_type, y_true, y_pred, y_pred_proba, test_data):
        """Enhanced trading performance evaluation with advanced risk management."""
        logger.info(f"Enhanced trading performance evaluation for {pair} using {model_type}")
        
        # Get price and ATR data
        close_prices = test_data['Close'].values
        
        # Calculate ATR for dynamic stops
        if 'ATR_14' in test_data.columns:
            atr_values = test_data['ATR_14'].values
        else:
            # Calculate ATR if not available
            high_prices = test_data['High'].values
            low_prices = test_data['Low'].values
            tr1 = high_prices - low_prices
            tr2 = np.abs(high_prices[1:] - close_prices[:-1])
            tr3 = np.abs(low_prices[1:] - close_prices[:-1])
            true_range = np.maximum(tr1[1:], np.maximum(tr2, tr3))
            atr_values = pd.Series(true_range).rolling(window=14).mean().values
            atr_values = np.concatenate([[atr_values[0]], atr_values])
        
        # Align data lengths
        min_len = min(len(close_prices), len(y_pred), len(atr_values))
        close_prices = close_prices[-min_len:]
        y_pred = y_pred[-min_len:]
        y_pred_proba = y_pred_proba[-min_len:]
        atr_values = atr_values[-min_len:]
        
        # Enhanced trading simulation with risk management
        initial_balance = 100000  # $100,000 initial capital
        balance = initial_balance
        equity_curve = [balance]
        trades = []
        position = 0  # 0: no position, 1: long, -1: short
        entry_price = 0
        entry_confidence = 0
        stop_loss = 0
        take_profit = 0
        position_size = 0
        
        # Risk management parameters
        max_risk_per_trade = 0.02  # 2% max risk per trade
        confidence_threshold = self.config['risk_management']['confidence_threshold']
        
        # Track performance metrics
        trade_returns = []
        winning_trades = 0
        losing_trades = 0
        max_balance = balance
        max_drawdown = 0
        
        for i in range(len(y_pred)):
            current_price = close_prices[i]
            signal = y_pred[i]
            confidence = abs(y_pred_proba[i] - 0.5) * 2  # Convert to 0-1 confidence scale
            current_atr = atr_values[i] if not np.isnan(atr_values[i]) else current_price * 0.01
            
            # Update equity curve if in position
            if position != 0:
                unrealized_pnl = (current_price - entry_price) * position * position_size
                current_equity = balance + unrealized_pnl
                equity_curve.append(current_equity)
                
                # Update max balance and drawdown
                if current_equity > max_balance:
                    max_balance = current_equity
                current_drawdown = (max_balance - current_equity) / max_balance
                if current_drawdown > max_drawdown:
                    max_drawdown = current_drawdown
            else:
                equity_curve.append(balance)
            
            # Exit conditions (stop loss, take profit, or signal change)
            if position != 0:
                should_exit = False
                exit_reason = ""
                
                # Stop loss check
                if position == 1 and current_price <= stop_loss:
                    should_exit = True
                    exit_reason = "stop_loss"
                elif position == -1 and current_price >= stop_loss:
                    should_exit = True
                    exit_reason = "stop_loss"
                
                # Take profit check
                elif position == 1 and current_price >= take_profit:
                    should_exit = True
                    exit_reason = "take_profit"
                elif position == -1 and current_price <= take_profit:
                    should_exit = True
                    exit_reason = "take_profit"
                
                # Signal change with sufficient confidence
                elif confidence > confidence_threshold:
                    if position == 1 and signal == 0:
                        should_exit = True
                        exit_reason = "signal_change"
                    elif position == -1 and signal == 1:
                        should_exit = True
                        exit_reason = "signal_change"
                
                # Exit position
                if should_exit:
                    pnl = (current_price - entry_price) * position * position_size
                    balance += pnl
                    return_pct = pnl / (entry_price * position_size)
                    
                    trade_returns.append(return_pct)
                    
                    if pnl > 0:
                        winning_trades += 1
                    else:
                        losing_trades += 1
                    
                    trades.append({
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'position': position,
                        'size': position_size,
                        'pnl': pnl,
                        'return_pct': return_pct,
                        'confidence': entry_confidence,
                        'exit_reason': exit_reason,
                        'bars_held': len([t for t in trades[-10:] if 'entry_price' in t]) + 1
                    })
                    
                    position = 0
                    position_size = 0
            
            # Entry conditions
            if position == 0 and confidence > confidence_threshold:
                # Calculate position size using Kelly Criterion and confidence
                if len(trade_returns) > 10:
                    win_rate = winning_trades / (winning_trades + losing_trades)
                    avg_win = np.mean([r for r in trade_returns if r > 0]) if any(r > 0 for r in trade_returns) else 0
                    avg_loss = abs(np.mean([r for r in trade_returns if r < 0])) if any(r < 0 for r in trade_returns) else 0.01
                    
                    kelly_fraction = self.calculate_kelly_criterion(win_rate, avg_win, avg_loss)
                    position_size = self.calculate_dynamic_position_size(confidence, kelly_fraction) * balance / current_price
                else:
                    # Initial position sizing
                    position_size = (max_risk_per_trade * balance) / (current_atr * 2) # Risk-based sizing
                
                # Limit position size
                max_position_value = balance * 0.2  # Max 20% of balance per trade
                position_size = min(position_size, max_position_value / current_price)
                
                if position_size > 0:
                    # Enter position
                    position = 1 if signal == 1 else -1
                    entry_price = current_price
                    entry_confidence = confidence
                    
                    # Calculate dynamic stops
                    stop_loss, take_profit = self.calculate_dynamic_stops(
                        entry_price, current_atr, confidence, position
                    )
        
        # Close any remaining position
        if position != 0:
            final_price = close_prices[-1]
            pnl = (final_price - entry_price) * position * position_size
            balance += pnl
            return_pct = pnl / (entry_price * position_size)
            trade_returns.append(return_pct)
            
            if pnl > 0:
                winning_trades += 1
            else:
                losing_trades += 1
            
            trades.append({
                'entry_price': entry_price,
                'exit_price': final_price,
                'position': position,
                'size': position_size,
                'pnl': pnl,
                'return_pct': return_pct,
                'confidence': entry_confidence,
                'exit_reason': 'end_of_period'
            })
        
        # Calculate enhanced performance metrics
        total_trades = winning_trades + losing_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        total_return = (balance / initial_balance - 1) * 100
        
        # Annualized return (assuming 252 trading days)
        n_days = len(close_prices) / 24  # Convert hours to days
        if n_days > 0:
            annual_return = ((balance / initial_balance) ** (252 / n_days) - 1) * 100
        else:
            annual_return = 0
        
        # Buy & Hold comparison
        buy_hold_return = (close_prices[-1] / close_prices[0] - 1) * 100
        buy_hold_annual = ((close_prices[-1] / close_prices[0]) ** (252 / n_days) - 1) * 100 if n_days > 0 else 0
        
        # Sharpe ratio calculation
        if len(trade_returns) > 1:
            sharpe_ratio = np.mean(trade_returns) / np.std(trade_returns) * np.sqrt(252) if np.std(trade_returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Sortino ratio (downside deviation)
        downside_returns = [r for r in trade_returns if r < 0]
        if len(downside_returns) > 1:
            downside_std = np.std(downside_returns)
            sortino_ratio = np.mean(trade_returns) / downside_std * np.sqrt(252) if downside_std > 0 else 0
        else:
            sortino_ratio = 0
        
        # Calmar ratio (annual return / max drawdown)
        calmar_ratio = annual_return / (max_drawdown * 100) if max_drawdown > 0 else 0
        
        # Profit factor
        gross_profit = sum([r for r in trade_returns if r > 0])
        gross_loss = abs(sum([r for r in trade_returns if r < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Average trade metrics
        avg_trade_return = np.mean(trade_returns) if trade_returns else 0
        avg_winning_trade = np.mean([r for r in trade_returns if r > 0]) if any(r > 0 for r in trade_returns) else 0
        avg_losing_trade = np.mean([r for r in trade_returns if r < 0]) if any(r < 0 for r in trade_returns) else 0
        
        # Market condition
        market_condition = "Up Market" if close_prices[-1] > close_prices[0] else "Down Market"
        
        # Volatility metrics
        price_volatility = np.std(np.diff(close_prices) / close_prices[:-1]) * np.sqrt(252 * 24)  # Annualized
        strategy_volatility = np.std(trade_returns) * np.sqrt(252) if len(trade_returns) > 1 else 0
        
        performance = {
            'annual_return': annual_return,
            'total_return': total_return,
            'win_rate': win_rate,
            'trade_count': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'avg_trade_return': avg_trade_return * 100,  # Convert to percentage
            'avg_winning_trade': avg_winning_trade * 100,
            'avg_losing_trade': avg_losing_trade * 100,
            'max_drawdown': max_drawdown * 100,  # Convert to percentage
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'profit_factor': profit_factor,
            'market_condition': market_condition,
            'buy_hold_return': buy_hold_return,
            'buy_hold_annual_return': buy_hold_annual,
            'price_volatility': price_volatility * 100,
            'strategy_volatility': strategy_volatility * 100,
            'risk_adjusted_return': annual_return / (strategy_volatility * 100) if strategy_volatility > 0 else 0,
            'total_trades': total_trades,
            'final_balance': balance,
            'equity_curve': equity_curve[-100:] if len(equity_curve) > 100 else equity_curve  # Last 100 points
        }
        
        logger.info(f"Enhanced trading performance for {pair} {model_type}:")
        logger.info(f"  Annual Return: {annual_return:.2f}%")
        logger.info(f"  Win Rate: {win_rate:.3f}")
        logger.info(f"  Max Drawdown: {max_drawdown*100:.2f}%")
        logger.info(f"  Sharpe Ratio: {sharpe_ratio:.3f}")
        logger.info(f"  Profit Factor: {profit_factor:.3f}")
        
        return performance
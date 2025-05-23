"""
Enhanced Forex Trend Prediction System for Master's Thesis
Version: 2.0 - Complete Rewrite with Advanced Features
Author: Master's Thesis Student
Date: 2024

Key Enhancements:
1. Elliott Wave Pattern Recognition
2. Fibonacci Retracement Analysis  
3. Market Regime Detection (Trending/Ranging/Volatility)
4. Dynamic Ensemble with Market-Adaptive Weights
5. Advanced Risk Management with Kelly Criterion
6. Walk-Forward Analysis Support
7. Multi-Objective Optimization Ready
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import json
import pickle
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from collections import defaultdict, OrderedDict

# Scientific Computing
from scipy import signal, stats
from scipy.optimize import minimize, differential_evolution
from scipy.signal import find_peaks, argrelextrema
from scipy.stats import linregress, pearsonr, spearmanr

# Machine Learning
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, QuantileTransformer
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           roc_auc_score, confusion_matrix, classification_report,
                           mean_squared_error, mean_absolute_error)
from sklearn.feature_selection import (SelectKBest, mutual_info_classif, chi2, 
                                     RFE, SelectFromModel, VarianceThreshold)
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, 
                            VotingClassifier, StackingClassifier, AdaBoostClassifier)
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, optimizers, regularizers
from tensorflow.keras.layers import (Dense, LSTM, GRU, Conv1D, Conv2D, MaxPooling1D, 
                                   Dropout, BatchNormalization, LayerNormalization,
                                   MultiHeadAttention, GlobalAveragePooling1D,
                                   Bidirectional, TimeDistributed, Flatten, Reshape,
                                   Add, Concatenate, Multiply, Permute, RepeatVector,
                                   Embedding, SpatialDropout1D)
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
                                      TensorBoard, LearningRateScheduler, CSVLogger)
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.optimizers.schedules import ExponentialDecay, CosineDecay
from tensorflow.keras.utils import to_categorical

# Technical Analysis
import talib
import pandas_ta as ta

# Optimization
import optuna
from optuna.samplers import TPESampler, CmaEsSampler
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
from hyperopt import hp, tpe, Trials, fmin
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")

# Setup enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('forex_prediction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Enhanced Configuration
@dataclass
class TradingConfig:
    """Enhanced configuration for trading system"""
    # Data Configuration
    train_start: str = '2020-01-01'
    train_end: str = '2021-08-31'
    validation_start: str = '2021-09-01'
    validation_end: str = '2022-04-30'
    test_start: str = '2022-05-01'
    test_end: str = '2022-12-31'
    
    # Model Configuration
    window_size: int = 60  # Look-back window
    prediction_horizon: int = 1  # Predict 1 hour ahead
    batch_size: int = 32
    epochs: int = 100
    early_stopping_patience: int = 15
    
    # Feature Engineering
    use_elliott_wave: bool = True
    use_fibonacci: bool = True
    use_market_regime: bool = True
    n_technical_indicators: int = 50
    feature_engineering_jobs: int = -1
    
    # Model Architecture
    use_attention: bool = True
    use_residual_connections: bool = True
    ensemble_method: str = 'dynamic'  # 'voting', 'stacking', 'dynamic'
    
    # Risk Management
    initial_capital: float = 100000.0
    max_position_size: float = 0.1  # Max 10% per trade
    use_kelly_criterion: bool = True
    use_dynamic_stops: bool = True
    stop_loss_atr_multiplier: float = 2.0
    take_profit_atr_multiplier: float = 3.0
    
    # Optimization
    optimization_method: str = 'bayesian'  # 'grid', 'random', 'bayesian', 'optuna'
    n_optimization_trials: int = 100
    optimization_timeout: int = 3600  # 1 hour
    
    # Evaluation
    use_walk_forward: bool = True
    walk_forward_window: int = 252  # 1 year in trading days
    walk_forward_step: int = 21  # 1 month step
    
    # Paths
    data_dir: str = './data'
    output_dir: str = './output'
    model_dir: str = './models'
    

class MarketRegime(Enum):
    """Market regime classifications"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


class ElliottWaveAnalyzer:
    """Advanced Elliott Wave Pattern Recognition"""
    
    def __init__(self, min_wave_size: int = 5):
        self.min_wave_size = min_wave_size
        self.wave_patterns = {
            'impulse': [1, 2, 3, 4, 5],
            'corrective': ['A', 'B', 'C'],
            'complex_corrective': ['W', 'X', 'Y', 'Z']
        }
        
    def find_pivots(self, data: pd.Series, window: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Find pivot highs and lows"""
        highs = argrelextrema(data.values, np.greater, order=window)[0]
        lows = argrelextrema(data.values, np.less, order=window)[0]
        return highs, lows
    
    def identify_waves(self, data: pd.Series) -> Dict[str, Any]:
        """Identify Elliott Wave patterns"""
        highs, lows = self.find_pivots(data)
        
        # Combine and sort pivots
        pivots = []
        for h in highs:
            pivots.append({'index': h, 'type': 'high', 'price': data.iloc[h]})
        for l in lows:
            pivots.append({'index': l, 'type': 'low', 'price': data.iloc[l]})
        
        pivots.sort(key=lambda x: x['index'])
        
        # Analyze wave structure
        waves = self._analyze_wave_structure(pivots)
        
        # Calculate wave features
        features = {
            'wave_count': len(waves),
            'current_wave': waves[-1]['wave'] if waves else 0,
            'wave_direction': waves[-1]['direction'] if waves else 0,
            'wave_strength': self._calculate_wave_strength(waves),
            'wave_momentum': self._calculate_wave_momentum(waves),
            'is_impulse': self._is_impulse_wave(waves),
            'is_corrective': self._is_corrective_wave(waves)
        }
        
        return features
    
    def _analyze_wave_structure(self, pivots: List[Dict]) -> List[Dict]:
        """Analyze wave structure from pivots"""
        waves = []
        if len(pivots) < 3:
            return waves
            
        for i in range(1, len(pivots)):
            wave = {
                'start': pivots[i-1]['index'],
                'end': pivots[i]['index'],
                'start_price': pivots[i-1]['price'],
                'end_price': pivots[i]['price'],
                'direction': 1 if pivots[i]['price'] > pivots[i-1]['price'] else -1,
                'magnitude': abs(pivots[i]['price'] - pivots[i-1]['price']),
                'wave': (i % 5) + 1  # Simplified wave counting
            }
            waves.append(wave)
            
        return waves
    
    def _calculate_wave_strength(self, waves: List[Dict]) -> float:
        """Calculate wave strength"""
        if not waves:
            return 0.0
            
        strengths = []
        for i in range(1, len(waves)):
            if waves[i]['direction'] == waves[i-1]['direction']:
                strength = waves[i]['magnitude'] / (waves[i-1]['magnitude'] + 1e-10)
                strengths.append(strength)
                
        return np.mean(strengths) if strengths else 0.0
    
    def _calculate_wave_momentum(self, waves: List[Dict]) -> float:
        """Calculate wave momentum"""
        if len(waves) < 3:
            return 0.0
            
        recent_waves = waves[-3:]
        momentum = sum(w['direction'] * w['magnitude'] for w in recent_waves)
        return momentum
    
    def _is_impulse_wave(self, waves: List[Dict]) -> bool:
        """Check if current pattern is impulse wave"""
        if len(waves) < 5:
            return False
            
        # Simplified impulse wave rules
        recent_waves = waves[-5:]
        
        # Wave 3 should not be the shortest
        wave3_magnitude = recent_waves[2]['magnitude']
        other_magnitudes = [recent_waves[0]['magnitude'], recent_waves[4]['magnitude']]
        
        return wave3_magnitude >= min(other_magnitudes)
    
    def _is_corrective_wave(self, waves: List[Dict]) -> bool:
        """Check if current pattern is corrective wave"""
        if len(waves) < 3:
            return False
            
        # Simplified corrective wave detection
        recent_waves = waves[-3:]
        directions = [w['direction'] for w in recent_waves]
        
        # ABC pattern typically has alternating directions
        return directions[0] != directions[1] and directions[1] != directions[2]


class FibonacciAnalyzer:
    """Advanced Fibonacci Retracement and Extension Analysis"""
    
    def __init__(self):
        self.fib_ratios = {
            'retracement': [0.236, 0.382, 0.5, 0.618, 0.786],
            'extension': [1.0, 1.272, 1.618, 2.0, 2.618]
        }
        
    def calculate_fibonacci_levels(self, high: float, low: float, 
                                 trend: str = 'up') -> Dict[str, float]:
        """Calculate Fibonacci levels"""
        diff = high - low
        levels = {}
        
        if trend == 'up':
            # Retracement levels from high
            for ratio in self.fib_ratios['retracement']:
                levels[f'fib_{ratio}'] = high - (diff * ratio)
            # Extension levels from high
            for ratio in self.fib_ratios['extension']:
                levels[f'fib_ext_{ratio}'] = high + (diff * (ratio - 1))
        else:
            # Retracement levels from low
            for ratio in self.fib_ratios['retracement']:
                levels[f'fib_{ratio}'] = low + (diff * ratio)
            # Extension levels from low
            for ratio in self.fib_ratios['extension']:
                levels[f'fib_ext_{ratio}'] = low - (diff * (ratio - 1))
                
        return levels
    
    def find_fibonacci_clusters(self, data: pd.DataFrame, 
                              window: int = 50) -> pd.DataFrame:
        """Find Fibonacci clusters from multiple swings"""
        high_roll = data['High'].rolling(window=window).max()
        low_roll = data['Low'].rolling(window=window).min()
        
        fib_features = pd.DataFrame(index=data.index)
        
        # Calculate distance to nearest Fibonacci level
        for i in range(window, len(data)):
            high = high_roll.iloc[i]
            low = low_roll.iloc[i]
            current_price = data['Close'].iloc[i]
            
            # Determine trend
            mid_point = (high + low) / 2
            trend = 'up' if current_price > mid_point else 'down'
            
            # Get Fibonacci levels
            levels = self.calculate_fibonacci_levels(high, low, trend)
            
            # Calculate distances
            distances = []
            for level_name, level_price in levels.items():
                distance = abs(current_price - level_price) / current_price
                distances.append(distance)
                fib_features.loc[data.index[i], f'{level_name}_distance'] = distance
            
            # Find nearest level
            min_distance = min(distances)
            fib_features.loc[data.index[i], 'nearest_fib_distance'] = min_distance
            fib_features.loc[data.index[i], 'at_fib_level'] = 1 if min_distance < 0.002 else 0
            
        return fib_features.fillna(0)
    
    def calculate_fibonacci_time_zones(self, data: pd.Series, 
                                     start_idx: int) -> np.ndarray:
        """Calculate Fibonacci time zones"""
        fib_numbers = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
        time_zones = np.zeros(len(data))
        
        for fib in fib_numbers:
            zone_idx = start_idx + fib
            if zone_idx < len(data):
                time_zones[zone_idx] = 1
                
        return time_zones


class MarketRegimeDetector:
    """Advanced Market Regime Detection"""
    
    def __init__(self, lookback: int = 50):
        self.lookback = lookback
        self.regime_thresholds = {
            'trend_strength': 0.7,
            'volatility_high': 1.5,
            'volatility_low': 0.5,
            'range_bound': 0.3
        }
        
    def detect_regime(self, data: pd.DataFrame) -> pd.Series:
        """Detect current market regime"""
        regimes = pd.Series(index=data.index, dtype='object')
        
        # Calculate regime indicators
        close = data['Close']
        high = data['High']
        low = data['Low']
        
        # Trend strength using linear regression
        trend_strength = self._calculate_trend_strength(close)
        
        # Volatility regime
        volatility_regime = self._calculate_volatility_regime(close)
        
        # Range detection
        range_indicator = self._calculate_range_indicator(high, low, close)
        
        # Classify regimes
        for i in range(self.lookback, len(data)):
            if trend_strength.iloc[i] > self.regime_thresholds['trend_strength']:
                regimes.iloc[i] = MarketRegime.TRENDING_UP.value
            elif trend_strength.iloc[i] < -self.regime_thresholds['trend_strength']:
                regimes.iloc[i] = MarketRegime.TRENDING_DOWN.value
            elif volatility_regime.iloc[i] > self.regime_thresholds['volatility_high']:
                regimes.iloc[i] = MarketRegime.HIGH_VOLATILITY.value
            elif volatility_regime.iloc[i] < self.regime_thresholds['volatility_low']:
                regimes.iloc[i] = MarketRegime.LOW_VOLATILITY.value
            else:
                regimes.iloc[i] = MarketRegime.RANGING.value
                
        return regimes
    
    def _calculate_trend_strength(self, close: pd.Series) -> pd.Series:
        """Calculate trend strength using multiple methods"""
        trend_strength = pd.Series(index=close.index, dtype=float)
        
        for i in range(self.lookback, len(close)):
            # Linear regression
            y = close.iloc[i-self.lookback:i].values
            x = np.arange(len(y))
            slope, _, r_value, _, _ = linregress(x, y)
            
            # Normalize by price level
            normalized_slope = slope / close.iloc[i] * 100
            
            # Weight by R-squared for reliability
            trend_strength.iloc[i] = normalized_slope * (r_value ** 2)
            
        return trend_strength.fillna(0)
    
    def _calculate_volatility_regime(self, close: pd.Series) -> pd.Series:
        """Calculate volatility regime"""
        returns = close.pct_change()
        
        # Short-term volatility
        short_vol = returns.rolling(self.lookback // 2).std()
        
        # Long-term volatility
        long_vol = returns.rolling(self.lookback).std()
        
        # Volatility regime ratio
        vol_regime = short_vol / (long_vol + 1e-10)
        
        return vol_regime.fillna(1)
    
    def _calculate_range_indicator(self, high: pd.Series, low: pd.Series, 
                                 close: pd.Series) -> pd.Series:
        """Calculate range-bound market indicator"""
        # Average True Range
        atr = talib.ATR(high.values, low.values, close.values, timeperiod=14)
        
        # Price range over period
        period_high = high.rolling(self.lookback).max()
        period_low = low.rolling(self.lookback).min()
        period_range = period_high - period_low
        
        # Normalized ATR
        normalized_atr = pd.Series(atr) / (period_range + 1e-10)
        
        return normalized_atr.fillna(0.5)
    
    def get_regime_features(self, data: pd.DataFrame, regime: pd.Series) -> pd.DataFrame:
        """Extract regime-specific features"""
        features = pd.DataFrame(index=data.index)
        
        # One-hot encode regimes
        for regime_type in MarketRegime:
            features[f'regime_{regime_type.value}'] = (regime == regime_type.value).astype(int)
        
        # Regime duration
        features['regime_duration'] = self._calculate_regime_duration(regime)
        
        # Regime transition probability
        features['regime_change_prob'] = self._calculate_regime_change_probability(regime)
        
        return features
    
    def _calculate_regime_duration(self, regime: pd.Series) -> pd.Series:
        """Calculate how long current regime has lasted"""
        duration = pd.Series(index=regime.index, dtype=int)
        current_regime = None
        count = 0
        
        for i in range(len(regime)):
            if pd.isna(regime.iloc[i]):
                duration.iloc[i] = 0
                continue
                
            if regime.iloc[i] != current_regime:
                current_regime = regime.iloc[i]
                count = 1
            else:
                count += 1
                
            duration.iloc[i] = count
            
        return duration
    
    def _calculate_regime_change_probability(self, regime: pd.Series, 
                                           window: int = 20) -> pd.Series:
        """Calculate probability of regime change"""
        changes = pd.Series(index=regime.index, dtype=float)
        
        for i in range(window, len(regime)):
            window_regimes = regime.iloc[i-window:i]
            n_changes = (window_regimes != window_regimes.shift()).sum()
            changes.iloc[i] = n_changes / window
            
        return changes.fillna(0)


class AdvancedFeatureEngineer:
    """Advanced Feature Engineering for Forex Trading"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.elliott_analyzer = ElliottWaveAnalyzer()
        self.fibonacci_analyzer = FibonacciAnalyzer()
        self.regime_detector = MarketRegimeDetector()
        
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer all features"""
        logger.info("Starting advanced feature engineering...")
        
        # Copy data to avoid modifying original
        df = data.copy()
        
        # 1. Basic price features
        df = self._add_price_features(df)
        
        # 2. Technical indicators
        df = self._add_technical_indicators(df)
        
        # 3. Elliott Wave features
        if self.config.use_elliott_wave:
            df = self._add_elliott_wave_features(df)
        
        # 4. Fibonacci features
        if self.config.use_fibonacci:
            df = self._add_fibonacci_features(df)
        
        # 5. Market regime features
        if self.config.use_market_regime:
            df = self._add_market_regime_features(df)
        
        # 6. Microstructure features
        df = self._add_microstructure_features(df)
        
        # 7. Time-based features
        df = self._add_time_features(df)
        
        # 8. Statistical features
        df = self._add_statistical_features(df)
        
        # 9. Inter-market features (if multiple pairs available)
        # df = self._add_intermarket_features(df)
        
        # 10. Target variable
        df = self._add_target_variable(df)
        
        logger.info(f"Feature engineering complete. Total features: {len(df.columns)}")
        
        return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic price-based features"""
        # Returns
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Price relationships
        df['high_low_ratio'] = df['High'] / df['Low']
        df['close_open_ratio'] = df['Close'] / df['Open']
        
        # Price position
        df['price_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
        
        # Gaps
        df['gap_up'] = (df['Open'] > df['High'].shift(1)).astype(int)
        df['gap_down'] = (df['Open'] < df['Low'].shift(1)).astype(int)
        
        # Price changes
        for period in [1, 5, 10, 20]:
            df[f'price_change_{period}'] = df['Close'].pct_change(period)
            df[f'high_{period}'] = df['High'].rolling(period).max()
            df[f'low_{period}'] = df['Low'].rolling(period).min()
            
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators"""
        high = df['High'].values
        low = df['Low'].values
        close = df['Close'].values
        volume = df['Volume'].values
        
        # Overlap Studies
        df['SMA_10'] = talib.SMA(close, timeperiod=10)
        df['SMA_20'] = talib.SMA(close, timeperiod=20)
        df['SMA_50'] = talib.SMA(close, timeperiod=50)
        df['EMA_10'] = talib.EMA(close, timeperiod=10)
        df['EMA_20'] = talib.EMA(close, timeperiod=20)
        df['EMA_50'] = talib.EMA(close, timeperiod=50)
        df['DEMA_20'] = talib.DEMA(close, timeperiod=20)
        df['TEMA_20'] = talib.TEMA(close, timeperiod=20)
        df['WMA_20'] = talib.WMA(close, timeperiod=20)
        
        # Bollinger Bands
        df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(close)
        df['BB_width'] = df['BB_upper'] - df['BB_lower']
        df['BB_position'] = (close - df['BB_lower']) / (df['BB_width'] + 1e-10)
        
        # Moving Average Convergence Divergence
        df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(close)
        
        # Momentum Indicators
        df['RSI_14'] = talib.RSI(close, timeperiod=14)
        df['RSI_28'] = talib.RSI(close, timeperiod=28)
        df['STOCH_K'], df['STOCH_D'] = talib.STOCH(high, low, close)
        df['STOCHF_K'], df['STOCHF_D'] = talib.STOCHF(high, low, close)
        df['STOCHRSI_K'], df['STOCHRSI_D'] = talib.STOCHRSI(close)
        df['WILLR'] = talib.WILLR(high, low, close)
        df['ADX'] = talib.ADX(high, low, close)
        df['ADXR'] = talib.ADXR(high, low, close)
        df['CCI'] = talib.CCI(high, low, close)
        df['MFI'] = talib.MFI(high, low, close, volume)
        df['ULTOSC'] = talib.ULTOSC(high, low, close)
        df['ROC'] = talib.ROC(close)
        df['MOM'] = talib.MOM(close)
        
        # Volatility Indicators
        df['ATR'] = talib.ATR(high, low, close)
        df['NATR'] = talib.NATR(high, low, close)
        df['TRANGE'] = talib.TRANGE(high, low, close)
        
        # Volume Indicators
        if volume.sum() > 0:  # Check if volume data is available
            df['OBV'] = talib.OBV(close, volume)
            df['AD'] = talib.AD(high, low, close, volume)
            df['ADOSC'] = talib.ADOSC(high, low, close, volume)
        
        # Pattern Recognition (returns -100, 0, or 100)
        df['CDL_DOJI'] = talib.CDLDOJI(df['Open'].values, high, low, close)
        df['CDL_HAMMER'] = talib.CDLHAMMER(df['Open'].values, high, low, close)
        df['CDL_ENGULFING'] = talib.CDLENGULFING(df['Open'].values, high, low, close)
        df['CDL_HARAMI'] = talib.CDLHARAMI(df['Open'].values, high, low, close)
        df['CDL_SHOOTINGSTAR'] = talib.CDLSHOOTINGSTAR(df['Open'].values, high, low, close)
        
        # Normalize pattern recognition outputs
        pattern_cols = [col for col in df.columns if col.startswith('CDL_')]
        for col in pattern_cols:
            df[col] = df[col] / 100  # Convert to -1, 0, 1
            
        return df
    
    def _add_elliott_wave_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Elliott Wave pattern features"""
        logger.info("Adding Elliott Wave features...")
        
        # Analyze Elliott Waves
        wave_features = self.elliott_analyzer.identify_waves(df['Close'])
        
        # Add features to dataframe
        for feature_name, value in wave_features.items():
            if isinstance(value, (int, float, bool)):
                df[f'elliott_{feature_name}'] = value
            
        # Add wave-based signals
        df['elliott_impulse_signal'] = wave_features.get('is_impulse', False).astype(int)
        df['elliott_corrective_signal'] = wave_features.get('is_corrective', False).astype(int)
        
        return df
    
    def _add_fibonacci_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Fibonacci retracement features"""
        logger.info("Adding Fibonacci features...")
        
        # Calculate Fibonacci levels
        fib_features = self.fibonacci_analyzer.find_fibonacci_clusters(df)
        
        # Merge with main dataframe
        df = pd.concat([df, fib_features], axis=1)
        
        # Add Fibonacci time zones
        # Find significant pivot points
        highs, lows = self.elliott_analyzer.find_pivots(df['Close'])
        
        if len(highs) > 0:
            # Use most recent significant high as starting point
            fib_time_zones = self.fibonacci_analyzer.calculate_fibonacci_time_zones(
                df['Close'], highs[-1]
            )
            df['fib_time_zone'] = fib_time_zones
            
        return df
    
    def _add_market_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime features"""
        logger.info("Adding market regime features...")
        
        # Detect market regimes
        regimes = self.regime_detector.detect_regime(df)
        df['market_regime'] = regimes
        
        # Get regime-specific features
        regime_features = self.regime_detector.get_regime_features(df, regimes)
        
        # Merge with main dataframe
        df = pd.concat([df, regime_features], axis=1)
        
        return df
    
    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features"""
        # Spread estimation (if bid-ask not available)
        df['estimated_spread'] = 2 * np.sqrt(np.abs(
            df['log_returns'] * df['log_returns'].shift(1)
        ))
        
        # Price efficiency
        df['price_efficiency'] = df['Close'].rolling(20).apply(
            lambda x: 1 - abs(stats.autocorrelation(x))
        )
        
        # Tick-based features
        df['upticks'] = (df['Close'] > df['Close'].shift(1)).rolling(20).sum()
        df['downticks'] = (df['Close'] < df['Close'].shift(1)).rolling(20).sum()
        df['tick_ratio'] = df['upticks'] / (df['downticks'] + 1)
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        # Basic time features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        
        # Trading session features
        df['london_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
        df['newyork_session'] = ((df['hour'] >= 13) & (df['hour'] < 21)).astype(int)
        df['tokyo_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
        df['session_overlap'] = ((df['london_session'] + df['newyork_session']) > 1).astype(int)
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features"""
        # Rolling statistics
        for window in [10, 20, 50]:
            df[f'return_mean_{window}'] = df['returns'].rolling(window).mean()
            df[f'return_std_{window}'] = df['returns'].rolling(window).std()
            df[f'return_skew_{window}'] = df['returns'].rolling(window).skew()
            df[f'return_kurt_{window}'] = df['returns'].rolling(window).kurt()
            
        # Price statistics
        for window in [20, 50]:
            rolling_prices = df['Close'].rolling(window)
            df[f'price_zscore_{window}'] = (df['Close'] - rolling_prices.mean()) / rolling_prices.std()
            
        # Entropy
        df['price_entropy'] = df['returns'].rolling(20).apply(
            lambda x: stats.entropy(np.histogram(x, bins=10)[0] + 1e-10)
        )
        
        return df
    
    def _add_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add target variable for prediction"""
        # Future return
        df['future_return'] = df['Close'].shift(-self.config.prediction_horizon).pct_change(
            self.config.prediction_horizon
        )
        
        # Classification target (1 for up, 0 for down)
        df['target'] = (df['future_return'] > 0).astype(int)
        
        # Multi-class target (strong up, up, down, strong down)
        threshold = df['future_return'].std()
        conditions = [
            df['future_return'] > threshold,
            (df['future_return'] > 0) & (df['future_return'] <= threshold),
            (df['future_return'] < 0) & (df['future_return'] >= -threshold),
            df['future_return'] < -threshold
        ]
        choices = [3, 2, 1, 0]  # Strong up, up, down, strong down
        df['target_multiclass'] = np.select(conditions, choices, default=1)
        
        return df


class DynamicEnsemble:
    """Dynamic Ensemble that adapts weights based on market conditions"""
    
    def __init__(self, base_models: Dict[str, Any], config: TradingConfig):
        self.base_models = base_models
        self.config = config
        self.regime_weights = {}
        self.performance_tracker = defaultdict(lambda: defaultdict(list))
        self.current_weights = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, market_regimes: pd.Series):
        """Fit ensemble with market regime awareness"""
        logger.info("Fitting dynamic ensemble...")
        
        # Train base models if not already trained
        for name, model in self.base_models.items():
            if not hasattr(model, 'is_fitted_'):
                logger.info(f"Training base model: {name}")
                model.fit(X, y)
                model.is_fitted_ = True
        
        # Calculate regime-specific weights
        self._calculate_regime_weights(X, y, market_regimes)
        
        # Initialize current weights
        self.current_weights = self._get_equal_weights()
        
        return self
    
    def predict(self, X: np.ndarray, market_regime: Optional[str] = None) -> np.ndarray:
        """Predict with dynamic weights"""
        predictions = {}
        
        # Get predictions from all base models
        for name, model in self.base_models.items():
            if hasattr(model, 'predict_proba'):
                predictions[name] = model.predict_proba(X)[:, 1]
            else:
                predictions[name] = model.predict(X)
        
        # Get appropriate weights
        if market_regime and market_regime in self.regime_weights:
            weights = self.regime_weights[market_regime]
        else:
            weights = self.current_weights
        
        # Weighted average
        weighted_pred = np.zeros(len(X))
        for name, pred in predictions.items():
            weight = weights.get(name, 1.0 / len(self.base_models))
            weighted_pred += weight * pred
            
        return weighted_pred
    
    def predict_proba(self, X: np.ndarray, market_regime: Optional[str] = None) -> np.ndarray:
        """Predict probabilities with dynamic weights"""
        pred = self.predict(X, market_regime)
        # Convert to probability format
        proba = np.column_stack([1 - pred, pred])
        return proba
    
    def update_weights(self, X: np.ndarray, y: np.ndarray, 
                      market_regime: str, alpha: float = 0.1):
        """Update weights based on recent performance"""
        # Get predictions
        predictions = {}
        for name, model in self.base_models.items():
            if hasattr(model, 'predict_proba'):
                predictions[name] = model.predict_proba(X)[:, 1]
            else:
                predictions[name] = model.predict(X)
        
        # Calculate performance
        for name, pred in predictions.items():
            accuracy = accuracy_score(y, (pred > 0.5).astype(int))
            self.performance_tracker[market_regime][name].append(accuracy)
        
        # Update regime weights using exponential moving average
        if market_regime not in self.regime_weights:
            self.regime_weights[market_regime] = self._get_equal_weights()
        
        # Calculate new weights based on recent performance
        new_weights = {}
        total_weight = 0
        
        for name in self.base_models.keys():
            recent_performance = self.performance_tracker[market_regime][name][-10:]
            if recent_performance:
                avg_performance = np.mean(recent_performance)
                new_weights[name] = avg_performance
                total_weight += avg_performance
        
        # Normalize weights
        if total_weight > 0:
            for name in new_weights:
                new_weights[name] /= total_weight
        
        # Blend with existing weights
        for name in self.base_models.keys():
            old_weight = self.regime_weights[market_regime].get(name, 1.0 / len(self.base_models))
            new_weight = new_weights.get(name, old_weight)
            self.regime_weights[market_regime][name] = (1 - alpha) * old_weight + alpha * new_weight
            
    def _calculate_regime_weights(self, X: np.ndarray, y: np.ndarray, 
                                 market_regimes: pd.Series):
        """Calculate initial regime-specific weights"""
        unique_regimes = market_regimes.unique()
        
        for regime in unique_regimes:
            if pd.isna(regime):
                continue
                
            # Get data for this regime
            regime_mask = market_regimes == regime
            X_regime = X[regime_mask]
            y_regime = y[regime_mask]
            
            if len(X_regime) < 100:  # Skip if too few samples
                continue
            
            # Evaluate each model on this regime
            performances = {}
            for name, model in self.base_models.items():
                # Use cross-validation for more robust estimates
                scores = []
                tscv = TimeSeriesSplit(n_splits=3)
                
                for train_idx, val_idx in tscv.split(X_regime):
                    X_train, X_val = X_regime[train_idx], X_regime[val_idx]
                    y_train, y_val = y_regime[train_idx], y_regime[val_idx]
                    
                    # Clone and train model
                    model_clone = self._clone_model(model)
                    model_clone.fit(X_train, y_train)
                    
                    # Evaluate
                    if hasattr(model_clone, 'predict_proba'):
                        y_pred = model_clone.predict_proba(X_val)[:, 1]
                    else:
                        y_pred = model_clone.predict(X_val)
                    
                    score = roc_auc_score(y_val, y_pred)
                    scores.append(score)
                
                performances[name] = np.mean(scores)
            
            # Convert performances to weights
            total_performance = sum(performances.values())
            if total_performance > 0:
                weights = {name: perf / total_performance 
                          for name, perf in performances.items()}
            else:
                weights = self._get_equal_weights()
            
            self.regime_weights[regime] = weights
            
    def _clone_model(self, model):
        """Clone a model"""
        from sklearn.base import clone
        try:
            return clone(model)
        except:
            # For deep learning models, create new instance
            return type(model)(**model.get_config())
    
    def _get_equal_weights(self) -> Dict[str, float]:
        """Get equal weights for all models"""
        n_models = len(self.base_models)
        return {name: 1.0 / n_models for name in self.base_models.keys()}


class AdvancedCNNLSTM:
    """Enhanced CNN-LSTM with Attention Mechanism and Residual Connections"""
    
    def __init__(self, config: TradingConfig, input_shape: Tuple[int, int]):
        self.config = config
        self.input_shape = input_shape
        self.model = None
        self.history = None
        self.is_fitted_ = False
        
    def build_model(self):
        """Build advanced CNN-LSTM architecture"""
        inputs = layers.Input(shape=self.input_shape)
        
        # CNN Feature Extraction
        x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Dropout(0.3)(x)
        
        # Residual CNN Block
        residual = x
        x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(128, kernel_size=3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, residual])  # Residual connection
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Bidirectional LSTM layers
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
        
        # Attention Mechanism
        if self.config.use_attention:
            # Self-attention
            attention = layers.MultiHeadAttention(
                num_heads=8, 
                key_dim=64,
                dropout=0.2
            )(x, x)
            x = layers.Add()([x, attention])  # Residual connection
            x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Global pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Dense layers
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        # Create model
        self.model = models.Model(inputs=inputs, outputs=outputs)
        
        # Compile with advanced optimizer
        optimizer = optimizers.Adam(
            learning_rate=optimizers.schedules.ExponentialDecay(
                initial_learning_rate=0.001,
                decay_steps=1000,
                decay_rate=0.95
            ),
            clipnorm=1.0
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        return self.model
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            **kwargs):
        """Fit the model"""
        if self.model is None:
            self.build_model()
        
        # Default callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.early_stopping_patience,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            ),
            callbacks.ModelCheckpoint(
                filepath=f"{self.config.model_dir}/cnn_lstm_best.h5",
                monitor='val_auc',
                mode='max',
                save_best_only=True
            )
        ]
        
        # Add custom callbacks if provided
        if 'callbacks' in kwargs:
            callbacks_list.extend(kwargs['callbacks'])
            kwargs.pop('callbacks')
        
        # Fit model
        self.history = self.model.fit(
            X, y,
            validation_data=validation_data,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks_list,
            **kwargs
        )
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.model.predict(X).flatten()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities"""
        pred = self.predict(X)
        return np.column_stack([1 - pred, pred])
    
    def get_config(self):
        """Get model configuration"""
        return {
            'config': self.config,
            'input_shape': self.input_shape
        }


class AdvancedTransformer:
    """Advanced Transformer model for time series prediction"""
    
    def __init__(self, config: TradingConfig, input_shape: Tuple[int, int]):
        self.config = config
        self.input_shape = input_shape
        self.model = None
        self.history = None
        self.is_fitted_ = False
        
    def build_model(self):
        """Build advanced Transformer architecture"""
        inputs = layers.Input(shape=self.input_shape)
        
        # Positional encoding
        positions = tf.range(start=0, limit=self.input_shape[0], delta=1)
        positions = layers.Embedding(
            input_dim=self.input_shape[0], 
            output_dim=self.input_shape[1]
        )(positions)
        
        # Add positional encoding to input
        x = inputs + positions
        
        # Transformer blocks
        for _ in range(3):
            # Multi-head attention
            attn_output = layers.MultiHeadAttention(
                num_heads=8,
                key_dim=self.input_shape[1] // 8,
                dropout=0.2
            )(x, x)
            
            # Residual connection and normalization
            x = layers.Add()([x, attn_output])
            x = layers.LayerNormalization(epsilon=1e-6)(x)
            
            # Feed-forward network
            ffn_output = layers.Dense(256, activation='relu')(x)
            ffn_output = layers.Dropout(0.2)(ffn_output)
            ffn_output = layers.Dense(self.input_shape[1])(ffn_output)
            
            # Residual connection and normalization
            x = layers.Add()([x, ffn_output])
            x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Global pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Dense layers
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        # Create model
        self.model = models.Model(inputs=inputs, outputs=outputs)
        
        # Compile
        optimizer = optimizers.Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.98,
            epsilon=1e-9
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        return self.model
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            **kwargs):
        """Fit the model"""
        if self.model is None:
            self.build_model()
        
        # Default callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.early_stopping_patience,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # Fit model
        self.history = self.model.fit(
            X, y,
            validation_data=validation_data,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks_list,
            **kwargs
        )
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.model.predict(X).flatten()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities"""
        pred = self.predict(X)
        return np.column_stack([1 - pred, pred])
    
    def get_config(self):
        """Get model configuration"""
        return {
            'config': self.config,
            'input_shape': self.input_shape
        }


class AdvancedTradingStrategy:
    """Advanced trading strategy with dynamic position sizing and risk management"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.capital = config.initial_capital
        self.positions = []
        self.trades = []
        self.equity_curve = [self.capital]
        
    def kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Calculate Kelly Criterion for position sizing"""
        if avg_loss == 0:
            return 0
        
        # Kelly formula: f = (p * b - q) / b
        # where p = win rate, q = loss rate, b = avg_win / avg_loss
        q = 1 - win_rate
        b = avg_win / avg_loss
        
        kelly = (win_rate * b - q) / b
        
        # Cap Kelly at 25% for safety
        return min(max(kelly, 0), 0.25)
    
    def calculate_position_size(self, prediction_proba: float, 
                              volatility: float, atr: float) -> float:
        """Calculate dynamic position size"""
        # Base size from Kelly Criterion
        if len(self.trades) >= 20:
            recent_trades = self.trades[-20:]
            wins = [t for t in recent_trades if t['pnl'] > 0]
            losses = [t for t in recent_trades if t['pnl'] < 0]
            
            if wins and losses:
                win_rate = len(wins) / len(recent_trades)
                avg_win = np.mean([t['pnl'] for t in wins])
                avg_loss = abs(np.mean([t['pnl'] for t in losses]))
                kelly_size = self.kelly_criterion(win_rate, avg_win, avg_loss)
            else:
                kelly_size = 0.02  # Default 2%
        else:
            kelly_size = 0.02
        
        # Adjust for prediction confidence
        confidence = abs(prediction_proba - 0.5) * 2  # Scale to 0-1
        confidence_multiplier = 0.5 + confidence * 1.5  # 0.5x to 2x
        
        # Adjust for volatility
        volatility_multiplier = 1 / (1 + volatility * 10)  # Reduce size in high volatility
        
        # Final position size
        position_size = kelly_size * confidence_multiplier * volatility_multiplier
        
        # Apply limits
        position_size = min(position_size, self.config.max_position_size)
        
        return position_size
    
    def calculate_stops(self, entry_price: float, direction: int, 
                       atr: float, confidence: float) -> Tuple[float, float]:
        """Calculate dynamic stop loss and take profit"""
        # Base stop distance on ATR
        stop_distance = atr * self.config.stop_loss_atr_multiplier
        
        # Adjust for confidence
        confidence_adj = 1 - confidence * 0.3  # Tighter stops for low confidence
        stop_distance *= confidence_adj
        
        # Calculate levels
        if direction == 1:  # Long
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + (stop_distance * self.config.take_profit_atr_multiplier / 
                                       self.config.stop_loss_atr_multiplier)
        else:  # Short
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - (stop_distance * self.config.take_profit_atr_multiplier / 
                                       self.config.stop_loss_atr_multiplier)
            
        return stop_loss, take_profit
    
    def execute_trade(self, signal: int, price: float, timestamp: pd.Timestamp,
                     prediction_proba: float, volatility: float, atr: float):
        """Execute trade with advanced risk management"""
        # Check if we have an open position
        if self.positions:
            current_position = self.positions[-1]
            
            # Check exit conditions
            if current_position['direction'] != signal or self._should_exit(current_position, price):
                self._close_position(current_position, price, timestamp)
        
        # Open new position if signaled
        if signal != 0 and not self.positions:
            # Calculate position size
            position_size = self.calculate_position_size(prediction_proba, volatility, atr)
            
            # Calculate stops
            confidence = abs(prediction_proba - 0.5) * 2
            stop_loss, take_profit = self.calculate_stops(price, signal, atr, confidence)
            
            # Open position
            position = {
                'timestamp': timestamp,
                'direction': signal,
                'entry_price': price,
                'size': position_size * self.capital / price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'confidence': confidence
            }
            
            self.positions.append(position)
            
    def _should_exit(self, position: Dict, current_price: float) -> bool:
        """Check if position should be closed"""
        # Stop loss
        if position['direction'] == 1 and current_price <= position['stop_loss']:
            return True
        elif position['direction'] == -1 and current_price >= position['stop_loss']:
            return True
            
        # Take profit
        if position['direction'] == 1 and current_price >= position['take_profit']:
            return True
        elif position['direction'] == -1 and current_price <= position['take_profit']:
            return True
            
        return False
    
    def _close_position(self, position: Dict, exit_price: float, timestamp: pd.Timestamp):
        """Close position and record trade"""
        # Calculate PnL
        if position['direction'] == 1:  # Long
            pnl = (exit_price - position['entry_price']) * position['size']
        else:  # Short
            pnl = (position['entry_price'] - exit_price) * position['size']
            
        # Record trade
        trade = {
            'entry_time': position['timestamp'],
            'exit_time': timestamp,
            'direction': position['direction'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'size': position['size'],
            'pnl': pnl,
            'return': pnl / (position['entry_price'] * position['size']),
            'confidence': position['confidence']
        }
        
        self.trades.append(trade)
        
        # Update capital
        self.capital += pnl
        self.equity_curve.append(self.capital)
        
        # Remove position
        self.positions.remove(position)
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics"""
        if not self.trades:
            return {}
            
        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t['pnl'] > 0]
        losing_trades = [t for t in self.trades if t['pnl'] < 0]
        
        win_rate = len(winning_trades) / total_trades
        
        # Returns
        returns = [t['return'] for t in self.trades]
        total_return = (self.capital / self.config.initial_capital - 1) * 100
        
        # Risk metrics
        equity_curve = np.array(self.equity_curve)
        drawdowns = (equity_curve - np.maximum.accumulate(equity_curve)) / np.maximum.accumulate(equity_curve)
        max_drawdown = abs(drawdowns.min()) * 100
        
        # Sharpe ratio (assuming 252 trading days)
        if len(returns) > 1:
            sharpe_ratio = np.sqrt(252) * np.mean(returns) / (np.std(returns) + 1e-10)
        else:
            sharpe_ratio = 0
            
        # Sortino ratio
        downside_returns = [r for r in returns if r < 0]
        if downside_returns:
            sortino_ratio = np.sqrt(252) * np.mean(returns) / (np.std(downside_returns) + 1e-10)
        else:
            sortino_ratio = sharpe_ratio
            
        # Profit factor
        gross_profit = sum(t['pnl'] for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t['pnl'] for t in losing_trades)) if losing_trades else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'profit_factor': profit_factor,
            'avg_win': np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0,
            'avg_loss': np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0,
            'final_capital': self.capital
        }


class WalkForwardAnalyzer:
    """Walk-forward analysis for robust model evaluation"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.results = []
        
    def analyze(self, data: pd.DataFrame, model_builder, feature_engineer):
        """Perform walk-forward analysis"""
        logger.info("Starting walk-forward analysis...")
        
        # Calculate window parameters
        total_days = len(data)
        window_size = self.config.walk_forward_window
        step_size = self.config.walk_forward_step
        
        # Perform walk-forward windows
        for start_idx in range(0, total_days - window_size, step_size):
            end_idx = start_idx + window_size
            
            # Split data
            train_data = data.iloc[start_idx:start_idx + int(window_size * 0.7)]
            val_data = data.iloc[start_idx + int(window_size * 0.7):start_idx + int(window_size * 0.85)]
            test_data = data.iloc[start_idx + int(window_size * 0.85):end_idx]
            
            if len(test_data) < 50:  # Skip if test set too small
                continue
                
            logger.info(f"Walk-forward window: {train_data.index[0]} to {test_data.index[-1]}")
            
            # Feature engineering
            train_features = feature_engineer.engineer_features(train_data)
            val_features = feature_engineer.engineer_features(val_data)
            test_features = feature_engineer.engineer_features(test_data)
            
            # Prepare data
            X_train, y_train = self._prepare_data(train_features)
            X_val, y_val = self._prepare_data(val_features)
            X_test, y_test = self._prepare_data(test_features)
            
            # Build and train model
            model = model_builder(input_shape=(X_train.shape[1], X_train.shape[2]))
            model.fit(X_train, y_train, validation_data=(X_val, y_val))
            
            # Evaluate
            predictions = model.predict_proba(X_test)[:, 1]
            
            # Trading simulation
            strategy = AdvancedTradingStrategy(self.config)
            
            for i in range(len(predictions)):
                signal = 1 if predictions[i] > 0.5 else -1
                price = test_features['Close'].iloc[i]
                timestamp = test_features.index[i]
                volatility = test_features['return_std_20'].iloc[i]
                atr = test_features['ATR'].iloc[i]
                
                strategy.execute_trade(signal, price, timestamp, 
                                     predictions[i], volatility, atr)
            
            # Get performance metrics
            metrics = strategy.get_performance_metrics()
            metrics['window_start'] = train_data.index[0]
            metrics['window_end'] = test_data.index[-1]
            
            self.results.append(metrics)
            
        return self.results
    
    def _prepare_data(self, features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for model training"""
        # Remove target and non-feature columns
        feature_cols = [col for col in features.columns 
                       if col not in ['target', 'future_return', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        X = features[feature_cols].values
        y = features['target'].values
        
        # Create sequences for LSTM
        X_seq, y_seq = [], []
        for i in range(self.config.window_size, len(X)):
            X_seq.append(X[i-self.config.window_size:i])
            y_seq.append(y[i])
            
        return np.array(X_seq), np.array(y_seq)
    
    def plot_results(self):
        """Plot walk-forward analysis results"""
        if not self.results:
            logger.warning("No results to plot")
            return
            
        # Convert results to DataFrame
        df_results = pd.DataFrame(self.results)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Returns over time
        axes[0, 0].plot(df_results['window_end'], df_results['total_return'])
        axes[0, 0].set_title('Total Return Over Time')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Return (%)')
        
        # Plot 2: Win rate over time
        axes[0, 1].plot(df_results['window_end'], df_results['win_rate'])
        axes[0, 1].set_title('Win Rate Over Time')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Win Rate')
        
        # Plot 3: Sharpe ratio over time
        axes[1, 0].plot(df_results['window_end'], df_results['sharpe_ratio'])
        axes[1, 0].set_title('Sharpe Ratio Over Time')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Sharpe Ratio')
        
        # Plot 4: Max drawdown over time
        axes[1, 1].plot(df_results['window_end'], df_results['max_drawdown'])
        axes[1, 1].set_title('Max Drawdown Over Time')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Max Drawdown (%)')
        
        plt.tight_layout()
        plt.savefig(f"{self.config.output_dir}/walk_forward_analysis.png")
        plt.close()
        
        # Summary statistics
        logger.info("Walk-forward analysis summary:")
        logger.info(f"Average return: {df_results['total_return'].mean():.2f}%")
        logger.info(f"Average win rate: {df_results['win_rate'].mean():.2f}")
        logger.info(f"Average Sharpe ratio: {df_results['sharpe_ratio'].mean():.2f}")
        logger.info(f"Average max drawdown: {df_results['max_drawdown'].mean():.2f}%")


class EnhancedForexPredictor:
    """Main class for enhanced Forex prediction system"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.feature_engineer = AdvancedFeatureEngineer(config)
        self.models = {}
        self.ensemble = None
        self.results = {}
        
        # Create directories
        for directory in [config.data_dir, config.output_dir, config.model_dir]:
            Path(directory).mkdir(parents=True, exist_ok=True)
            
    def load_data(self, currency_pair: str) -> pd.DataFrame:
        """Load and validate data"""
        logger.info(f"Loading data for {currency_pair}...")
        
        # Load from CSV
        file_path = f"{self.config.data_dir}/{currency_pair}_1H.csv"
        df = pd.read_csv(file_path, parse_dates=['Time'], index_col='Time')
        
        # Filter date range
        df = df[(df.index >= self.config.train_start) & 
                (df.index <= self.config.test_end)]
        
        # Validate data
        assert not df.isnull().any().any(), "Data contains NaN values"
        assert len(df) > 1000, "Insufficient data points"
        
        logger.info(f"Loaded {len(df)} data points for {currency_pair}")
        
        return df
    
    def prepare_data(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Prepare train, validation, and test sets"""
        # Split by date
        train_mask = (data.index >= self.config.train_start) & (data.index <= self.config.train_end)
        val_mask = (data.index >= self.config.validation_start) & (data.index <= self.config.validation_end)
        test_mask = (data.index >= self.config.test_start) & (data.index <= self.config.test_end)
        
        splits = {
            'train': data[train_mask],
            'validation': data[val_mask],
            'test': data[test_mask]
        }
        
        # Log split sizes
        for split_name, split_data in splits.items():
            logger.info(f"{split_name}: {len(split_data)} samples "
                       f"({split_data.index[0]} to {split_data.index[-1]})")
            
        return splits
    
    def create_sequences(self, features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series models"""
        # Remove non-feature columns
        feature_cols = [col for col in features.columns 
                       if col not in ['target', 'future_return', 'Open', 'High', 
                                     'Low', 'Close', 'Volume', 'target_multiclass']]
        
        X = features[feature_cols].values
        y = features['target'].values
        
        # Create sequences
        X_seq, y_seq = [], []
        for i in range(self.config.window_size, len(X)):
            X_seq.append(X[i-self.config.window_size:i])
            y_seq.append(y[i])
            
        return np.array(X_seq), np.array(y_seq)
    
    def build_models(self, input_shape: Tuple[int, int]):
        """Build all models"""
        logger.info("Building models...")
        
        # CNN-LSTM
        self.models['cnn_lstm'] = AdvancedCNNLSTM(self.config, input_shape)
        
        # Transformer
        self.models['transformer'] = AdvancedTransformer(self.config, input_shape)
        
        # XGBoost
        self.models['xgboost'] = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.01,
            objective='binary:logistic',
            use_label_encoder=False,
            eval_metric='auc',
            tree_method='hist',
            random_state=self.config.RANDOM_SEED
        )
        
        # LightGBM
        self.models['lightgbm'] = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.01,
            objective='binary',
            metric='auc',
            boosting_type='gbdt',
            random_state=self.config.RANDOM_SEED
        )
        
        return self.models
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray):
        """Train all models"""
        logger.info("Training models...")
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            
            if name in ['cnn_lstm', 'transformer']:
                # Deep learning models
                model.fit(X_train, y_train, validation_data=(X_val, y_val))
            else:
                # Tree-based models need flattened input
                X_train_flat = X_train.reshape(X_train.shape[0], -1)
                X_val_flat = X_val.reshape(X_val.shape[0], -1)
                
                if name == 'xgboost':
                    model.fit(X_train_flat, y_train, 
                             eval_set=[(X_val_flat, y_val)],
                             early_stopping_rounds=50,
                             verbose=False)
                else:
                    model.fit(X_train_flat, y_train,
                             eval_set=[(X_val_flat, y_val)],
                             callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
                             
    def create_ensemble(self, X_train: np.ndarray, y_train: np.ndarray, 
                       market_regimes: pd.Series):
        """Create dynamic ensemble"""
        logger.info("Creating dynamic ensemble...")
        
        # Prepare base models for ensemble
        base_models = {}
        for name, model in self.models.items():
            base_models[name] = model
            
        # Create dynamic ensemble
        self.ensemble = DynamicEnsemble(base_models, self.config)
        self.ensemble.fit(X_train, y_train, market_regimes)
        
        return self.ensemble
    
    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray,
                      test_features: pd.DataFrame, model_name: str) -> Dict:
        """Evaluate a single model"""
        # Get predictions
        if hasattr(model, 'predict_proba'):
            if model_name in ['xgboost', 'lightgbm']:
                X_test_flat = X_test.reshape(X_test.shape[0], -1)
                predictions_proba = model.predict_proba(X_test_flat)[:, 1]
            else:
                predictions_proba = model.predict_proba(X_test)[:, 1]
        else:
            predictions_proba = model.predict(X_test)
            
        predictions = (predictions_proba > 0.5).astype(int)
        
        # Classification metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)
        
        try:
            auc = roc_auc_score(y_test, predictions_proba)
        except:
            auc = 0.5
            
        # Trading simulation
        strategy = AdvancedTradingStrategy(self.config)
        
        # Get required features for trading
        test_data_aligned = test_features.iloc[self.config.window_size:]
        
        for i in range(len(predictions)):
            signal = 1 if predictions_proba[i] > 0.5 else -1
            price = test_data_aligned['Close'].iloc[i]
            timestamp = test_data_aligned.index[i]
            volatility = test_data_aligned['return_std_20'].iloc[i] if 'return_std_20' in test_data_aligned else 0.01
            atr = test_data_aligned['ATR'].iloc[i] if 'ATR' in test_data_aligned else price * 0.01
            
            strategy.execute_trade(signal, price, timestamp,
                                 predictions_proba[i], volatility, atr)
            
        # Get performance metrics
        trading_metrics = strategy.get_performance_metrics()
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            **trading_metrics
        }
    
    def plot_results(self, results: Dict):
        """Plot comprehensive results"""
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Model comparison - Classification metrics
        ax1 = plt.subplot(3, 3, 1)
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        model_names = list(results.keys())
        
        x = np.arange(len(metrics))
        width = 0.15
        
        for i, model_name in enumerate(model_names):
            values = [results[model_name].get(metric, 0) for metric in metrics]
            ax1.bar(x + i * width, values, width, label=model_name)
            
        ax1.set_xlabel('Metrics')
        ax1.set_ylabel('Score')
        ax1.set_title('Classification Metrics Comparison')
        ax1.set_xticks(x + width * (len(model_names) - 1) / 2)
        ax1.set_xticklabels(metrics)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Trading performance comparison
        ax2 = plt.subplot(3, 3, 2)
        trading_metrics = ['total_return', 'sharpe_ratio', 'win_rate']
        
        x = np.arange(len(trading_metrics))
        for i, model_name in enumerate(model_names):
            values = [results[model_name].get(metric, 0) for metric in trading_metrics]
            ax2.bar(x + i * width, values, width, label=model_name)
            
        ax2.set_xlabel('Metrics')
        ax2.set_ylabel('Value')
        ax2.set_title('Trading Performance Comparison')
        ax2.set_xticks(x + width * (len(model_names) - 1) / 2)
        ax2.set_xticklabels(trading_metrics)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Risk metrics comparison
        ax3 = plt.subplot(3, 3, 3)
        risk_metrics = ['max_drawdown', 'sortino_ratio', 'profit_factor']
        
        x = np.arange(len(risk_metrics))
        for i, model_name in enumerate(model_names):
            values = [results[model_name].get(metric, 0) for metric in risk_metrics]
            ax3.bar(x + i * width, values, width, label=model_name)
            
        ax3.set_xlabel('Metrics')
        ax3.set_ylabel('Value')
        ax3.set_title('Risk Metrics Comparison')
        ax3.set_xticks(x + width * (len(model_names) - 1) / 2)
        ax3.set_xticklabels(risk_metrics)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(f"{self.config.output_dir}/model_comparison.png", dpi=300)
        plt.close()
        
    def save_results(self, results: Dict, currency_pair: str):
        """Save results to file"""
        # Convert numpy values to Python native types
        clean_results = {}
        for model_name, metrics in results.items():
            clean_results[model_name] = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in metrics.items()
            }
            
        # Save as JSON
        output_file = f"{self.config.output_dir}/results_{currency_pair}.json"
        with open(output_file, 'w') as f:
            json.dump(clean_results, f, indent=4)
            
        logger.info(f"Results saved to {output_file}")
        
    def run_complete_pipeline(self, currency_pair: str):
        """Run complete prediction pipeline"""
        logger.info(f"Starting complete pipeline for {currency_pair}")
        
        # 1. Load data
        raw_data = self.load_data(currency_pair)
        
        # 2. Split data
        data_splits = self.prepare_data(raw_data)
        
        # 3. Feature engineering
        logger.info("Engineering features...")
        train_features = self.feature_engineer.engineer_features(data_splits['train'])
        val_features = self.feature_engineer.engineer_features(data_splits['validation'])
        test_features = self.feature_engineer.engineer_features(data_splits['test'])
        
        # Save feature importance plot
        self._plot_feature_importance(train_features)
        
        # 4. Create sequences
        X_train, y_train = self.create_sequences(train_features)
        X_val, y_val = self.create_sequences(val_features)
        X_test, y_test = self.create_sequences(test_features)
        
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Validation data shape: {X_val.shape}")
        logger.info(f"Test data shape: {X_test.shape}")
        
        # 5. Build models
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.build_models(input_shape)
        
        # 6. Train models
        self.train_models(X_train, y_train, X_val, y_val)
        
        # 7. Create ensemble
        if self.config.ensemble_method == 'dynamic':
            # Get market regimes for training data
            train_regimes = train_features['market_regime'].iloc[self.config.window_size:]
            self.create_ensemble(X_train, y_train, train_regimes)
            
        # 8. Evaluate models
        results = {}
        
        # Evaluate individual models
        for name, model in self.models.items():
            logger.info(f"Evaluating {name}...")
            results[name] = self.evaluate_model(model, X_test, y_test, test_features, name)
            
        # Evaluate ensemble
        if self.ensemble:
            logger.info("Evaluating ensemble...")
            test_regimes = test_features['market_regime'].iloc[self.config.window_size:]
            
            # Get ensemble predictions with market regime awareness
            ensemble_predictions = []
            for i in range(len(X_test)):
                regime = test_regimes.iloc[i] if i < len(test_regimes) else None
                pred = self.ensemble.predict(X_test[i:i+1], regime)
                ensemble_predictions.append(pred[0])
                
            ensemble_predictions = np.array(ensemble_predictions)
            
            # Evaluate ensemble
            results['ensemble'] = self._evaluate_predictions(
                ensemble_predictions, y_test, test_features
            )
            
        # 9. Plot results
        self.plot_results(results)
        
        # 10. Save results
        self.save_results(results, currency_pair)
        
        # 11. Walk-forward analysis (optional)
        if self.config.use_walk_forward:
            logger.info("Performing walk-forward analysis...")
            wf_analyzer = WalkForwardAnalyzer(self.config)
            wf_results = wf_analyzer.analyze(
                raw_data, 
                lambda input_shape: AdvancedCNNLSTM(self.config, input_shape),
                self.feature_engineer
            )
            wf_analyzer.plot_results()
            
        return results
    
    def _evaluate_predictions(self, predictions_proba: np.ndarray, y_test: np.ndarray,
                            test_features: pd.DataFrame) -> Dict:
        """Evaluate predictions and return metrics"""
        predictions = (predictions_proba > 0.5).astype(int)
        
        # Classification metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)
        
        try:
            auc = roc_auc_score(y_test, predictions_proba)
        except:
            auc = 0.5
            
        # Trading simulation
        strategy = AdvancedTradingStrategy(self.config)
        
        test_data_aligned = test_features.iloc[self.config.window_size:]
        
        for i in range(len(predictions)):
            signal = 1 if predictions_proba[i] > 0.5 else -1
            price = test_data_aligned['Close'].iloc[i]
            timestamp = test_data_aligned.index[i]
            volatility = test_data_aligned['return_std_20'].iloc[i] if 'return_std_20' in test_data_aligned else 0.01
            atr = test_data_aligned['ATR'].iloc[i] if 'ATR' in test_data_aligned else price * 0.01
            
            strategy.execute_trade(signal, price, timestamp,
                                 predictions_proba[i], volatility, atr)
            
        # Get performance metrics
        trading_metrics = strategy.get_performance_metrics()
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            **trading_metrics
        }
    
    def _plot_feature_importance(self, features: pd.DataFrame):
        """Plot feature importance"""
        # Remove non-feature columns
        feature_cols = [col for col in features.columns 
                       if col not in ['target', 'future_return', 'Open', 'High', 
                                     'Low', 'Close', 'Volume', 'target_multiclass']]
        
        if 'target' not in features.columns:
            return
            
        # Use Random Forest for feature importance
        X = features[feature_cols].fillna(0).values
        y = features['target'].values
        
        # Remove samples with NaN in target
        mask = ~np.isnan(y)
        X = X[mask]
        y = y[mask]
        
        if len(np.unique(y)) < 2:
            logger.warning("Not enough classes for feature importance")
            return
            
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        # Get feature importances
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1][:30]  # Top 30 features
        
        # Plot
        plt.figure(figsize=(12, 8))
        plt.title('Top 30 Feature Importances')
        plt.bar(range(len(indices)), importances[indices])
        plt.xticks(range(len(indices)), [feature_cols[i] for i in indices], rotation=90)
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.savefig(f"{self.config.output_dir}/feature_importance.png", dpi=300)
        plt.close()


# Main execution
if __name__ == "__main__":
    # Configuration
    config = TradingConfig()
    
    # Initialize predictor
    predictor = EnhancedForexPredictor(config)
    
    # Run for each currency pair
    currency_pairs = ['EURUSD', 'GBPUSD', 'USDJPY']
    all_results = {}
    
    for pair in currency_pairs:
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing {pair}")
        logger.info(f"{'='*50}")
        
        try:
            results = predictor.run_complete_pipeline(pair)
            all_results[pair] = results
        except Exception as e:
            logger.error(f"Error processing {pair}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Save combined results
    with open(f"{config.output_dir}/all_results.json", 'w') as f:
        json.dump(all_results, f, indent=4)
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("SUMMARY OF RESULTS")
    logger.info("="*50)
    
    for pair, results in all_results.items():
        logger.info(f"\n{pair}:")
        for model_name, metrics in results.items():
            logger.info(f"  {model_name}:")
            logger.info(f"    Accuracy: {metrics.get('accuracy', 0):.4f}")
            logger.info(f"    Total Return: {metrics.get('total_return', 0):.2f}%")
            logger.info(f"    Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            logger.info(f"    Max Drawdown: {metrics.get('max_drawdown', 0):.2f}%")
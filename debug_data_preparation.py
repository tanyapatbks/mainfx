"""
Debug Data Preparation Script
ตรวจสอบและแก้ไขปัญหาการเตรียมข้อมูลก่อนรัน hyperparameter optimization
"""

import os
import numpy as np
import pandas as pd
import logging
from datetime import datetime
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_data_files():
    """ตรวจสอบไฟล์ข้อมูล"""
    logger.info("🔍 ตรวจสอบไฟล์ข้อมูล...")
    
    required_files = ['EURUSD_1H.csv', 'GBPUSD_1H.csv', 'USDJPY_1H.csv']
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
        else:
            try:
                df = pd.read_csv(file)
                logger.info(f"✅ {file}: {df.shape[0]} rows, {df.shape[1]} columns")
                
                # ตรวจสอบ columns ที่จำเป็น
                required_columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
                missing_cols = [col for col in required_columns if col not in df.columns]
                if missing_cols:
                    logger.warning(f"⚠️  {file} missing columns: {missing_cols}")
                
                # ตรวจสอบ data types
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    if col in df.columns:
                        if not pd.api.types.is_numeric_dtype(df[col]):
                            logger.warning(f"⚠️  {file}: {col} is not numeric")
                        
                        # ตรวจสอบ NaN values
                        nan_count = df[col].isna().sum()
                        if nan_count > 0:
                            logger.warning(f"⚠️  {file}: {col} has {nan_count} NaN values")
                            
            except Exception as e:
                logger.error(f"❌ Error reading {file}: {e}")
                missing_files.append(file)
    
    if missing_files:
        logger.error(f"❌ ไฟล์ข้อมูลที่หายไป: {missing_files}")
        return False
    
    return True

def test_data_preprocessing():
    """ทดสอบการ preprocess ข้อมูล"""
    logger.info("🔄 ทดสอบการ preprocessing ข้อมูล...")
    
    try:
        # Import เฉพาะส่วนที่จำเป็น
        from forex_prediction import EnhancedForexPrediction
        
        # สร้าง instance แบบ minimal config
        config = {
            'window_size': 60,
            'prediction_horizons': [1],
            'train_start': '2020-01-01',
            'train_end': '2021-06-30',
            'validation_start': '2021-07-01',
            'validation_end': '2021-12-31',
            'test_start': '2022-01-01',
            'test_end': '2022-12-31',
            'currency_pairs': ['EURUSD'],  # ทดสอบเฉพาะ EURUSD ก่อน
            'models_to_train': ['xgboost'],  # ใช้ model ที่ง่ายที่สุด
            'use_bagging': False,
            'use_advanced_features': False,
            'use_elliott_wave': False,
            'use_fibonacci': False,
            'n_features': 20
        }
        
        forex_pred = EnhancedForexPrediction(config)
        
        # Step 1: Load data
        logger.info("📊 Loading data...")
        forex_pred.load_data()
        logger.info("✅ Data loading completed")
        
        # Step 2: Preprocess data
        logger.info("🔄 Preprocessing data...")
        forex_pred.preprocess_data()
        logger.info("✅ Data preprocessing completed")
        
        # ตรวจสอบข้อมูลหลัง preprocessing
        for pair in config['currency_pairs']:
            if pair in forex_pred.preprocessed_data:
                train_data = forex_pred.preprocessed_data[pair]['train']
                val_data = forex_pred.preprocessed_data[pair]['validation']
                test_data = forex_pred.preprocessed_data[pair]['test']
                
                logger.info(f"📈 {pair} preprocessed data:")
                logger.info(f"   Train: {train_data.shape}")
                logger.info(f"   Validation: {val_data.shape}")
                logger.info(f"   Test: {test_data.shape}")
                
                # ตรวจสอบ NaN values
                train_nans = train_data.isna().sum().sum()
                if train_nans > 0:
                    logger.warning(f"⚠️  {pair} train data has {train_nans} NaN values")
        
        # Step 3: Calculate technical indicators (แบบง่าย)
        logger.info("📊 Calculating basic technical indicators...")
        try:
            forex_pred.calculate_technical_indicators()
            logger.info("✅ Technical indicators calculation completed")
            
            # ตรวจสอบ enhanced features
            for pair in config['currency_pairs']:
                if pair in forex_pred.enhanced_features:
                    enhanced_data = forex_pred.enhanced_features[pair]['train']
                    logger.info(f"📈 {pair} enhanced features: {enhanced_data.shape[1]} columns")
                    
                    # ตรวจสอบ infinite values
                    inf_count = np.isinf(enhanced_data.select_dtypes(include=[np.number])).sum().sum()
                    if inf_count > 0:
                        logger.warning(f"⚠️  {pair} has {inf_count} infinite values")
                    
                    # ตรวจสอบ NaN values
                    nan_count = enhanced_data.isna().sum().sum()
                    if nan_count > 0:
                        logger.warning(f"⚠️  {pair} has {nan_count} NaN values")
                        
        except Exception as e:
            logger.error(f"❌ Error in technical indicators: {e}")
            return False
        
        # Step 4: Feature selection
        logger.info("🎯 Testing feature selection...")
        try:
            forex_pred.select_features()
            logger.info("✅ Feature selection completed")
            
            # ตรวจสอบ selected features
            for pair in config['currency_pairs']:
                if pair in forex_pred.selected_features:
                    selected_data = forex_pred.selected_features[pair]['train']
                    logger.info(f"📈 {pair} selected features: {selected_data.shape[1]} columns")
                    
                    # ตรวจสอบ target column
                    if 'Target_1h' not in selected_data.columns:
                        logger.error(f"❌ {pair} missing Target_1h column")
                        return False
                    
                    # ตรวจสอบ data quality
                    target_values = selected_data['Target_1h'].unique()
                    logger.info(f"📊 {pair} Target_1h unique values: {target_values}")
                    
        except Exception as e:
            logger.error(f"❌ Error in feature selection: {e}")
            return False
        
        # Step 5: ทดสอบ data preparation สำหรับ model
        logger.info("🤖 Testing model data preparation...")
        try:
            pair = config['currency_pairs'][0]
            train_data = forex_pred.selected_features[pair]['train']
            
            # ทดสอบ XGBoost data preparation
            X_train, y_train, scaler, feature_names = forex_pred.prepare_model_data(train_data, is_lstm=False)
            
            logger.info(f"✅ Model data preparation successful:")
            logger.info(f"   X_train shape: {X_train.shape}")
            logger.info(f"   y_train shape: {y_train.shape}")
            logger.info(f"   Feature count: {len(feature_names)}")
            
            # ตรวจสอบ data quality
            if np.isnan(X_train).any():
                logger.warning("⚠️  X_train contains NaN values")
            if np.isinf(X_train).any():
                logger.warning("⚠️  X_train contains infinite values")
            if np.isnan(y_train).any():
                logger.warning("⚠️  y_train contains NaN values")
                
        except Exception as e:
            logger.error(f"❌ Error in model data preparation: {e}")
            return False
        
        logger.info("🎉 ทุกขั้นตอนผ่านการทดสอบเรียบร้อย!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Critical error in data preprocessing test: {e}")
        return False

def test_simple_optimization():
    """ทดสอบ optimization แบบง่าย"""
    logger.info("🔧 ทดสอบ simple hyperparameter optimization...")
    
    try:
        from hyperparameter_tuning import EnhancedHyperparameterOptimizer
        
        # สร้าง optimizer
        optimizer = EnhancedHyperparameterOptimizer(optimization_type='bayesian')
        
        # ทดสอบ optimization สำหรับ XGBoost เท่านั้น (ง่ายที่สุด)
        result = optimizer.optimize_hyperparameters(
            model_type='xgboost',
            pair='EURUSD',
            n_trials=5,  # ทดสอบ 5 trials เท่านั้น
            optimization_target='f1_score'
        )
        
        logger.info(f"✅ Simple optimization test successful!")
        logger.info(f"   Best value: {result['best_value']:.4f}")
        logger.info(f"   Best params: {result['best_params']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error in simple optimization test: {e}")
        return False

def main():
    """Main function สำหรับการตรวจสอบและแก้ไขปัญหา"""
    logger.info("🚀 เริ่มการตรวจสอบและแก้ไขปัญหา Data Preparation")
    
    # Step 1: ตรวจสอบไฟล์ข้อมูล
    if not check_data_files():
        logger.error("❌ กรุณาตรวจสอบไฟล์ข้อมูลก่อนดำเนินการต่อ")
        return False
    
    # Step 2: ทดสอบการ preprocessing
    if not test_data_preprocessing():
        logger.error("❌ การ preprocessing ข้อมูลล้มเหลว")
        return False
    
    # Step 3: ทดสอบ optimization แบบง่าย
    if not test_simple_optimization():
        logger.error("❌ การทดสอบ optimization ล้มเหลว")
        return False
    
    logger.info("🎉 การตรวจสอบทั้งหมดเสร็จสมบูรณ์!")
    logger.info("💡 ตอนนี้คุณสามารถรัน hyperparameter optimization ได้แล้ว:")
    logger.info("   python hyperparameter_tuning.py --model xgboost --pair EURUSD --trials 50")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
"""
Run the Forex Trend Prediction Pipeline
- Main script to execute the complete pipeline
- Provides command line arguments for customization
"""

import os
import json
import argparse
import logging
from datetime import datetime

from forex_prediction import ForexPrediction, load_config, save_config

# Set up logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"run_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main function to run the Forex Prediction Pipeline."""
    parser = argparse.ArgumentParser(description='Run Forex Trend Prediction Pipeline')
    
    # General configuration
    parser.add_argument('--config', type=str, help='Path to configuration JSON file')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory')
    
    # Data options
    parser.add_argument('--window-size', type=int, default=60, help='Lookback window size (hours)')
    parser.add_argument('--prediction-horizon', type=int, default=1, help='Prediction horizon (hours)')
    parser.add_argument('--train-start', type=str, default='2020-01-01', help='Training data start date')
    parser.add_argument('--train-end', type=str, default='2021-12-31', help='Training data end date')
    parser.add_argument('--test-start', type=str, default='2022-01-01', help='Test data start date')
    parser.add_argument('--test-end', type=str, default='2022-04-30', help='Test data end date')
    
    # Model options
    parser.add_argument('--models', type=str, default='cnn_lstm,tft,xgboost', 
                       help='Models to train (comma-separated list: cnn_lstm,tft,xgboost)')
    parser.add_argument('--currency-pairs', type=str, default='EURUSD,GBPUSD,USDJPY',
                       help='Currency pairs to use (comma-separated list)')
    parser.add_argument('--no-bagging', action='store_true', help='Disable bagging approach')
    
    # Training options
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    
    # Feature options
    parser.add_argument('--n-features', type=int, default=30, help='Number of features to select')
    parser.add_argument('--feature-selection', type=str, default='random_forest,mutual_info,pca',
                       help='Feature selection methods (comma-separated list)')
    parser.add_argument('--scaler', type=str, default='standard', 
                       choices=['standard', 'minmax', 'robust'], help='Scaler type')
    
    # Hyperparameter tuning
    parser.add_argument('--tune-hyperparams', action='store_true', help='Enable hyperparameter tuning')
    parser.add_argument('--n-trials', type=int, default=25, help='Number of hyperparameter tuning trials')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    else:
        # Create configuration from command-line arguments
        config = {
            'window_size': args.window_size,
            'prediction_horizon': args.prediction_horizon,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'patience': args.patience,
            'test_size': 0.2,
            'validation_size': 0.2,
            'random_state': 42,
            'train_start': args.train_start,
            'train_end': args.train_end,
            'test_start': args.test_start,
            'test_end': args.test_end,
            'feature_selection_methods': args.feature_selection.split(','),
            'n_features': args.n_features,
            'models_to_train': args.models.split(','),
            'currency_pairs': args.currency_pairs.split(','),
            'use_bagging': not args.no_bagging,
            'scaler_type': args.scaler,
            'hyperparameter_tuning': args.tune_hyperparams,
            'n_trials': args.n_trials,
            'evaluation_metrics': [
                'annual_return', 'win_rate', 'market_condition', 
                'buy_hold_comparison', 'single_bagging_comparison'
            ]
        }
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(args.output_dir, "run_config.json")
    save_config(config, config_path)
    logger.info(f"Saved configuration to {config_path}")
    
    # Initialize and run pipeline
    forex_pred = ForexPrediction(config)
    logger.info("Starting Forex Prediction Pipeline")
    
    try:
        results = forex_pred.run_pipeline()
        logger.info("Pipeline completed successfully")
        
        # Save results
        results_path = os.path.join(args.output_dir, "final_results.json")
        
        # Extract the trading performance for each model for easier comparison
        summary_results = {}
        for model_key, model_results in results.items():
            if 'trading_performance' in model_results:
                summary_results[model_key] = model_results['trading_performance']
        
        with open(results_path, 'w') as f:
            json.dump(summary_results, f, indent=2)
        
        logger.info(f"Results saved to {results_path}")
        
        # Print summary of best models
        best_model = max(summary_results.items(), key=lambda x: x[1]['annual_return'])
        logger.info(f"Best model: {best_model[0]} - Annual Return: {best_model[1]['annual_return']:.2f}%")
        
        # Compare single vs bagging if both are available
        if config['use_bagging']:
            logger.info("=== Single vs Bagging Comparison ===")
            for model_type in config['models_to_train']:
                avg_single_return = 0
                avg_bagging_return = 0
                count = 0
                
                for pair in config['currency_pairs']:
                    single_key = f"{pair}_{model_type}"
                    bagging_key = f"Bagging_{model_type}_{pair}"
                    
                    if single_key in summary_results and bagging_key in summary_results:
                        single_return = summary_results[single_key]['annual_return']
                        bagging_return = summary_results[bagging_key]['annual_return']
                        
                        improvement = bagging_return - single_return
                        
                        logger.info(f"{pair} - {model_type}: Single={single_return:.2f}%, Bagging={bagging_return:.2f}%, Diff={improvement:.2f}%")
                        
                        avg_single_return += single_return
                        avg_bagging_return += bagging_return
                        count += 1
                
                if count > 0:
                    avg_single_return /= count
                    avg_bagging_return /= count
                    avg_improvement = avg_bagging_return - avg_single_return
                    
                    logger.info(f"Average for {model_type}: Single={avg_single_return:.2f}%, Bagging={avg_bagging_return:.2f}%, Diff={avg_improvement:.2f}%")
        
    except Exception as e:
        logger.error(f"Error running pipeline: {e}", exc_info=True)
        print(f"Error running pipeline: {e}")
        return 1
    
    print("Forex Prediction Pipeline completed successfully.")
    print(f"Results saved to {args.output_dir}")
    return 0

if __name__ == "__main__":
    exit(main())
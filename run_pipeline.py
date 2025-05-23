"""
Enhanced Run Pipeline for Master's Thesis Forex Prediction
Version: 2.0 - Complete Pipeline with Advanced Features
Author: Master's Thesis Student
Date: 2024

This script provides a comprehensive command-line interface for running
the enhanced Forex prediction system with all advanced features.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import yaml

# Import the main forex prediction module
from forex_prediction import (
    TradingConfig, EnhancedForexPredictor, MarketRegime,
    WalkForwardAnalyzer, AdvancedFeatureEngineer
)

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None):
    """Setup enhanced logging configuration"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        handlers=handlers
    )
    
    return logging.getLogger(__name__)


class PipelineRunner:
    """Enhanced pipeline runner with advanced features"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.results = {}
        self.metadata = {
            'start_time': datetime.now(),
            'config': config.__dict__,
            'system_info': self._get_system_info()
        }
        
    def _get_system_info(self) -> Dict:
        """Get system information"""
        import platform
        import psutil
        
        return {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'tensorflow_gpu': self._check_gpu()
        }
    
    def _check_gpu(self) -> bool:
        """Check if GPU is available"""
        try:
            import tensorflow as tf
            return len(tf.config.list_physical_devices('GPU')) > 0
        except:
            return False
    
    def validate_data(self, currency_pairs: List[str]) -> bool:
        """Validate data files exist and are valid"""
        self.logger.info("Validating data files...")
        
        for pair in currency_pairs:
            file_path = Path(self.config.data_dir) / f"{pair}_1H.csv"
            
            if not file_path.exists():
                self.logger.error(f"Data file not found: {file_path}")
                return False
            
            # Check file can be read and has required columns
            try:
                df = pd.read_csv(file_path, nrows=10)
                required_columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
                
                for col in required_columns:
                    if col not in df.columns:
                        self.logger.error(f"Missing required column '{col}' in {file_path}")
                        return False
                        
            except Exception as e:
                self.logger.error(f"Error reading {file_path}: {str(e)}")
                return False
                
        self.logger.info("Data validation successful")
        return True
    
    def run_single_pair(self, currency_pair: str, predictor: EnhancedForexPredictor) -> Dict:
        """Run pipeline for a single currency pair"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Processing {currency_pair}")
        self.logger.info(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Run complete pipeline
            results = predictor.run_complete_pipeline(currency_pair)
            
            # Add metadata
            results['_metadata'] = {
                'currency_pair': currency_pair,
                'execution_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"Completed {currency_pair} in {time.time() - start_time:.2f} seconds")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing {currency_pair}: {str(e)}")
            import traceback
            traceback.print_exc()
            
            return {
                '_metadata': {
                    'currency_pair': currency_pair,
                    'error': str(e),
                    'execution_time': time.time() - start_time,
                    'timestamp': datetime.now().isoformat()
                }
            }
    
    def run_bagging_analysis(self, currency_pairs: List[str], 
                           predictor: EnhancedForexPredictor) -> Dict:
        """Run bagging analysis across multiple currency pairs"""
        self.logger.info("\n" + "="*60)
        self.logger.info("Running Bagging Analysis")
        self.logger.info("="*60)
        
        # Combine data from all pairs
        all_data = []
        
        for pair in currency_pairs:
            self.logger.info(f"Loading data for {pair}...")
            data = predictor.load_data(pair)
            data['currency_pair'] = pair
            all_data.append(data)
        
        # Combine all data
        combined_data = pd.concat(all_data).sort_index()
        
        # Feature engineering on combined data
        self.logger.info("Engineering features for combined data...")
        feature_engineer = AdvancedFeatureEngineer(self.config)
        combined_features = feature_engineer.engineer_features(combined_data)
        
        # Split data
        data_splits = predictor.prepare_data(combined_features)
        
        # Create sequences
        X_train, y_train = predictor.create_sequences(data_splits['train'])
        X_val, y_val = predictor.create_sequences(data_splits['validation'])
        X_test, y_test = predictor.create_sequences(data_splits['test'])
        
        # Build and train models
        input_shape = (X_train.shape[1], X_train.shape[2])
        predictor.build_models(input_shape)
        predictor.train_models(X_train, y_train, X_val, y_val)
        
        # Evaluate on each pair separately
        bagging_results = {}
        
        for pair in currency_pairs:
            self.logger.info(f"Evaluating bagging model on {pair}...")
            
            # Filter test data for this pair
            pair_mask = data_splits['test']['currency_pair'] == pair
            pair_test_features = data_splits['test'][pair_mask]
            
            if len(pair_test_features) > self.config.window_size:
                X_pair_test, y_pair_test = predictor.create_sequences(pair_test_features)
                
                # Evaluate each model
                pair_results = {}
                for model_name, model in predictor.models.items():
                    pair_results[f"bagging_{model_name}"] = predictor.evaluate_model(
                        model, X_pair_test, y_pair_test, pair_test_features, model_name
                    )
                
                bagging_results[pair] = pair_results
        
        return bagging_results
    
    def generate_report(self, all_results: Dict):
        """Generate comprehensive HTML report"""
        self.logger.info("Generating comprehensive report...")
        
        # Create report directory
        report_dir = Path(self.config.output_dir) / 'report'
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate HTML content
        html_content = self._generate_html_report(all_results)
        
        # Save report
        report_path = report_dir / f"thesis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        self.logger.info(f"Report saved to: {report_path}")
        
        # Generate summary plots
        self._generate_summary_plots(all_results, report_dir)
        
        return report_path
    
    def _generate_html_report(self, results: Dict) -> str:
        """Generate HTML report content"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Forex Prediction Thesis Report</title>
            <meta charset="utf-8">
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }}
                h1, h2, h3 {{
                    color: #333;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 20px 0;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 12px;
                    text-align: left;
                }}
                th {{
                    background-color: #4CAF50;
                    color: white;
                }}
                tr:nth-child(even) {{
                    background-color: #f2f2f2;
                }}
                .metric {{
                    display: inline-block;
                    margin: 10px;
                    padding: 15px;
                    background-color: #e7f3ff;
                    border-radius: 5px;
                    min-width: 150px;
                    text-align: center;
                }}
                .metric-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #2196F3;
                }}
                .metric-label {{
                    font-size: 14px;
                    color: #666;
                }}
                .positive {{
                    color: #4CAF50;
                }}
                .negative {{
                    color: #f44336;
                }}
                .section {{
                    margin: 30px 0;
                    padding: 20px;
                    background-color: #f9f9f9;
                    border-radius: 5px;
                }}
                .timestamp {{
                    color: #999;
                    font-size: 12px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Forex Trend Prediction - Master's Thesis Results</h1>
                <p class="timestamp">Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <div class="section">
                    <h2>Executive Summary</h2>
                    {self._generate_executive_summary(results)}
                </div>
                
                <div class="section">
                    <h2>Model Performance Comparison</h2>
                    {self._generate_performance_table(results)}
                </div>
                
                <div class="section">
                    <h2>Trading Performance Analysis</h2>
                    {self._generate_trading_analysis(results)}
                </div>
                
                <div class="section">
                    <h2>Risk Analysis</h2>
                    {self._generate_risk_analysis(results)}
                </div>
                
                <div class="section">
                    <h2>Configuration Details</h2>
                    {self._generate_config_details()}
                </div>
                
                <div class="section">
                    <h2>System Information</h2>
                    {self._generate_system_info()}
                </div>
            </div>
        </body>
        </html>
        """
        return html
    
    def _generate_executive_summary(self, results: Dict) -> str:
        """Generate executive summary section"""
        # Calculate overall statistics
        all_returns = []
        all_sharpes = []
        all_accuracies = []
        
        for pair_results in results.values():
            if isinstance(pair_results, dict) and '_metadata' not in pair_results:
                for model_results in pair_results.values():
                    if isinstance(model_results, dict):
                        all_returns.append(model_results.get('total_return', 0))
                        all_sharpes.append(model_results.get('sharpe_ratio', 0))
                        all_accuracies.append(model_results.get('accuracy', 0))
        
        avg_return = np.mean(all_returns) if all_returns else 0
        avg_sharpe = np.mean(all_sharpes) if all_sharpes else 0
        avg_accuracy = np.mean(all_accuracies) if all_accuracies else 0
        best_return = max(all_returns) if all_returns else 0
        
        summary_html = f"""
        <div class="metric">
            <div class="metric-value {'positive' if avg_return > 0 else 'negative'}">
                {avg_return:.2f}%
            </div>
            <div class="metric-label">Average Return</div>
        </div>
        
        <div class="metric">
            <div class="metric-value {'positive' if best_return > 0 else 'negative'}">
                {best_return:.2f}%
            </div>
            <div class="metric-label">Best Return</div>
        </div>
        
        <div class="metric">
            <div class="metric-value">{avg_sharpe:.2f}</div>
            <div class="metric-label">Avg Sharpe Ratio</div>
        </div>
        
        <div class="metric">
            <div class="metric-value">{avg_accuracy:.2%}</div>
            <div class="metric-label">Avg Accuracy</div>
        </div>
        """
        
        return summary_html
    
    def _generate_performance_table(self, results: Dict) -> str:
        """Generate performance comparison table"""
        table_html = """
        <table>
            <thead>
                <tr>
                    <th>Currency Pair</th>
                    <th>Model</th>
                    <th>Accuracy</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1 Score</th>
                    <th>AUC</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for pair, pair_results in results.items():
            if isinstance(pair_results, dict) and '_metadata' not in pair_results:
                for model_name, metrics in pair_results.items():
                    if isinstance(metrics, dict):
                        table_html += f"""
                        <tr>
                            <td>{pair}</td>
                            <td>{model_name}</td>
                            <td>{metrics.get('accuracy', 0):.4f}</td>
                            <td>{metrics.get('precision', 0):.4f}</td>
                            <td>{metrics.get('recall', 0):.4f}</td>
                            <td>{metrics.get('f1', 0):.4f}</td>
                            <td>{metrics.get('auc', 0):.4f}</td>
                        </tr>
                        """
        
        table_html += """
            </tbody>
        </table>
        """
        
        return table_html
    
    def _generate_trading_analysis(self, results: Dict) -> str:
        """Generate trading performance analysis"""
        table_html = """
        <table>
            <thead>
                <tr>
                    <th>Currency Pair</th>
                    <th>Model</th>
                    <th>Total Return (%)</th>
                    <th>Win Rate</th>
                    <th>Sharpe Ratio</th>
                    <th>Max Drawdown (%)</th>
                    <th>Profit Factor</th>
                    <th>Total Trades</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for pair, pair_results in results.items():
            if isinstance(pair_results, dict) and '_metadata' not in pair_results:
                for model_name, metrics in pair_results.items():
                    if isinstance(metrics, dict):
                        table_html += f"""
                        <tr>
                            <td>{pair}</td>
                            <td>{model_name}</td>
                            <td class="{'positive' if metrics.get('total_return', 0) > 0 else 'negative'}">
                                {metrics.get('total_return', 0):.2f}
                            </td>
                            <td>{metrics.get('win_rate', 0):.2%}</td>
                            <td>{metrics.get('sharpe_ratio', 0):.2f}</td>
                            <td>{metrics.get('max_drawdown', 0):.2f}</td>
                            <td>{metrics.get('profit_factor', 0):.2f}</td>
                            <td>{metrics.get('total_trades', 0)}</td>
                        </tr>
                        """
        
        table_html += """
            </tbody>
        </table>
        """
        
        return table_html
    
    def _generate_risk_analysis(self, results: Dict) -> str:
        """Generate risk analysis section"""
        risk_html = "<h3>Risk Metrics Summary</h3>"
        
        # Calculate risk statistics
        all_drawdowns = []
        all_sortinos = []
        
        for pair_results in results.values():
            if isinstance(pair_results, dict) and '_metadata' not in pair_results:
                for model_results in pair_results.values():
                    if isinstance(model_results, dict):
                        all_drawdowns.append(model_results.get('max_drawdown', 0))
                        all_sortinos.append(model_results.get('sortino_ratio', 0))
        
        if all_drawdowns:
            risk_html += f"""
            <p>Average Maximum Drawdown: {np.mean(all_drawdowns):.2f}%</p>
            <p>Worst Drawdown: {max(all_drawdowns):.2f}%</p>
            <p>Average Sortino Ratio: {np.mean(all_sortinos):.2f}</p>
            """
        
        return risk_html
    
    def _generate_config_details(self) -> str:
        """Generate configuration details"""
        config_html = "<pre>"
        config_dict = self.config.__dict__.copy()
        
        # Convert to readable format
        for key, value in config_dict.items():
            config_html += f"{key}: {value}\n"
            
        config_html += "</pre>"
        return config_html
    
    def _generate_system_info(self) -> str:
        """Generate system information"""
        sys_info = self.metadata['system_info']
        
        info_html = "<pre>"
        for key, value in sys_info.items():
            info_html += f"{key}: {value}\n"
        info_html += "</pre>"
        
        return info_html
    
    def _generate_summary_plots(self, results: Dict, output_dir: Path):
        """Generate summary visualization plots"""
        self.logger.info("Generating summary plots...")
        
        # Prepare data for plotting
        model_performances = []
        
        for pair, pair_results in results.items():
            if isinstance(pair_results, dict) and '_metadata' not in pair_results:
                for model_name, metrics in pair_results.items():
                    if isinstance(metrics, dict):
                        model_performances.append({
                            'pair': pair,
                            'model': model_name,
                            'return': metrics.get('total_return', 0),
                            'sharpe': metrics.get('sharpe_ratio', 0),
                            'accuracy': metrics.get('accuracy', 0),
                            'win_rate': metrics.get('win_rate', 0)
                        })
        
        if not model_performances:
            return
            
        df_perf = pd.DataFrame(model_performances)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Returns by model and pair
        pivot_returns = df_perf.pivot(index='model', columns='pair', values='return')
        pivot_returns.plot(kind='bar', ax=axes[0, 0])
        axes[0, 0].set_title('Total Returns by Model and Currency Pair')
        axes[0, 0].set_ylabel('Return (%)')
        axes[0, 0].axhline(0, color='black', linestyle='-', alpha=0.3)
        axes[0, 0].legend(title='Currency Pair')
        
        # 2. Sharpe ratios
        pivot_sharpe = df_perf.pivot(index='model', columns='pair', values='sharpe')
        pivot_sharpe.plot(kind='bar', ax=axes[0, 1])
        axes[0, 1].set_title('Sharpe Ratios by Model and Currency Pair')
        axes[0, 1].set_ylabel('Sharpe Ratio')
        axes[0, 1].axhline(1, color='green', linestyle='--', alpha=0.3, label='Good (>1)')
        axes[0, 1].legend(title='Currency Pair')
        
        # 3. Accuracy comparison
        pivot_accuracy = df_perf.pivot(index='model', columns='pair', values='accuracy')
        pivot_accuracy.plot(kind='bar', ax=axes[1, 0])
        axes[1, 0].set_title('Model Accuracy by Currency Pair')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].axhline(0.5, color='red', linestyle='--', alpha=0.3, label='Random')
        axes[1, 0].legend(title='Currency Pair')
        
        # 4. Win rates
        pivot_winrate = df_perf.pivot(index='model', columns='pair', values='win_rate')
        pivot_winrate.plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title('Win Rates by Model and Currency Pair')
        axes[1, 1].set_ylabel('Win Rate')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axhline(0.5, color='red', linestyle='--', alpha=0.3, label='Break-even')
        axes[1, 1].legend(title='Currency Pair')
        
        # Adjust layout and save
        plt.tight_layout()
        plot_path = output_dir / 'summary_performance.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Summary plots saved to: {plot_path}")
        
        # Generate correlation heatmap
        self._generate_correlation_heatmap(df_perf, output_dir)
    
    def _generate_correlation_heatmap(self, df_perf: pd.DataFrame, output_dir: Path):
        """Generate correlation heatmap of model performances"""
        # Pivot data for correlation analysis
        metrics = ['return', 'sharpe', 'accuracy', 'win_rate']
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(20, 5))
        
        for idx, metric in enumerate(metrics):
            pivot_data = df_perf.pivot(index='model', columns='pair', values=metric)
            
            if pivot_data.shape[1] > 1:
                corr = pivot_data.corr()
                
                sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', 
                          center=0, ax=axes[idx], vmin=-1, vmax=1)
                axes[idx].set_title(f'{metric.title()} Correlation Across Pairs')
        
        plt.tight_layout()
        plot_path = output_dir / 'correlation_analysis.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Correlation analysis saved to: {plot_path}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Enhanced Forex Prediction Pipeline for Master\'s Thesis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run for all currency pairs with default settings
  python run_pipeline.py
  
  # Run for specific pairs with custom dates
  python run_pipeline.py --pairs EURUSD GBPUSD --train-end 2021-12-31
  
  # Run with walk-forward analysis and bagging
  python run_pipeline.py --walk-forward --bagging
  
  # Run with custom configuration file
  python run_pipeline.py --config custom_config.yaml
        """
    )
    
    # Data arguments
    parser.add_argument('--pairs', nargs='+', default=['EURUSD', 'GBPUSD', 'USDJPY'],
                       help='Currency pairs to process')
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Directory containing data files')
    
    # Date range arguments
    parser.add_argument('--train-start', type=str, default='2020-01-01',
                       help='Training start date (YYYY-MM-DD)')
    parser.add_argument('--train-end', type=str, default='2021-08-31',
                       help='Training end date (YYYY-MM-DD)')
    parser.add_argument('--validation-start', type=str, default='2021-09-01',
                       help='Validation start date (YYYY-MM-DD)')
    parser.add_argument('--validation-end', type=str, default='2022-04-30',
                       help='Validation end date (YYYY-MM-DD)')
    parser.add_argument('--test-start', type=str, default='2022-05-01',
                       help='Test start date (YYYY-MM-DD)')
    parser.add_argument('--test-end', type=str, default='2022-12-31',
                       help='Test end date (YYYY-MM-DD)')
    
    # Model arguments
    parser.add_argument('--window-size', type=int, default=60,
                       help='Look-back window size')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Maximum training epochs')
    parser.add_argument('--early-stopping', type=int, default=15,
                       help='Early stopping patience')
    
    # Feature engineering arguments
    parser.add_argument('--no-elliott-wave', action='store_true',
                       help='Disable Elliott Wave features')
    parser.add_argument('--no-fibonacci', action='store_true',
                       help='Disable Fibonacci features')
    parser.add_argument('--no-regime', action='store_true',
                       help='Disable market regime detection')
    
    # Advanced features
    parser.add_argument('--walk-forward', action='store_true',
                       help='Enable walk-forward analysis')
    parser.add_argument('--bagging', action='store_true',
                       help='Enable bagging analysis')
    parser.add_argument('--ensemble', choices=['voting', 'stacking', 'dynamic'],
                       default='dynamic', help='Ensemble method')
    
    # Risk management
    parser.add_argument('--initial-capital', type=float, default=100000,
                       help='Initial trading capital')
    parser.add_argument('--max-position', type=float, default=0.1,
                       help='Maximum position size (fraction)')
    parser.add_argument('--no-kelly', action='store_true',
                       help='Disable Kelly Criterion sizing')
    parser.add_argument('--no-dynamic-stops', action='store_true',
                       help='Disable dynamic stop losses')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='./output',
                       help='Output directory')
    parser.add_argument('--model-dir', type=str, default='./models',
                       help='Model save directory')
    parser.add_argument('--no-report', action='store_true',
                       help='Skip HTML report generation')
    
    # Optimization arguments
    parser.add_argument('--optimization', choices=['grid', 'random', 'bayesian', 'optuna'],
                       default='bayesian', help='Hyperparameter optimization method')
    parser.add_argument('--n-trials', type=int, default=100,
                       help='Number of optimization trials')
    
    # System arguments
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')
    parser.add_argument('--log-file', type=str, help='Log file path')
    parser.add_argument('--config', type=str, help='Configuration file (YAML)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Validate configuration without running')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level, args.log_file)
    
    # Load configuration
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        # Create config from file
        config = TradingConfig(**config_dict)
    else:
        # Create config from command line arguments
        config = TradingConfig(
            train_start=args.train_start,
            train_end=args.train_end,
            validation_start=args.validation_start,
            validation_end=args.validation_end,
            test_start=args.test_start,
            test_end=args.test_end,
            window_size=args.window_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            early_stopping_patience=args.early_stopping,
            use_elliott_wave=not args.no_elliott_wave,
            use_fibonacci=not args.no_fibonacci,
            use_market_regime=not args.no_regime,
            use_walk_forward=args.walk_forward,
            ensemble_method=args.ensemble,
            initial_capital=args.initial_capital,
            max_position_size=args.max_position,
            use_kelly_criterion=not args.no_kelly,
            use_dynamic_stops=not args.no_dynamic_stops,
            optimization_method=args.optimization,
            n_optimization_trials=args.n_trials,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            model_dir=args.model_dir
        )
    
    # Initialize pipeline runner
    runner = PipelineRunner(config)
    
    # Validate data
    if not runner.validate_data(args.pairs):
        logger.error("Data validation failed. Exiting.")
        return 1
    
    if args.dry_run:
        logger.info("Dry run complete. Configuration is valid.")
        return 0
    
    # Print configuration summary
    logger.info("\n" + "="*60)
    logger.info("CONFIGURATION SUMMARY")
    logger.info("="*60)
    logger.info(f"Currency Pairs: {args.pairs}")
    logger.info(f"Training Period: {config.train_start} to {config.train_end}")
    logger.info(f"Validation Period: {config.validation_start} to {config.validation_end}")
    logger.info(f"Test Period: {config.test_start} to {config.test_end}")
    logger.info(f"Features: Elliott Wave={config.use_elliott_wave}, "
                f"Fibonacci={config.use_fibonacci}, "
                f"Regime Detection={config.use_market_regime}")
    logger.info(f"Ensemble Method: {config.ensemble_method}")
    logger.info(f"Risk Management: Kelly={config.use_kelly_criterion}, "
                f"Dynamic Stops={config.use_dynamic_stops}")
    logger.info("="*60 + "\n")
    
    # Initialize predictor
    predictor = EnhancedForexPredictor(config)
    
    # Run pipeline for each currency pair
    all_results = {}
    
    for pair in args.pairs:
        results = runner.run_single_pair(pair, predictor)
        all_results[pair] = results
    
    # Run bagging analysis if requested
    if args.bagging:
        bagging_results = runner.run_bagging_analysis(args.pairs, predictor)
        all_results['bagging'] = bagging_results
    
    # Save all results
    results_file = Path(config.output_dir) / f"complete_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=4, default=str)
    logger.info(f"Results saved to: {results_file}")
    
    # Generate report
    if not args.no_report:
        report_path = runner.generate_report(all_results)
        logger.info(f"Report generated: {report_path}")
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("EXECUTION SUMMARY")
    logger.info("="*60)
    
    total_time = (datetime.now() - runner.metadata['start_time']).total_seconds()
    logger.info(f"Total execution time: {total_time/60:.2f} minutes")
    
    # Print best results
    best_return = -float('inf')
    best_model = None
    best_pair = None
    
    for pair, pair_results in all_results.items():
        if isinstance(pair_results, dict) and '_metadata' not in pair_results:
            for model_name, metrics in pair_results.items():
                if isinstance(metrics, dict):
                    total_return = metrics.get('total_return', -float('inf'))
                    if total_return > best_return:
                        best_return = total_return
                        best_model = model_name
                        best_pair = pair
    
    if best_model:
        logger.info(f"\nBest performing model: {best_model} on {best_pair}")
        logger.info(f"Total return: {best_return:.2f}%")
    
    logger.info("\nPipeline execution completed successfully!")
    logger.info("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
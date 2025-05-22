"""
Enhanced Run the Forex Trend Prediction Pipeline
- Main script to execute the complete enhanced pipeline
- Provides command line arguments for customization
- Includes comprehensive visualization capabilities
"""

import os
import json
import argparse
import logging
from datetime import datetime

# Import the enhanced main class
from forex_prediction import ForexPrediction

# Set up logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"enhanced_run_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def save_config(config, config_path):
    """Save configuration to JSON file."""
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

def create_comprehensive_report(results, output_dir):
    """Create a comprehensive HTML report of the results."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Forex Trend Prediction - Comprehensive Report</title>
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
            .metric-value {{
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
            }}
            .metric-label {{
                font-size: 12px;
                color: #7f8c8d;
            }}
            .positive {{ color: #27ae60; }}
            .negative {{ color: #e74c3c; }}
            .neutral {{ color: #f39c12; }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: center;
            }}
            th {{
                background-color: #3498db;
                color: white;
            }}
            .best-model {{
                background-color: #d5f4e6;
            }}
            .visualization-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            .viz-item {{
                text-align: center;
            }}
            .viz-item img {{
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                border-radius: 5px;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🏦 Forex Trend Prediction System</h1>
            <h2>Comprehensive Analysis Report</h2>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    """
    
    # Extract performance data for the report
    model_performance = []
    best_annual_return = -float('inf')
    best_model = ""
    
    for model_key, model_results in results.items():
        if 'trading_performance' in model_results:
            perf = model_results['trading_performance']
            model_performance.append({
                'model': model_key,
                'annual_return': perf['annual_return'],
                'win_rate': perf['win_rate'] * 100,
                'max_drawdown': perf['max_drawdown'],
                'sharpe_ratio': perf['sharpe_ratio'],
                'trade_count': perf['trade_count'],
                'market_condition': perf['market_condition'],
                'buy_hold_return': perf['buy_hold_annual_return']
            })
            
            if perf['annual_return'] > best_annual_return:
                best_annual_return = perf['annual_return']
                best_model = model_key
    
    # Sort by annual return
    model_performance.sort(key=lambda x: x['annual_return'], reverse=True)
    
    # Executive Summary
    avg_return = sum(m['annual_return'] for m in model_performance) / len(model_performance)
    avg_win_rate = sum(m['win_rate'] for m in model_performance) / len(model_performance)
    profitable_models = len([m for m in model_performance if m['annual_return'] > 0])
    
    html_content += f"""
        <div class="section">
            <h2>📊 Executive Summary</h2>
            <div class="metric">
                <div class="metric-value {'positive' if best_annual_return > 0 else 'negative'}">{best_annual_return:.2f}%</div>
                <div class="metric-label">Best Annual Return</div>
            </div>
            <div class="metric">
                <div class="metric-value">{best_model}</div>
                <div class="metric-label">Best Performing Model</div>
            </div>
            <div class="metric">
                <div class="metric-value {'positive' if avg_return > 0 else 'negative'}">{avg_return:.2f}%</div>
                <div class="metric-label">Average Annual Return</div>
            </div>
            <div class="metric">
                <div class="metric-value">{avg_win_rate:.1f}%</div>
                <div class="metric-label">Average Win Rate</div>
            </div>
            <div class="metric">
                <div class="metric-value {'positive' if profitable_models > len(model_performance)/2 else 'negative'}">{profitable_models}/{len(model_performance)}</div>
                <div class="metric-label">Profitable Models</div>
            </div>
        </div>
    """
    
    # Performance Table
    html_content += f"""
        <div class="section">
            <h2>📈 Model Performance Comparison</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Annual Return (%)</th>
                    <th>Win Rate (%)</th>
                    <th>Max Drawdown (%)</th>
                    <th>Sharpe Ratio</th>
                    <th>Trades</th>
                    <th>Market Condition</th>
                    <th>Buy & Hold (%)</th>
                </tr>
    """
    
    for i, model in enumerate(model_performance):
        row_class = "best-model" if i == 0 else ""
        return_class = "positive" if model['annual_return'] > 0 else "negative"
        
        html_content += f"""
                <tr class="{row_class}">
                    <td><strong>{model['model']}</strong></td>
                    <td class="{return_class}"><strong>{model['annual_return']:.2f}%</strong></td>
                    <td>{model['win_rate']:.1f}%</td>
                    <td>{model['max_drawdown']:.2f}%</td>
                    <td>{model['sharpe_ratio']:.2f}</td>
                    <td>{model['trade_count']}</td>
                    <td>{model['market_condition']}</td>
                    <td>{model['buy_hold_return']:.2f}%</td>
                </tr>
        """
    
    html_content += """
            </table>
        </div>
    """
    
    # Visualizations Section
    plot_files = []
    plots_dir = os.path.join(output_dir, "plots")
    if os.path.exists(plots_dir):
        plot_files = [f for f in os.listdir(plots_dir) if f.endswith('.png')]
    
    if plot_files:
        html_content += """
            <div class="section">
                <h2>📊 Visualizations</h2>
                <div class="visualization-grid">
        """
        
        # Organize plots by category
        plot_categories = {
            'Data Analysis': [f for f in plot_files if 'data_' in f],
            'Training Progress': [f for f in plot_files if 'training_history' in f or 'loss' in f],
            'Prediction Analysis': [f for f in plot_files if 'prediction_analysis' in f],
            'Performance Metrics': [f for f in plot_files if 'evaluation' in f or 'comparison' in f or 'performance' in f],
            'Feature Importance': [f for f in plot_files if 'feature_importance' in f],
            'Trading Performance': [f for f in plot_files if 'trading_performance' in f]
        }
        
        for category, files in plot_categories.items():
            if files:
                html_content += f"<h3>{category}</h3>"
                for plot_file in files:
                    plot_path = f"plots/{plot_file}"
                    plot_title = plot_file.replace('.png', '').replace('_', ' ').title()
                    html_content += f"""
                        <div class="viz-item">
                            <h4>{plot_title}</h4>
                            <img src="{plot_path}" alt="{plot_title}">
                        </div>
                    """
        
        html_content += """
                </div>
            </div>
        """
    
    # Key Insights
    insights = []
    
    # Performance insights
    if best_annual_return > 10:
        insights.append(f"🎯 Excellent performance: {best_model} achieved {best_annual_return:.2f}% annual return")
    elif best_annual_return > 5:
        insights.append(f"✅ Good performance: {best_model} achieved {best_annual_return:.2f}% annual return")
    elif best_annual_return > 0:
        insights.append(f"📊 Positive performance: {best_model} achieved {best_annual_return:.2f}% annual return")
    else:
        insights.append(f"⚠️ All models showed negative returns. Market conditions were challenging.")
    
    # Win rate insights
    if avg_win_rate > 60:
        insights.append(f"🎯 High prediction accuracy: Average win rate of {avg_win_rate:.1f}%")
    elif avg_win_rate > 50:
        insights.append(f"✅ Above-random prediction accuracy: Average win rate of {avg_win_rate:.1f}%")
    else:
        insights.append(f"⚠️ Prediction accuracy needs improvement: Average win rate of {avg_win_rate:.1f}%")
    
    # Model comparison insights
    single_models = [m for m in model_performance if 'Bagging' not in m['model']]
    bagging_models = [m for m in model_performance if 'Bagging' in m['model']]
    
    if single_models and bagging_models:
        avg_single = sum(m['annual_return'] for m in single_models) / len(single_models)
        avg_bagging = sum(m['annual_return'] for m in bagging_models) / len(bagging_models)
        
        if avg_bagging > avg_single:
            insights.append(f"📈 Bagging approach showed improvement: {avg_bagging:.2f}% vs {avg_single:.2f}% average return")
        else:
            insights.append(f"📉 Single-pair models outperformed bagging: {avg_single:.2f}% vs {avg_bagging:.2f}% average return")
    
    html_content += f"""
        <div class="section">
            <h2>💡 Key Insights</h2>
            <ul>
    """
    
    for insight in insights:
        html_content += f"<li>{insight}</li>"
    
    html_content += """
            </ul>
        </div>
        
        <div class="section">
            <h2>🔧 Technical Details</h2>
            <p><strong>Models Used:</strong> CNN-LSTM, Temporal Fusion Transformer (TFT), XGBoost</p>
            <p><strong>Currency Pairs:</strong> EURUSD, GBPUSD, USDJPY</p>
            <p><strong>Features:</strong> Technical indicators, price patterns, time-based features</p>
            <p><strong>Evaluation Period:</strong> 2022-01-01 to 2022-04-30</p>
            <p><strong>Performance Metrics:</strong> Annual Return, Win Rate, Maximum Drawdown, Sharpe Ratio</p>
        </div>
        
        <div class="section">
            <h2>📝 Recommendations</h2>
            <ul>
                <li>Focus on the best-performing model configuration for live trading</li>
                <li>Consider ensemble methods combining multiple successful models</li>
                <li>Implement risk management strategies to limit maximum drawdown</li>
                <li>Regular model retraining with fresh market data</li>
                <li>Monitor performance in different market conditions</li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    # Save the HTML report
    report_path = os.path.join(output_dir, "comprehensive_report.html")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"Comprehensive HTML report saved to {report_path}")
    return report_path

def main():
    """Main function to run the Enhanced Forex Prediction Pipeline."""
    parser = argparse.ArgumentParser(description='Run Enhanced Forex Trend Prediction Pipeline')
    
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
    
    # Visualization options
    parser.add_argument('--skip-visualizations', action='store_true', help='Skip creating visualizations')
    parser.add_argument('--create-report', action='store_true', default=True, help='Create comprehensive HTML report')
    
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
            ],
            'create_visualizations': not args.skip_visualizations
        }
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(args.output_dir, "enhanced_run_config.json")
    save_config(config, config_path)
    logger.info(f"Saved configuration to {config_path}")
    
    # Initialize and run enhanced pipeline
    forex_pred = ForexPrediction(config)
    logger.info("Starting Enhanced Forex Prediction Pipeline")
    
    try:
        results = forex_pred.run_pipeline()
        logger.info("Enhanced pipeline completed successfully")
        
        # Save results
        results_path = os.path.join(args.output_dir, "enhanced_final_results.json")
        
        # Extract the trading performance for each model for easier comparison
        summary_results = {}
        for model_key, model_results in results.items():
            if 'trading_performance' in model_results:
                summary_results[model_key] = {
                    'trading_performance': model_results['trading_performance'],
                    'accuracy': model_results.get('accuracy', 0),
                    'precision': model_results.get('precision', 0),
                    'recall': model_results.get('recall', 0),
                    'f1': model_results.get('f1', 0)
                }
        
        with open(results_path, 'w') as f:
            json.dump(summary_results, f, indent=2)
        
        logger.info(f"Enhanced results saved to {results_path}")
        
        # Print summary of best models
        best_model = max(summary_results.items(), key=lambda x: x[1]['trading_performance']['annual_return'])
        logger.info(f"🏆 Best model: {best_model[0]} - Annual Return: {best_model[1]['trading_performance']['annual_return']:.2f}%")
        
        # Compare single vs bagging if both are available
        if config['use_bagging']:
            logger.info("=== 📊 Single vs Bagging Comparison ===")
            for model_type in config['models_to_train']:
                avg_single_return = 0
                avg_bagging_return = 0
                count = 0
                
                for pair in config['currency_pairs']:
                    single_key = f"{pair}_{model_type}"
                    bagging_key = f"Bagging_{model_type}_{pair}"
                    
                    if single_key in summary_results and bagging_key in summary_results:
                        single_return = summary_results[single_key]['trading_performance']['annual_return']
                        bagging_return = summary_results[bagging_key]['trading_performance']['annual_return']
                        
                        improvement = bagging_return - single_return
                        improvement_icon = "📈" if improvement > 0 else "📉"
                        
                        logger.info(f"{improvement_icon} {pair} - {model_type}: Single={single_return:.2f}%, Bagging={bagging_return:.2f}%, Diff={improvement:.2f}%")
                        
                        avg_single_return += single_return
                        avg_bagging_return += bagging_return
                        count += 1
                
                if count > 0:
                    avg_single_return /= count
                    avg_bagging_return /= count
                    avg_improvement = avg_bagging_return - avg_single_return
                    improvement_icon = "🚀" if avg_improvement > 0 else "⚠️"
                    
                    logger.info(f"{improvement_icon} Average for {model_type}: Single={avg_single_return:.2f}%, Bagging={avg_bagging_return:.2f}%, Diff={avg_improvement:.2f}%")
        
        # Create comprehensive HTML report
        if args.create_report:
            logger.info("📊 Creating comprehensive HTML report...")
            report_path = create_comprehensive_report(summary_results, args.output_dir)
            logger.info(f"📋 Comprehensive report created: {report_path}")
        
        # Print visualization summary
        plots_dir = os.path.join(args.output_dir, "plots")
        if os.path.exists(plots_dir):
            plot_files = [f for f in os.listdir(plots_dir) if f.endswith('.png')]
            logger.info(f"📈 Created {len(plot_files)} visualization plots in {plots_dir}")
        
        # Print final summary
        print("\n" + "="*80)
        print("🎯 ENHANCED FOREX PREDICTION PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"📁 Results directory: {args.output_dir}")
        print(f"🏆 Best performing model: {best_model[0]}")
        print(f"💰 Best annual return: {best_model[1]['trading_performance']['annual_return']:.2f}%")
        print(f"📊 Total models evaluated: {len(summary_results)}")
        
        if args.create_report:
            print(f"📋 Comprehensive report: {report_path}")
        
        print("="*80)
        
    except Exception as e:
        logger.error(f"❌ Error running enhanced pipeline: {e}", exc_info=True)
        print(f"❌ Error running enhanced pipeline: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
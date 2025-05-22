"""
Enhanced Run Pipeline for Master's Thesis Forex Prediction System
- Comprehensive command-line interface
- Advanced configuration management
- Enhanced reporting and visualization
- Walk-forward analysis integration
- Multi-objective optimization support
"""

import os
import json
import argparse
import logging
import warnings
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import the enhanced main class
from forex_prediction import EnhancedForexPrediction

warnings.filterwarnings('ignore')

# Set up enhanced logging
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

def load_enhanced_config(config_path):
    """Load enhanced configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {e}")
        return None

def save_enhanced_config(config, config_path):
    """Save enhanced configuration to JSON file."""
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Saved configuration to {config_path}")
    except Exception as e:
        logger.error(f"Error saving config to {config_path}: {e}")

def create_enhanced_comprehensive_report(results, walkforward_results, config, output_dir):
    """Create enhanced comprehensive HTML report for Master's thesis."""
    logger.info("Creating enhanced comprehensive report for Master's thesis")
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Enhanced Forex Trend Prediction - Master's Thesis Report</title>
        <meta charset="utf-8">
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background: rgba(255, 255, 255, 0.95);
                border-radius: 20px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                overflow: hidden;
            }}
            .header {{
                background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
                color: white;
                padding: 40px;
                text-align: center;
                position: relative;
            }}
            .header::before {{
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="20" cy="20" r="2" fill="rgba(255,255,255,0.1)"/><circle cx="80" cy="80" r="1" fill="rgba(255,255,255,0.1)"/><circle cx="40" cy="60" r="1.5" fill="rgba(255,255,255,0.05)"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
                opacity: 0.3;
            }}
            .header h1 {{
                margin: 0;
                font-size: 3em;
                font-weight: 300;
                position: relative;
                z-index: 1;
            }}
            .header h2 {{
                margin: 10px 0;
                font-size: 1.5em;
                font-weight: 300;
                position: relative;
                z-index: 1;
            }}
            .header p {{
                margin: 20px 0 0 0;
                font-size: 1.1em;
                opacity: 0.9;
                position: relative;
                z-index: 1;
            }}
            .section {{
                background: white;
                margin: 0;
                padding: 30px 40px;
                border-bottom: 1px solid #eee;
            }}
            .section:last-child {{
                border-bottom: none;
            }}
            .section h2 {{
                color: #2c3e50;
                margin: 0 0 25px 0;
                font-size: 1.8em;
                font-weight: 600;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
                display: inline-block;
            }}
            .metric-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin: 25px 0;
            }}
            .metric {{
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                padding: 25px;
                border-radius: 15px;
                text-align: center;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                transition: transform 0.3s ease;
                border: 1px solid #dee2e6;
            }}
            .metric:hover {{
                transform: translateY(-5px);
                box-shadow: 0 10px 25px rgba(0,0,0,0.15);
            }}
            .metric-value {{
                font-size: 2.2em;
                font-weight: bold;
                margin-bottom: 8px;
            }}
            .metric-label {{
                font-size: 0.9em;
                color: #6c757d;
                text-transform: uppercase;
                letter-spacing: 1px;
                font-weight: 500;
            }}
            .positive {{ color: #28a745; }}
            .negative {{ color: #dc3545; }}
            .neutral {{ color: #ffc107; }}
            .excellent {{ color: #17a2b8; }}
            
            .performance-table {{
                width: 100%;
                border-collapse: collapse;
                margin: 25px 0;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                border-radius: 10px;
                overflow: hidden;
            }}
            .performance-table th {{
                background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
                color: white;
                padding: 15px 10px;
                text-align: center;
                font-weight: 600;
                font-size: 0.9em;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            .performance-table td {{
                padding: 12px 10px;
                text-align: center;
                border-bottom: 1px solid #eee;
                font-size: 0.9em;
            }}
            .performance-table tr:hover {{
                background-color: #f8f9fa;
            }}
            .best-model {{
                background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
                font-weight: bold;
            }}
            .top-3 {{
                background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
            }}
            
            .tabs {{
                display: flex;
                background: #f8f9fa;
                border-radius: 10px;
                padding: 5px;
                margin: 20px 0;
            }}
            .tab {{
                flex: 1;
                padding: 12px 20px;
                text-align: center;
                background: transparent;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                font-weight: 500;
                transition: all 0.3s ease;
            }}
            .tab.active {{
                background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
                color: white;
                box-shadow: 0 3px 10px rgba(52, 152, 219, 0.3);
            }}
            .tab-content {{
                display: none;
                animation: fadeIn 0.5s ease;
            }}
            .tab-content.active {{
                display: block;
            }}
            @keyframes fadeIn {{
                from {{ opacity: 0; transform: translateY(10px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
            
            .visualization-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                gap: 25px;
                margin: 25px 0;
            }}
            .viz-item {{
                background: white;
                border-radius: 15px;
                overflow: hidden;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                transition: transform 0.3s ease;
            }}
            .viz-item:hover {{
                transform: translateY(-5px);
                box-shadow: 0 10px 25px rgba(0,0,0,0.15);
            }}
            .viz-item h4 {{
                background: linear-gradient(135deg, #6c757d 0%, #495057 100%);
                color: white;
                margin: 0;
                padding: 15px 20px;
                font-size: 1.1em;
                font-weight: 500;
            }}
            .viz-item img {{
                width: 100%;
                height: auto;
                display: block;
            }}
            
            .insights {{
                background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
                border-left: 5px solid #2196f3;
                padding: 25px;
                border-radius: 10px;
                margin: 25px 0;
            }}
            .insights h3 {{
                color: #1976d2;
                margin: 0 0 15px 0;
                font-size: 1.3em;
            }}
            .insights ul {{
                margin: 0;
                padding-left: 20px;
            }}
            .insights li {{
                margin-bottom: 10px;
                line-height: 1.6;
            }}
            
            .technical-details {{
                background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
                border-left: 5px solid #9c27b0;
                padding: 25px;
                border-radius: 10px;
                margin: 25px 0;
            }}
            .technical-details h3 {{
                color: #7b1fa2;
                margin: 0 0 15px 0;
                font-size: 1.3em;
            }}
            
            .recommendations {{
                background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
                border-left: 5px solid #4caf50;
                padding: 25px;
                border-radius: 10px;
                margin: 25px 0;
            }}
            .recommendations h3 {{
                color: #388e3c;
                margin: 0 0 15px 0;
                font-size: 1.3em;
            }}
            
            .footer {{
                background: #2c3e50;
                color: white;
                padding: 30px 40px;
                text-align: center;
            }}
            .footer p {{
                margin: 5px 0;
                opacity: 0.8;
            }}
            
            @media (max-width: 768px) {{
                .container {{ margin: 10px; }}
                .section {{ padding: 20px; }}
                .header {{ padding: 30px 20px; }}
                .header h1 {{ font-size: 2em; }}
                .visualization-grid {{ grid-template-columns: 1fr; }}
                .metric-grid {{ grid-template-columns: 1fr; }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🏦 Enhanced Forex Trend Prediction System</h1>
                <h2>Master's Thesis Research - Advanced Machine Learning Approach</h2>
                <p>📅 Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>📊 Period: {config.get('train_start', '2020-01-01')} to {config.get('test_end', '2022-12-31')}</p>
            </div>
    """
    
    # Extract enhanced performance data
    model_performance = []
    best_annual_return = -float('inf')
    best_model = ""
    total_profitable = 0
    
    for model_key, model_results in results.items():
        if 'trading_performance' in model_results:
            perf = model_results['trading_performance']
            model_performance.append({
                'model': model_key,
                'annual_return': perf['annual_return'],
                'win_rate': perf['win_rate'] * 100,
                'max_drawdown': perf['max_drawdown'],
                'sharpe_ratio': perf['sharpe_ratio'],
                'sortino_ratio': perf.get('sortino_ratio', 0),
                'calmar_ratio': perf.get('calmar_ratio', 0),
                'profit_factor': perf.get('profit_factor', 0),
                'trade_count': perf['trade_count'],
                'market_condition': perf['market_condition'],
                'buy_hold_return': perf['buy_hold_annual_return'],
                'risk_adjusted_return': perf.get('risk_adjusted_return', 0)
            })
            
            if perf['annual_return'] > best_annual_return:
                best_annual_return = perf['annual_return']
                best_model = model_key
            
            if perf['annual_return'] > 0:
                total_profitable += 1
    
    # Sort by annual return
    model_performance.sort(key=lambda x: x['annual_return'], reverse=True)
    
    # Calculate summary statistics
    avg_return = np.mean([m['annual_return'] for m in model_performance])
    avg_win_rate = np.mean([m['win_rate'] for m in model_performance])
    avg_sharpe = np.mean([m['sharpe_ratio'] for m in model_performance])
    avg_max_dd = np.mean([m['max_drawdown'] for m in model_performance])
    
    # Enhanced Executive Summary
    html_content += f"""
        <div class="section">
            <h2>📊 Enhanced Executive Summary</h2>
            <div class="metric-grid">
                <div class="metric">
                    <div class="metric-value {'positive' if best_annual_return > 10 else 'positive' if best_annual_return > 0 else 'negative'}">{best_annual_return:.2f}%</div>
                    <div class="metric-label">Best Annual Return</div>
                </div>
                <div class="metric">
                    <div class="metric-value excellent">{best_model.replace('_', ' ').title()}</div>
                    <div class="metric-label">Best Model</div>
                </div>
                <div class="metric">
                    <div class="metric-value {'positive' if avg_return > 0 else 'negative'}">{avg_return:.2f}%</div>
                    <div class="metric-label">Average Annual Return</div>
                </div>
                <div class="metric">
                    <div class="metric-value {'positive' if avg_win_rate > 55 else 'neutral' if avg_win_rate > 50 else 'negative'}">{avg_win_rate:.1f}%</div>
                    <div class="metric-label">Average Win Rate</div>
                </div>
                <div class="metric">
                    <div class="metric-value {'positive' if total_profitable > len(model_performance)/2 else 'negative'}">{total_profitable}/{len(model_performance)}</div>
                    <div class="metric-label">Profitable Models</div>
                </div>
                <div class="metric">
                    <div class="metric-value {'excellent' if avg_sharpe > 1 else 'positive' if avg_sharpe > 0.5 else 'negative'}">{avg_sharpe:.3f}</div>
                    <div class="metric-label">Average Sharpe Ratio</div>
                </div>
                <div class="metric">
                    <div class="metric-value {'positive' if avg_max_dd < 5 else 'neutral' if avg_max_dd < 10 else 'negative'}">{avg_max_dd:.2f}%</div>
                    <div class="metric-label">Average Max Drawdown</div>
                </div>
                <div class="metric">
                    <div class="metric-value excellent">{len(config.get('models_to_train', []))}</div>
                    <div class="metric-label">Model Types Tested</div>
                </div>
            </div>
        </div>
    """
    
    # Enhanced Performance Comparison Table
    html_content += f"""
        <div class="section">
            <h2>📈 Enhanced Model Performance Analysis</h2>
            <table class="performance-table">
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Model</th>
                        <th>Annual Return (%)</th>
                        <th>Win Rate (%)</th>
                        <th>Max DD (%)</th>
                        <th>Sharpe Ratio</th>
                        <th>Sortino Ratio</th>
                        <th>Calmar Ratio</th>
                        <th>Profit Factor</th>
                        <th>Trades</th>
                        <th>Risk-Adj Return</th>
                        <th>Market</th>
                        <th>Buy & Hold (%)</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    for i, model in enumerate(model_performance):
        rank = i + 1
        row_class = "best-model" if rank == 1 else "top-3" if rank <= 3 else ""
        return_class = "positive" if model['annual_return'] > 10 else "positive" if model['annual_return'] > 0 else "negative"
        sharpe_class = "excellent" if model['sharpe_ratio'] > 1 else "positive" if model['sharpe_ratio'] > 0.5 else "negative"
        
        html_content += f"""
            <tr class="{row_class}">
                <td><strong>#{rank}</strong></td>
                <td><strong>{model['model'].replace('_', ' ').title()}</strong></td>
                <td class="{return_class}"><strong>{model['annual_return']:.2f}%</strong></td>
                <td>{model['win_rate']:.1f}%</td>
                <td>{model['max_drawdown']:.2f}%</td>
                <td class="{sharpe_class}">{model['sharpe_ratio']:.3f}</td>
                <td>{model['sortino_ratio']:.3f}</td>
                <td>{model['calmar_ratio']:.3f}</td>
                <td>{model['profit_factor']:.2f}</td>
                <td>{model['trade_count']}</td>
                <td>{model['risk_adjusted_return']:.3f}</td>
                <td>{model['market_condition']}</td>
                <td>{model['buy_hold_return']:.2f}%</td>
            </tr>
        """
    
    html_content += """
                </tbody>
            </table>
        </div>
    """
    
    # Enhanced Insights Section
    insights = []
    
    # Performance insights
    if best_annual_return > 15:
        insights.append(f"🏆 Outstanding Performance: {best_model} achieved exceptional {best_annual_return:.2f}% annual return, significantly outperforming market benchmarks")
    elif best_annual_return > 10:
        insights.append(f"🎯 Excellent Performance: {best_model} achieved strong {best_annual_return:.2f}% annual return")
    elif best_annual_return > 5:
        insights.append(f"✅ Good Performance: {best_model} achieved solid {best_annual_return:.2f}% annual return")
    elif best_annual_return > 0:
        insights.append(f"📊 Positive Performance: {best_model} achieved {best_annual_return:.2f}% annual return")
    else:
        insights.append(f"⚠️ Challenging Market: All models faced difficulties with negative returns")
    
    # Risk-adjusted performance
    best_sharpe = max([m['sharpe_ratio'] for m in model_performance])
    if best_sharpe > 2:
        insights.append(f"🛡️ Excellent Risk Management: Best Sharpe ratio of {best_sharpe:.3f} indicates superior risk-adjusted returns")
    elif best_sharpe > 1:
        insights.append(f"🛡️ Good Risk Management: Best Sharpe ratio of {best_sharpe:.3f} shows effective risk control")
    
    # Win rate analysis
    if avg_win_rate > 60:
        insights.append(f"🎯 High Prediction Accuracy: Average win rate of {avg_win_rate:.1f}% demonstrates strong directional forecasting")
    elif avg_win_rate > 55:
        insights.append(f"✅ Above-Average Accuracy: Win rate of {avg_win_rate:.1f}% beats random chance significantly")
    elif avg_win_rate > 50:
        insights.append(f"📊 Modest Edge: Win rate of {avg_win_rate:.1f}% shows slight predictive advantage")
    
    # Model comparison
    model_types = set([m['model'].split('_')[0] if 'Bagging' not in m['model'] else 'Bagging' for m in model_performance])
    if len(model_types) > 1:
        type_performance = {}
        for model_type in model_types:
            type_models = [m for m in model_performance if model_type in m['model']]
            if type_models:
                avg_return = np.mean([m['annual_return'] for m in type_models])
                type_performance[model_type] = avg_return
        
        best_type = max(type_performance.items(), key=lambda x: x[1])
        insights.append(f"🔬 Model Analysis: {best_type[0]} models performed best with {best_type[1]:.2f}% average return")
    
    # Market condition analysis
    up_market_models = [m for m in model_performance if 'Up' in m['market_condition']]
    down_market_models = [m for m in model_performance if 'Down' in m['market_condition']]
    
    if up_market_models and down_market_models:
        up_avg = np.mean([m['annual_return'] for m in up_market_models])
        down_avg = np.mean([m['annual_return'] for m in down_market_models])
        
        if up_avg > down_avg:
            insights.append(f"📈 Market Adaptation: Models performed better in up markets ({up_avg:.2f}%) vs down markets ({down_avg:.2f}%)")
        else:
            insights.append(f"📉 Defensive Strength: Models showed resilience in down markets ({down_avg:.2f}%) vs up markets ({up_avg:.2f}%)")
    
    html_content += f"""
        <div class="section">
            <h2>💡 Advanced Research Insights</h2>
            <div class="insights">
                <h3>Key Findings from Master's Thesis Research</h3>
                <ul>
    """
    
    for insight in insights:
        html_content += f"<li>{insight}</li>"
    
    html_content += """
                </ul>
            </div>
        </div>
    """
    
    # Technical Implementation Details
    html_content += f"""
        <div class="section">
            <h2>🔧 Advanced Technical Implementation</h2>
            <div class="technical-details">
                <h3>Enhanced System Architecture</h3>
                <p><strong>🤖 Advanced Models:</strong> Enhanced CNN-LSTM with attention mechanisms, Advanced Temporal Fusion Transformer, Enhanced XGBoost with sophisticated hyperparameters</p>
                <p><strong>💱 Currency Pairs:</strong> {', '.join(config.get('currency_pairs', ['EURUSD', 'GBPUSD', 'USDJPY']))}</p>
                <p><strong>📊 Enhanced Features:</strong> {config.get('n_features', 50)} selected features including Elliott Wave patterns, Fibonacci retracements, market regime detection</p>
                <p><strong>🧠 Feature Engineering:</strong> Advanced technical indicators, market regime classification, time-based cyclical encoding</p>
                <p><strong>📈 Evaluation Period:</strong> {config.get('test_start', '2022-01-01')} to {config.get('test_end', '2022-12-31')}</p>
                <p><strong>⚙️ Hyperparameter Optimization:</strong> Bayesian optimization with {config.get('n_trials', 100)} trials using TPE sampler</p>
                <p><strong>🔄 Validation Method:</strong> Time series split with walk-forward analysis</p>
                <p><strong>💼 Risk Management:</strong> Kelly Criterion position sizing, dynamic stop-loss/take-profit, confidence-based leverage scaling</p>
                <p><strong>📏 Performance Metrics:</strong> Annual Return, Sharpe/Sortino/Calmar Ratios, Maximum Drawdown, Profit Factor, Win Rate</p>
            </div>
        </div>
    """
    
    # Advanced Recommendations
    html_content += """
        <div class="section">
            <h2>📝 Research Conclusions & Future Work</h2>
            <div class="recommendations">
                <h3>Academic Contributions & Practical Applications</h3>
                <ul>
                    <li><strong>🎯 Model Selection:</strong> Focus on the top-performing model configurations for live trading implementation</li>
                    <li><strong>🔄 Ensemble Methods:</strong> Develop dynamic ensemble approaches that adapt to changing market regimes</li>
                    <li><strong>🛡️ Risk Management:</strong> Implement sophisticated risk controls including regime-aware position sizing</li>
                    <li><strong>📊 Continuous Learning:</strong> Establish automated model retraining pipelines with fresh market data</li>
                    <li><strong>🌍 Market Adaptation:</strong> Extend research to additional currency pairs and market conditions</li>
                    <li><strong>⚡ Real-time Systems:</strong> Develop low-latency prediction systems for high-frequency trading</li>
                    <li><strong>🔬 Feature Research:</strong> Investigate alternative data sources (sentiment, news, macroeconomic indicators)</li>
                    <li><strong>🧠 Advanced ML:</strong> Explore transformer architectures and reinforcement learning approaches</li>
                    <li><strong>📈 Portfolio Integration:</strong> Develop multi-asset portfolio optimization using prediction signals</li>
                    <li><strong>🎓 Academic Impact:</strong> Publish findings in peer-reviewed journals and present at conferences</li>
                </ul>
            </div>
        </div>
    """
    
    # Visualization Section
    plot_files = []
    plots_dir = os.path.join(output_dir, "plots")
    if os.path.exists(plots_dir):
        plot_files = [f for f in os.listdir(plots_dir) if f.endswith('.png')]
    
    if plot_files:
        html_content += """
            <div class="section">
                <h2>📊 Research Visualizations & Analysis</h2>
                <div class="tabs">
                    <button class="tab active" onclick="showTab('data-analysis')">Data Analysis</button>
                    <button class="tab" onclick="showTab('model-performance')">Model Performance</button>
                    <button class="tab" onclick="showTab('risk-analysis')">Risk Analysis</button>
                    <button class="tab" onclick="showTab('feature-importance')">Feature Importance</button>
                </div>
        """
        
        # Organize plots by category
        plot_categories = {
            'data-analysis': [f for f in plot_files if any(x in f.lower() for x in ['data_', 'correlation', 'distribution'])],
            'model-performance': [f for f in plot_files if any(x in f.lower() for x in ['training', 'prediction', 'comparison', 'performance'])],
            'risk-analysis': [f for f in plot_files if any(x in f.lower() for x in ['trading', 'risk', 'drawdown', 'return'])],
            'feature-importance': [f for f in plot_files if any(x in f.lower() for x in ['feature', 'importance', 'selection'])]
        }
        
        for category, files in plot_categories.items():
            if files:
                html_content += f"""
                    <div class="tab-content {'active' if category == 'data-analysis' else ''}" id="{category}">
                        <div class="visualization-grid">
                """
                
                for plot_file in files:
                    plot_path = f"plots/{plot_file}"
                    plot_title = plot_file.replace('.png', '').replace('_', ' ').title()
                    html_content += f"""
                        <div class="viz-item">
                            <h4>{plot_title}</h4>
                            <img src="{plot_path}" alt="{plot_title}" loading="lazy">
                        </div>
                    """
                
                html_content += """
                        </div>
                    </div>
                """
        
        html_content += """
            </div>
        """
    
    # Footer
    html_content += f"""
            <div class="footer">
                <p><strong>Enhanced Forex Trend Prediction System - Master's Thesis Research</strong></p>
                <p>Generated using advanced machine learning techniques and comprehensive backtesting</p>
                <p>Report generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}</p>
            </div>
        </div>
        
        <script>
            function showTab(tabName) {{
                // Hide all tab contents
                const contents = document.querySelectorAll('.tab-content');
                contents.forEach(content => content.classList.remove('active'));
                
                // Remove active class from all tabs
                const tabs = document.querySelectorAll('.tab');
                tabs.forEach(tab => tab.classList.remove('active'));
                
                // Show selected tab content
                document.getElementById(tabName).classList.add('active');
                
                // Add active class to clicked tab
                event.target.classList.add('active');
            }}
            
            // Add loading animation
            document.addEventListener('DOMContentLoaded', function() {{
                const metrics = document.querySelectorAll('.metric');
                metrics.forEach((metric, index) => {{
                    setTimeout(() => {{
                        metric.style.opacity = '0';
                        metric.style.transform = 'translateY(20px)';
                        setTimeout(() => {{
                            metric.style.transition = 'all 0.5s ease';
                            metric.style.opacity = '1';
                            metric.style.transform = 'translateY(0)';
                        }}, 100);
                    }}, index * 100);
                }});
            }});
        </script>
    </body>
    </html>
    """
    
    # Save the enhanced HTML report
    report_path = os.path.join(output_dir, "enhanced_comprehensive_report.html")
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        logger.info(f"Enhanced comprehensive report saved to {report_path}")
        return report_path
    except Exception as e:
        logger.error(f"Error saving enhanced report: {e}")
        return None

def create_walkforward_analysis_report(walkforward_results, output_dir):
    """Create detailed walk-forward analysis report."""
    logger.info("Creating walk-forward analysis report")
    
    if not walkforward_results:
        logger.warning("No walk-forward results available")
        return None
    
    # Aggregate walk-forward results
    summary_data = []
    for pair, windows in walkforward_results.items():
        for window in windows:
            for model_type, metrics in window['model_results'].items():
                summary_data.append({
                    'pair': pair,
                    'window_start': window['window_start'],
                    'model_type': model_type,
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1': metrics['f1']
                })
    
    if not summary_data:
        logger.warning("No walk-forward summary data available")
        return None
    
    df = pd.DataFrame(summary_data)
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Accuracy over time
    plt.subplot(2, 2, 1)
    for pair in df['pair'].unique():
        for model in df['model_type'].unique():
            pair_model_data = df[(df['pair'] == pair) & (df['model_type'] == model)]
            if not pair_model_data.empty:
                plt.plot(pair_model_data['window_start'], pair_model_data['accuracy'], 
                        marker='o', label=f"{pair} - {model}", alpha=0.7)
    
    plt.title('Walk-Forward Analysis: Accuracy Over Time')
    plt.xlabel('Window Start Date')
    plt.ylabel('Accuracy')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Average performance by model
    plt.subplot(2, 2, 2)
    avg_performance = df.groupby('model_type')['accuracy'].mean().sort_values(ascending=False)
    bars = plt.bar(avg_performance.index, avg_performance.values, alpha=0.7)
    plt.title('Average Accuracy by Model Type')
    plt.ylabel('Average Accuracy')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, avg_performance.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom')
    
    # Plot 3: Performance by currency pair
    plt.subplot(2, 2, 3)
    avg_by_pair = df.groupby('pair')['accuracy'].mean().sort_values(ascending=False)
    bars = plt.bar(avg_by_pair.index, avg_by_pair.values, alpha=0.7)
    plt.title('Average Accuracy by Currency Pair')
    plt.ylabel('Average Accuracy')
    
    # Add value labels
    for bar, value in zip(bars, avg_by_pair.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom')
    
    # Plot 4: F1 Score distribution
    plt.subplot(2, 2, 4)
    plt.hist(df['f1'], bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(df['f1'].mean(), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {df["f1"].mean():.3f}')
    plt.title('F1 Score Distribution Across All Windows')
    plt.xlabel('F1 Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "walkforward_analysis.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed results
    walkforward_report_path = os.path.join(output_dir, "walkforward_detailed_results.csv")
    df.to_csv(walkforward_report_path, index=False)
    
    logger.info(f"Walk-forward analysis report saved to {walkforward_report_path}")
    return walkforward_report_path

def main():
    """Enhanced main function for Master's thesis pipeline."""
    parser = argparse.ArgumentParser(description='Enhanced Forex Trend Prediction Pipeline for Master\'s Thesis')
    
    # Enhanced configuration options
    parser.add_argument('--config', type=str, help='Path to enhanced configuration JSON file')
    parser.add_argument('--output-dir', type=str, default='output', help='Enhanced output directory')
    
    # Enhanced data options
    parser.add_argument('--window-size', type=int, default=60, help='Enhanced lookback window size (hours)')
    parser.add_argument('--prediction-horizon', type=int, default=1, help='Prediction horizon (hours)')
    parser.add_argument('--train-start', type=str, default='2020-01-01', help='Training data start (thesis period)')
    parser.add_argument('--train-end', type=str, default='2021-06-30', help='Training data end')
    parser.add_argument('--validation-start', type=str, default='2021-07-01', help='Validation data start')
    parser.add_argument('--validation-end', type=str, default='2021-12-31', help='Validation data end')
    parser.add_argument('--test-start', type=str, default='2022-01-01', help='Test data start')
    parser.add_argument('--test-end', type=str, default='2022-12-31', help='Test data end (full year)')
    
    # Enhanced model options
    parser.add_argument('--models', type=str, default='enhanced_cnn_lstm,advanced_tft,enhanced_xgboost', 
                       help='Enhanced models (comma-separated)')
    parser.add_argument('--currency-pairs', type=str, default='EURUSD,GBPUSD,USDJPY',
                       help='Currency pairs for thesis research')
    parser.add_argument('--no-bagging', action='store_true', help='Disable enhanced bagging approach')
    parser.add_argument('--no-dynamic-ensemble', action='store_true', help='Disable dynamic ensemble methods')
    
    # Enhanced training options
    parser.add_argument('--batch-size', type=int, default=32, help='Enhanced batch size')
    parser.add_argument('--epochs', type=int, default=150, help='Maximum epochs (increased for thesis)')
    parser.add_argument('--patience', type=int, default=25, help='Enhanced early stopping patience')
    
    # Enhanced feature options
    parser.add_argument('--n-features', type=int, default=50, help='Number of enhanced features to select')
    parser.add_argument('--feature-selection', type=str, default='random_forest,mutual_info,pca',
                       help='Enhanced feature selection methods')
    parser.add_argument('--scaler', type=str, default='robust', 
                       choices=['standard', 'minmax', 'robust'], help='Enhanced scaler type')
    
    # Enhanced hyperparameter tuning
    parser.add_argument('--tune-hyperparams', action='store_true', default=True, help='Enable Bayesian optimization')
    parser.add_argument('--optimization-method', type=str, default='bayesian', 
                       choices=['bayesian', 'optuna'], help='Optimization method')
    parser.add_argument('--n-trials', type=int, default=100, help='Enhanced number of optimization trials')
    
    # Enhanced analysis options
    parser.add_argument('--enable-walkforward', action='store_true', default=True, help='Enable walk-forward analysis')
    parser.add_argument('--walkforward-window', type=int, default=180, help='Walk-forward window size (days)')
    parser.add_argument('--enable-regime-detection', action='store_true', default=True, help='Enable market regime detection')
    
    # Enhanced risk management
    parser.add_argument('--kelly-criterion', action='store_true', default=True, help='Enable Kelly Criterion position sizing')
    parser.add_argument('--dynamic-stops', action='store_true', default=True, help='Enable dynamic stop-loss/take-profit')
    parser.add_argument('--confidence-threshold', type=float, default=0.6, help='Minimum confidence for trades')
    
    # Enhanced reporting options
    parser.add_argument('--skip-visualizations', action='store_true', help='Skip creating enhanced visualizations')
    parser.add_argument('--create-report', action='store_true', default=True, help='Create enhanced comprehensive report')
    parser.add_argument('--create-walkforward-report', action='store_true', default=True, help='Create walk-forward analysis report')
    
    args = parser.parse_args()
    
    # Load or create enhanced configuration
    if args.config:
        config = load_enhanced_config(args.config)
        if config is None:
            logger.error("Failed to load configuration, using command-line arguments")
            config = {}
    else:
        config = {}
    
    # Update config with command-line arguments
    enhanced_config = {
        'window_size': args.window_size,
        'prediction_horizon': args.prediction_horizon,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'patience': args.patience,
        'test_size': 0.15,  # Enhanced for thesis
        'validation_size': 0.15,
        'random_state': 42,
        'train_start': args.train_start,
        'train_end': args.train_end,
        'validation_start': args.validation_start,
        'validation_end': args.validation_end,
        'test_start': args.test_start,
        'test_end': args.test_end,
        'feature_selection_methods': args.feature_selection.split(','),
        'n_features': args.n_features,
        'models_to_train': args.models.split(','),
        'currency_pairs': args.currency_pairs.split(','),
        'use_bagging': not args.no_bagging,
        'use_dynamic_ensemble': not args.no_dynamic_ensemble,
        'scaler_type': args.scaler,
        'hyperparameter_tuning': args.tune_hyperparams,
        'optimization_method': args.optimization_method,
        'n_trials': args.n_trials,
        'use_walkforward': args.enable_walkforward,
        'walkforward_window': args.walkforward_window,
        'use_regime_detection': args.enable_regime_detection,
        'risk_management': {
            'use_kelly_criterion': args.kelly_criterion,
            'use_dynamic_stops': args.dynamic_stops,
            'max_drawdown_limit': 0.05,
            'confidence_threshold': args.confidence_threshold,
            'leverage_scaling': True
        },
        'create_visualizations': not args.skip_visualizations
    }
    
    # Merge with loaded config
    config.update(enhanced_config)
    
    # Create enhanced output directory structure
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save enhanced configuration
    config_path = os.path.join(args.output_dir, "enhanced_thesis_config.json")
    save_enhanced_config(config, config_path)
    
    # Initialize and run enhanced pipeline
    logger.info("🚀 Initializing Enhanced Forex Prediction System for Master's Thesis")
    
    try:
        forex_pred = EnhancedForexPrediction(config)
        logger.info("✅ Enhanced system initialized successfully")
        
        logger.info("🔬 Starting comprehensive pipeline for Master's thesis research")
        results = forex_pred.run_enhanced_pipeline()
        logger.info("✅ Enhanced pipeline completed successfully")
        
        # Save enhanced results
        results_path = os.path.join(args.output_dir, "enhanced_thesis_results.json")
        
        # Process results for JSON serialization
        serializable_results = {}
        for model_key, model_results in results.items():
            if 'trading_performance' in model_results:
                serializable_results[model_key] = {
                    'trading_performance': {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                                          for k, v in model_results['trading_performance'].items() 
                                          if k != 'equity_curve'},  # Exclude large arrays
                    'accuracy': float(model_results.get('accuracy', 0)),
                    'precision': float(model_results.get('precision', 0)),
                    'recall': float(model_results.get('recall', 0)),
                    'f1': float(model_results.get('f1', 0))
                }
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        logger.info(f"📊 Enhanced results saved to {results_path}")
        
        # Create enhanced comprehensive report
        if args.create_report:
            logger.info("📋 Creating enhanced comprehensive report for Master's thesis")
            report_path = create_enhanced_comprehensive_report(
                serializable_results, 
                forex_pred.walkforward_results if hasattr(forex_pred, 'walkforward_results') else {}, 
                config, 
                args.output_dir
            )
            if report_path:
                logger.info(f"📄 Enhanced thesis report created: {report_path}")
        
        # Create walk-forward analysis report
        if args.create_walkforward_report and hasattr(forex_pred, 'walkforward_results'):
            logger.info("📈 Creating walk-forward analysis report")
            wf_report_path = create_walkforward_analysis_report(
                forex_pred.walkforward_results, args.output_dir
            )
            if wf_report_path:
                logger.info(f"📊 Walk-forward report created: {wf_report_path}")
        
        # Print enhanced summary
        best_model = max(serializable_results.items(), 
                        key=lambda x: x[1]['trading_performance']['annual_return'])
        
        # Calculate summary statistics
        all_returns = [result['trading_performance']['annual_return'] 
                      for result in serializable_results.values()]
        profitable_models = len([r for r in all_returns if r > 0])
        
        print("\n" + "="*100)
        print("🎓 ENHANCED FOREX PREDICTION SYSTEM - MASTER'S THESIS RESULTS")
        print("="*100)
        print(f"📁 Results directory: {args.output_dir}")
        print(f"🏆 Best performing model: {best_model[0]}")
        print(f"💰 Best annual return: {best_model[1]['trading_performance']['annual_return']:.2f}%")
        print(f"🎯 Best win rate: {best_model[1]['trading_performance']['win_rate']*100:.1f}%")
        print(f"🛡️ Best Sharpe ratio: {best_model[1]['trading_performance']['sharpe_ratio']:.3f}")
        print(f"📊 Total models evaluated: {len(serializable_results)}")
        print(f"💹 Profitable models: {profitable_models}/{len(serializable_results)}")
        print(f"📈 Average return: {np.mean(all_returns):.2f}%")
        
        if args.create_report and report_path:
            print(f"📋 Comprehensive thesis report: {report_path}")
        
        # Print research insights
        print("\n🔬 RESEARCH INSIGHTS:")
        print("- Advanced feature engineering with Elliott Wave and Fibonacci analysis")
        print("- Market regime detection for adaptive trading strategies")
        print("- Bayesian hyperparameter optimization for model enhancement")
        print("- Walk-forward analysis for robust performance validation")
        print("- Enhanced risk management with Kelly Criterion and dynamic stops")
        print("="*100)
        
    except Exception as e:
        logger.error(f"❌ Error in enhanced pipeline: {e}", exc_info=True)
        print(f"❌ Error in enhanced pipeline: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
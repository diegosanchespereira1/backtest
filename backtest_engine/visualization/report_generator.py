#!/usr/bin/env python3
"""
Report Generator Module
Provides visualization and reporting functionality for backtesting results
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import json
from datetime import datetime


class ReportGenerator:
    """Generates reports and visualizations for backtesting results"""
    
    def __init__(self):
        pass
    
    def generate_report(self, results: dict, metrics: dict, output_dir: str):
        """
        Generate a complete backtest report
        
        Args:
            results: Backtest results dictionary
            metrics: Performance metrics dictionary
            output_dir: Directory to save the report
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate equity curve chart
        self._generate_equity_chart(results, output_dir)
        
        # Generate drawdown chart
        self._generate_drawdown_chart(results, output_dir)
        
        # Generate trade analysis chart
        self._generate_trade_analysis_chart(results, output_dir)
        
        # Save metrics to JSON
        metrics_path = os.path.join(output_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
            
        # Generate HTML report
        self._generate_html_report(results, metrics, output_dir)
    
    def generate_comparison_report(self, results: dict, output_dir: str):
        """
        Generate a comparison report for multiple strategies
        
        Args:
            results: Dictionary of strategy results
            output_dir: Directory to save the comparison report
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate comparison equity chart
        self._generate_comparison_equity_chart(results, output_dir)
        
        # Save comparison metrics
        comparison_metrics = {}
        for strategy_name, result in results.items():
            analyzer = PerformanceAnalyzer()
            metrics = analyzer.calculate_metrics(result)
            comparison_metrics[strategy_name] = metrics
            
        metrics_path = os.path.join(output_dir, 'comparison_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(comparison_metrics, f, indent=2, default=str)
    
    def _generate_equity_chart(self, results: dict, output_dir: str):
        """Generate equity curve chart"""
        if 'portfolio_values' not in results or results['portfolio_values'].empty:
            return
            
        pv_df = results['portfolio_values']
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pv_df['date'],
            y=pv_df['portfolio_value'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title="Equity Curve",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            hovermode='x unified'
        )
        
        fig.write_html(os.path.join(output_dir, 'equity_curve.html'))
    
    def _generate_drawdown_chart(self, results: dict, output_dir: str):
        """Generate drawdown chart"""
        if 'portfolio_values' not in results or results['portfolio_values'].empty:
            return
            
        pv_df = results['portfolio_values']
        roll_max = pv_df['portfolio_value'].cummax()
        drawdown = (pv_df['portfolio_value'] - roll_max) / roll_max * 100
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pv_df['date'],
            y=drawdown,
            mode='lines',
            fill='tozeroy',
            name='Drawdown %',
            line=dict(color='red'),
            fillcolor='rgba(255,0,0,0.3)'
        ))
        
        fig.update_layout(
            title="Drawdown Chart",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            hovermode='x unified'
        )
        
        fig.write_html(os.path.join(output_dir, 'drawdown_chart.html'))
    
    def _generate_trade_analysis_chart(self, results: dict, output_dir: str):
        """Generate trade analysis chart"""
        if 'trades' not in results or results['trades'].empty:
            return
            
        trades_df = results['trades']
        sell_trades = trades_df[trades_df['action'] == 'SELL'].copy()
        
        if sell_trades.empty:
            return
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Trade P&L Distribution', 'Cumulative P&L'),
            specs=[[{"type": "histogram"}, {"type": "scatter"}]]
        )
        
        # P&L histogram
        fig.add_trace(
            go.Histogram(
                x=sell_trades['pnl'],
                nbinsx=20,
                name='P&L Distribution',
                marker_color='lightblue'
            ),
            row=1, col=1
        )
        
        # Cumulative P&L
        sell_trades = sell_trades.sort_values('date')
        sell_trades['cumulative_pnl'] = sell_trades['pnl'].cumsum()
        
        fig.add_trace(
            go.Scatter(
                x=sell_trades['date'],
                y=sell_trades['cumulative_pnl'],
                mode='lines+markers',
                name='Cumulative P&L',
                line=dict(color='green')
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text="Trade Analysis",
            showlegend=False
        )
        
        fig.write_html(os.path.join(output_dir, 'trade_analysis.html'))
    
    def _generate_comparison_equity_chart(self, results: dict, output_dir: str):
        """Generate comparison equity chart for multiple strategies"""
        fig = go.Figure()
        
        for strategy_name, result in results.items():
            if 'portfolio_values' in result and not result['portfolio_values'].empty:
                pv_df = result['portfolio_values']
                fig.add_trace(go.Scatter(
                    x=pv_df['date'],
                    y=pv_df['portfolio_value'],
                    mode='lines',
                    name=strategy_name
                ))
        
        fig.update_layout(
            title="Strategy Comparison - Equity Curves",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            hovermode='x unified'
        )
        
        fig.write_html(os.path.join(output_dir, 'strategy_comparison.html'))
    
    def _generate_html_report(self, results: dict, metrics: dict, output_dir: str):
        """Generate a simple HTML report"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Backtest Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .metric {{ background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Backtest Report</h1>
            <h2>Performance Metrics</h2>
            <div class="metric">
                <strong>Total Return:</strong> 
                <span class="{'positive' if metrics.get('total_return', 0) >= 0 else 'negative'}">
                    {metrics.get('total_return', 0):.2%}
                </span>
            </div>
            <div class="metric">
                <strong>CAGR:</strong> 
                <span class="{'positive' if metrics.get('cagr', 0) >= 0 else 'negative'}">
                    {metrics.get('cagr', 0):.2%}
                </span>
            </div>
            <div class="metric">
                <strong>Sharpe Ratio:</strong> {metrics.get('sharpe_ratio', 0):.2f}
            </div>
            <div class="metric">
                <strong>Max Drawdown:</strong> 
                <span class="negative">
                    {metrics.get('max_drawdown', 0):.2%}
                </span>
            </div>
            <div class="metric">
                <strong>Win Rate:</strong> {metrics.get('win_rate', 0):.2%}
            </div>
            <div class="metric">
                <strong>Total Trades:</strong> {metrics.get('total_trades', 0)}
            </div>
            
            <h2>Charts</h2>
            <p>See the individual chart files in this directory:</p>
            <ul>
                <li><a href="equity_curve.html">Equity Curve</a></li>
                <li><a href="drawdown_chart.html">Drawdown Chart</a></li>
                <li><a href="trade_analysis.html">Trade Analysis</a></li>
            </ul>
        </body>
        </html>
        """
        
        with open(os.path.join(output_dir, 'report.html'), 'w') as f:
            f.write(html_content)
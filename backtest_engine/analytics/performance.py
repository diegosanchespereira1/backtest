#!/usr/bin/env python3
"""
Performance Analyzer Module
Provides performance metrics calculation for backtesting results
"""

import pandas as pd
import numpy as np
from typing import Dict, Any


class PerformanceAnalyzer:
    """Calculates performance metrics from backtesting results"""
    
    def calculate_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate performance metrics from backtest results
        
        Args:
            results: Dictionary containing backtest results from BacktestRunner
            
        Returns:
            Dictionary of performance metrics
        """
        # For now, return basic metrics based on final portfolio value
        # In a full implementation, this would calculate CAGR, Sharpe ratio, etc.
        
        metrics = {}
        
        # If we have portfolio values, calculate simple return
        if 'portfolio_values' in results and not results['portfolio_values'].empty:
            pv_df = results['portfolio_values']
            start_value = pv_df['portfolio_value'].iloc[0]
            end_value = pv_df['portfolio_value'].iloc[-1]
            
            if start_value > 0:
                total_return = (end_value - start_value) / start_value
                metrics['total_return'] = total_return
                metrics['cagr'] = total_return  # Simplified
                
                # Calculate simple volatility (placeholder)
                returns = pv_df['portfolio_value'].pct_change().dropna()
                if len(returns) > 0:
                    volatility = returns.std() * np.sqrt(252)  # Annualized
                    metrics['volatility'] = volatility
                    if volatility > 0:
                        metrics['sharpe_ratio'] = total_return / volatility
                    else:
                        metrics['sharpe_ratio'] = 0.0
                else:
                    metrics['volatility'] = 0.0
                    metrics['sharpe_ratio'] = 0.0
                    
                # Max drawdown (placeholder)
                roll_max = pv_df['portfolio_value'].cummax()
                drawdown = (pv_df['portfolio_value'] - roll_max) / roll_max
                metrics['max_drawdown'] = drawdown.min() if len(drawdown) > 0 else 0.0
            else:
                metrics.update({
                    'total_return': 0.0,
                    'cagr': 0.0,
                    'volatility': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0
                })
        else:
            metrics.update({
                'total_return': 0.0,
                'cagr': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0
            })
        
        # Trade metrics if available
        if 'trades' in results and not results['trades'].empty:
            trades_df = results['trades']
            sell_trades = trades_df[trades_df['action'] == 'SELL'].copy()
            if not sell_trades.empty and 'pnl' in sell_trades.columns:
                winning_trades = sell_trades[sell_trades['pnl'] > 0]
                losing_trades = sell_trades[sell_trades['pnl'] <= 0]
                
                metrics['win_rate'] = len(winning_trades) / len(sell_trades) if len(sell_trades) > 0 else 0.0
                metrics['total_trades'] = len(sell_trades)
                metrics['winning_trades'] = len(winning_trades)
                metrics['losing_trades'] = len(losing_trades)
                
                if len(losing_trades) > 0 and losing_trades['pnl'].sum() != 0:
                    avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
                    avg_loss = abs(losing_trades['pnl'].mean()) if len(losing_trades) > 0 else 1
                    metrics['profit_factor'] = avg_win / avg_loss if avg_loss != 0 else 0
                else:
                    metrics['profit_factor'] = 0.0
                    
                metrics['avg_win'] = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
                metrics['avg_loss'] = abs(losing_trades['pnl'].mean()) if len(losing_trades) > 0 else 0
            else:
                metrics.update({
                    'win_rate': 0.0,
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'profit_factor': 0.0,
                    'avg_win': 0.0,
                    'avg_loss': 0.0
                })
        else:
            metrics.update({
                'win_rate': 0.0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'profit_factor': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0
            })
        
        return metrics
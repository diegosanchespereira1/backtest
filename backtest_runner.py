#!/usr/bin/env python3
"""
Stock Backtesting Framework - Main Runner Script
"""

import argparse
import sys
import os
from datetime import datetime
import pandas as pd

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest_engine.data_handler import DataHandler
from backtest_engine.strategies.factory import StrategyFactory
from backtest_engine.risk_management.position_sizer import PositionSizer
from backtest_engine.analytics.performance import PerformanceAnalyzer
from backtest_engine.visualization.report_generator import ReportGenerator
from backtest_engine.optimizer.grid_search import ParameterOptimizer


class BacktestRunner:
    def __init__(self, initial_capital=100000, commission=0.001, slippage=0.0005):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.data_handler = DataHandler()
        self.results = {}
        
    def fetch_data(self, symbols, start_date, end_date, interval='1d'):
        """Fetch historical data for given symbols"""
        print(f"Fetching data for {len(symbols)} symbols from {start_date} to {end_date}")
        data = self.data_handler.fetch_data(symbols, start_date, end_date, interval)
        print(f"Data fetched: {len(data)} rows per symbol on average")
        return data
    
    def run_backtest(self, data, strategy, symbols=None):
        """Run backtest for a single strategy"""
        if symbols is None:
            symbols = list(data.keys()) if isinstance(data, dict) else [data.get('symbol', 'UNKNOWN')]
        
        print(f"Running backtest for strategy: {strategy.name}")
        
        # Initialize results storage
        portfolio_values = []
        trades = []
        positions = {symbol: 0 for symbol in symbols}
        cash = self.initial_capital
        
        # Get date range (assuming all symbols have same dates)
        if isinstance(data, dict):
            first_symbol = list(data.keys())[0]
            dates = data[first_symbol].index
        else:
            dates = data.index
            
        # Iterate through each time step
        for i, current_date in enumerate(dates):
            # Update portfolio value at start of period
            portfolio_value = cash
            for symbol in symbols:
                if symbol in data:
                    price = data[symbol].loc[current_date, 'Close'] if isinstance(data, dict) else data.loc[current_date, 'Close']
                    portfolio_value += positions[symbol] * price
            portfolio_values.append({
                'date': current_date,
                'portfolio_value': portfolio_value,
                'cash': cash
            })
            
            # Generate signals for each symbol
            for symbol in symbols:
                if symbol not in data or i >= len(data[symbol]):
                    continue
                    
                # Get historical data up to current point
                hist_data = data[symbol].iloc[:i+1] if isinstance(data, dict) else data.iloc[:i+1]
                
                if len(hist_data) < 20:  # Need minimum data for indicators
                    continue
                    
                # Calculate indicators and generate signal
                try:
                    signal_data = strategy.calculate_indicators(hist_data.copy())
                    signal_data = strategy.generate_signals(signal_data)
                    
                    current_signal = signal_data.iloc[-1]['signal'] if 'signal' in signal_data.columns else 0
                    current_price = signal_data.iloc[-1]['Close']
                    
                    # Execute trades based on signal
                    if current_signal == 1 and positions[symbol] <= 0:  # Buy signal
                        # Calculate position size
                        position_size = PositionSizer.calculate_position_size(
                            cash, current_price, self.commission, self.slippage
                        )
                        if position_size > 0:
                            cost = position_size * current_price * (1 + self.commission + self.slippage)
                            if cost <= cash:
                                positions[symbol] += position_size
                                cash -= cost
                                trades.append({
                                    'date': current_date,
                                    'symbol': symbol,
                        'action': 'BUY',
                        'quantity': position_size,
                        'price': current_price,
                        'cost': cost,
                        'signal': 'BUY'
                                })
                    
                    elif current_signal == -1 and positions[symbol] >= 0:  # Sell signal
                        if positions[symbol] > 0:  # Only sell if we have position
                            proceeds = positions[symbol] * current_price * (1 - self.commission - self.slippage)
                            cash += proceeds
                            trades.append({
                                'date': current_date,
                                'symbol': symbol,
                                'action': 'SELL',
                                'quantity': positions[symbol],
                                'price': current_price,
                                'proceeds': proceeds,
                                'signal': 'SELL'
                            })
                            positions[symbol] = 0
                            
                except Exception as e:
                    print(f"Error processing {symbol} at {current_date}: {e}")
                    continue
        
        # Calculate final portfolio value
        final_portfolio_value = cash
        for symbol in symbols:
            if symbol in data and len(data[symbol]) > 0:
                final_price = data[symbol].iloc[-1]['Close'] if isinstance(data, dict) else data.iloc[-1]['Close']
                final_portfolio_value += positions[symbol] * final_price
        
        # Store results
        self.results = {
            'portfolio_values': pd.DataFrame(portfolio_values),
            'trades': pd.DataFrame(trades) if trades else pd.DataFrame(),
            'final_portfolio_value': final_portfolio_value,
            'initial_capital': self.initial_capital,
            'symbols': symbols,
            'strategy': strategy,
            'data': data
        }
        
        return self.results
    
    def run_comparison(self, data, strategy_names, symbols=None):
        """Run backtest for multiple strategies and compare"""
        comparison_results = {}
        
        for strategy_name in strategy_names:
            print(f"\n{'='*50}")
            print(f"Testing strategy: {strategy_name}")
            print(f"{'='*50}")
            
            strategy = StrategyFactory.create(strategy_name)
            if strategy is None:
                print(f"Warning: Strategy '{strategy_name}' not found, skipping...")
                continue
                
            result = self.run_backtest(data, strategy, symbols)
            comparison_results[strategy_name] = result
            
        return comparison_results
    
    def generate_report(self, results, output_dir='./reports'):
        """Generate performance report"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Analyze performance
        analyzer = PerformanceAnalyzer()
        metrics = analyzer.calculate_metrics(results)
        
        # Generate visualizations
        reporter = ReportGenerator()
        reporter.generate_report(results, metrics, output_dir)
        
        # Save metrics
        metrics_path = os.path.join(output_dir, 'metrics.json')
        import json
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
            
        print(f"Report generated in {output_dir}")
        print(f"Metrics saved to {metrics_path}")
        
        return metrics
    
    def optimize_parameters(self, data, strategy_name, param_ranges, symbols=None):
        """Optimize strategy parameters using grid search"""
        optimizer = ParameterOptimizer(self.data_handler, StrategyFactory)
        best_params, best_result = optimizer.optimize(
            data, strategy_name, param_ranges, 
            initial_capital=self.initial_capital,
            commission=self.commission,
            slippage=self.slippage,
            symbols=symbols
        )
        return best_params, best_result


def main():
    parser = argparse.ArgumentParser(description='Stock Backtesting Framework')
    parser.add_argument('--strategy', type=str, help='Strategy to test (rsi_bbands, macd_rsi_vol, ema_adx_rsi, bbands_rsi_cci)')
    parser.add_argument('--compare-all', action='store_true', help='Compare all available strategies')
    parser.add_argument('--optimize', action='store_true', help='Run parameter optimization')
    parser.add_argument('--symbols', type=str, nargs='+', default=['SPY'], help='Symbols to test (e.g., AAPL MSFT GOOGL)')
    parser.add_argument('--start', type=str, default='2020-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2023-12-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--interval', type=str, default='1d', help='Data interval (1m,5m,15m,1h,1d,1wk,1mo)')
    parser.add_argument('--capital', type=float, default=100000, help='Initial capital')
    parser.add_argument('--commission', type=float, default=0.001, help='Commission rate (0.001 = 0.1%)')
    parser.add_argument('--slippage', type=float, default=0.0005, help='Slippage rate (0.0005 = 0.05%)')
    parser.add_argument('--output', type=str, default='./reports', help='Output directory for reports')
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = BacktestRunner(
        initial_capital=args.capital,
        commission=args.commission,
        slippage=args.slippage
    )
    
    # Fetch data
    print("Fetching market data...")
    data = runner.fetch_data(args.symbols, args.start, args.end, args.interval)
    
    if args.compare_all:
        # Test all available strategies
        strategy_names = ['rsi_bbands', 'macd_rsi_vol', 'ema_adx_rsi', 'bbands_rsi_cci']
        print(f"\nComparing {len(strategy_names)} strategies on {args.symbols}")
        results = runner.run_comparison(data, strategy_names, args.symbols)
        
        # Generate comparison report
        comparison_dir = os.path.join(args.output, 'strategy_comparison')
        os.makedirs(comparison_dir, exist_ok=True)
        
        from backtest_engine.visualization.report_generator import ReportGenerator
        reporter = ReportGenerator()
        reporter.generate_comparison_report(results, comparison_dir)
        
    elif args.optimize and args.strategy:
        # Run parameter optimization
        print(f"Optimizing parameters for strategy: {args.strategy}")
        
        # Define parameter ranges for optimization
        param_ranges = {
            'rsi_bbands': {
                'rsi_length': [10, 14, 20],
                'bb_length': [15, 20, 25],
                'bb_std': [1.5, 2.0, 2.5]
            },
            'macd_rsi_vol': {
                'macd_fast': [8, 12, 16],
                'macd_slow': [17, 26, 35],
                'macd_signal': [6, 9, 12],
                'rsi_length': [10, 14, 20],
                'volume_multiplier': [1.2, 1.5, 2.0]
            },
            'ema_adx_rsi': {
                'ema_fast': [10, 20, 50],
                'ema_slow': [50, 100, 200],
                'adx_length': [10, 14, 20],
                'adx_threshold': [20, 25, 30],
                'rsi_length': [10, 14, 20]
            },
            'bbands_rsi_cci': {
                'bb_length': [15, 20, 25],
                'bb_std': [1.5, 2.0, 2.5],
                'rsi_length': [10, 14, 20],
                'cci_length': [10, 14, 20],
                'squeeze_threshold': [0.01, 0.02, 0.03]
            }
        }
        
        if args.strategy in param_ranges:
            best_params, best_result = runner.optimize_parameters(
                data, args.strategy, param_ranges[args.strategy], args.symbols
            )
            
            print(f"\nOptimization Complete!")
            print(f"Best parameters: {best_params}")
            
            # Save optimization results
            opt_dir = os.path.join(args.output, f'{args.strategy}_optimization')
            os.makedirs(opt_dir, exist_ok=True)
            
            import json
            with open(os.path.join(opt_dir, 'best_params.json'), 'w') as f:
                json.dump(best_params, f, indent=2)
                
            # Run backtest with best parameters and generate report
            strategy = StrategyFactory.create(args.strategy, **best_params)
            results = runner.run_backtest(data, strategy, args.symbols)
            runner.generate_report(results, opt_dir)
        else:
            print(f"No parameter ranges defined for strategy: {args.strategy}")
            
    elif args.strategy:
        # Run single strategy backtest
        print(f"Running backtest for strategy: {args.strategy}")
        strategy = StrategyFactory.create(args.strategy)
        
        if strategy is None:
            print(f"Error: Strategy '{args.strategy}' not found")
            print("Available strategies: rsi_bbands, macd_rsi_vol, ema_adx_rsi, bbands_rsi_cci")
            return 1
            
        results = runner.run_backtest(data, strategy, args.symbols)
        metrics = runner.generate_report(results, args.output)
        
        # Print summary
        print(f"\n{'='*50}")
        print(f"BACKTEST RESULTS - {strategy.name}")
        print(f"{'='*50}")
        print(f"Initial Capital: ${args.capital:,.2f}")
        print(f"Final Portfolio Value: ${results['final_portfolio_value']:,.2f}")
        print(f"Total Return: {(results['final_portfolio_value'] / args.capital - 1):.2%}")
        if 'cagr' in metrics:
            print(f"CAGR: {metrics['cagr']:.2%}")
        if 'sharpe_ratio' in metrics:
            print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        if 'max_drawdown' in metrics:
            print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        if 'win_rate' in metrics:
            print(f"Win Rate: {metrics['win_rate']:.2%}")
        print(f"Total Trades: {len(results['trades']) if not results['trades'].empty else 0}")
        print(f"{'='*50}")
        
    else:
        print("Please specify a strategy to test or use --compare-all")
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
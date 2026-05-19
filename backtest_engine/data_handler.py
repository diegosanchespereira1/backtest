#!/usr/bin/env python3
"""
Data Handler for Stock Backtesting Framework
Handles data fetching, preprocessing, and management
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Union, Optional
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataHandler:
    def __init__(self, cache_dir: str = './data_cache'):
        """
        Initialize DataHandler
        
        Args:
            cache_dir: Directory to cache downloaded data
        """
        self.cache_dir = cache_dir
        import os
        os.makedirs(cache_dir, exist_ok=True)
        
    def fetch_data(self, symbols: Union[str, List[str]], 
                   start_date: str, 
                   end_date: str, 
                   interval: str = '1d',
                   use_cache: bool = True) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Fetch historical data for given symbols
        
        Args:
            symbols: Single symbol string or list of symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame for single symbol or dict of DataFrames for multiple symbols
        """
        if isinstance(symbols, str):
            symbols = [symbols]
            single_symbol = True
        else:
            single_symbol = False
            
        # Validate dates
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError as e:
            raise ValueError(f"Invalid date format. Use YYYY-MM-DD: {e}")
            
        if start >= end:
            raise ValueError("Start date must be before end date")
            
        # Adjust interval for intraday data limitations (yfinance caps)
        _intraday_limits = {'1m': 7, '2m': 60, '5m': 60, '15m': 60, '30m': 60, '90m': 60, '60m': 730, '1h': 730}
        if interval in _intraday_limits:
            max_intraday_days = _intraday_limits[interval]
            if (end - start).days > max_intraday_days:
                logger.warning(f"Intraday interval {interval} limited to {max_intraday_days} days. "
                             f"Adjusting start date from {start_date} to {(end - timedelta(days=max_intraday_days)).strftime('%Y-%m-%d')}")
                start = end - timedelta(days=max_intraday_days)
                start_date = start.strftime('%Y-%m-%d')
        
        data = {}
        
        for i, symbol in enumerate(symbols):
            # Clean symbol
            symbol = symbol.upper().strip()
            
            # Check cache
            cache_file = f"{self.cache_dir}/{symbol}_{interval}_{start_date}_{end_date}.pkl"
            if use_cache and self._cache_exists(cache_file):
                try:
                    logger.info(f"Loading {symbol} from cache...")
                    data[symbol] = pd.read_pickle(cache_file)
                    # Verify data has required columns
                    if not self._validate_ohlcv_data(data[symbol]):
                        logger.warning(f"Cached data for {symbol} is invalid, re-downloading...")
                        data[symbol] = self._download_symbol_data(symbol, start_date, end_date, interval)
                        self._save_to_cache(data[symbol], cache_file)
                except Exception as e:
                    logger.warning(f"Error loading cache for {symbol}: {e}. Downloading fresh data...")
                    data[symbol] = self._download_symbol_data(symbol, start_date, end_date, interval)
                    self._save_to_cache(data[symbol], cache_file)
            else:
                # Download fresh data
                logger.info(f"Downloading {symbol} ({i+1}/{len(symbols)})...")
                data[symbol] = self._download_symbol_data(symbol, start_date, end_date, interval)
                if use_cache:
                    self._save_to_cache(data[symbol], cache_file)
                    
            # Rate limiting to be nice to Yahoo Finance
            if i < len(symbols) - 1:  # Don't sleep after last symbol
                time.sleep(0.1)
        
        # Validate all data
        _removed = []
        for symbol in list(data.keys()):
            if not self._validate_ohlcv_data(data[symbol]):
                logger.warning(f"Removing {symbol} due to invalid data")
                _removed.append(symbol)
                del data[symbol]

        if not data:
            _msg = "No valid data retrieved for any symbols."
            if _removed:
                _msg += f" Failed symbols: {_removed}. Check tickers are valid US equity symbols and the date range is within yfinance limits."
            raise ValueError(_msg)
            
        return data[symbols[0]] if single_symbol and len(data) == 1 else data
    
    def _download_symbol_data(self, symbol: str, start_date: str, end_date: str, interval: str) -> pd.DataFrame:
        """Download data for a single symbol"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if df.empty:
                raise ValueError(f"No data returned for symbol {symbol}")
                
            # Ensure we have required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing columns for {symbol}: {missing_cols}")
                
            # Clean column names
            df.columns = [col.replace(' ', '_') for col in df.columns]
            
            # Ensure datetime index — strip timezone (yfinance 1.2+ returns tz-aware)
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index

            # Keep only OHLCV columns
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

            # Sort by date
            df = df.sort_index()

            # Remove any rows with all NaN in OHLC
            df = df.dropna(subset=['Open', 'High', 'Low', 'Close'], how='all')

            # Forward fill volume NaNs (sometimes missing for early data)
            df['Volume'] = df['Volume'].fillna(0)

            # Ensure all OHLCV are float64 (avoids TA-Lib type errors)
            for _col in ['Open', 'High', 'Low', 'Close']:
                df[_col] = df[_col].astype('float64')
            df['Volume'] = df['Volume'].astype('float64')
            
            logger.info(f"Downloaded {len(df)} rows for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to download data for {symbol}: {e}")
            raise
    
    def _validate_ohlcv_data(self, df: pd.DataFrame) -> bool:
        """Validate that DataFrame has proper OHLCV data."""
        if df is None or df.empty:
            logger.warning("Validation failed: empty DataFrame")
            return False

        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            logger.warning(f"Validation failed: missing columns {missing}")
            return False

        if len(df) < 10:
            logger.warning(f"Validation failed: only {len(df)} rows (need ≥10)")
            return False

        # Allow tiny floating-point tolerance (1e-6) for yfinance rounding
        tol = 1e-6
        if (df['High'] < df['Low'] - tol).any():
            logger.warning("Validation failed: High < Low")
            return False
        if (df['Close'] > df['High'] + tol).any() or (df['Close'] < df['Low'] - tol).any():
            logger.warning("Validation failed: Close outside High/Low range")
            return False
        if (df['Open'] > df['High'] + tol).any() or (df['Open'] < df['Low'] - tol).any():
            logger.warning("Validation failed: Open outside High/Low range")
            return False
        if (df['Volume'] < 0).any():
            logger.warning("Validation failed: negative Volume")
            return False

        return True
    
    def _cache_exists(self, cache_file: str) -> bool:
        """Check if cache file exists and is recent"""
        import os
        if not os.path.exists(cache_file):
            return False
            
        # Check if cache is less than 1 day old
        file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        return (datetime.now() - file_time).total_seconds() < 86400  # 24 hours
    
    def _save_to_cache(self, df: pd.DataFrame, cache_file: str):
        """Save DataFrame to cache"""
        import os
        try:
            df.to_pickle(cache_file)
            logger.debug(f"Saved data to cache: {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def get_symbol_info(self, symbol: str) -> dict:
        """Get basic information about a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return {
                'symbol': symbol,
                'name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'currency': info.get('currency', 'USD')
            }
        except Exception as e:
            logger.warning(f"Could not get info for {symbol}: {e}")
            return {'symbol': symbol, 'name': 'Unknown', 'sector': 'N/A', 
                     'industry': 'N/A', 'market_cap': 0, 'currency': 'USD'}
    
    def get_multiple_symbols_info(self, symbols: List[str]) -> List[dict]:
        """Get info for multiple symbols"""
        return [self.get_symbol_info(symbol) for symbol in symbols]


# Convenience functions for easy usage
def fetch_stock_data(symbols, start_date, end_date, interval='1d'):
    """
    Convenience function to fetch stock data
    
    Args:
        symbols: Single symbol or list of symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        interval: Data interval
        
    Returns:
        DataFrame or dict of DataFrames
    """
    handler = DataHandler()
    return handler.fetch_data(symbols, start_date, end_date, interval)


if __name__ == "__main__":
    # Test the data handler
    print("Testing DataHandler...")
    handler = DataHandler()
    
    # Test single symbol
    try:
        data = handler.fetch_data('AAPL', '2023-01-01', '2023-12-31', '1d')
        print(f"AAPL data shape: {data.shape}")
        print(f"AAPL columns: {list(data.columns)}")
        print(f"Date range: {data.index[0]} to {data.index[-1]}")
        print()
    except Exception as e:
        print(f"Error fetching AAPL: {e}")
    
    # Test multiple symbols
    try:
        data = handler.fetch_data(['AAPL', 'MSFT', 'GOOGL'], '2023-06-01', '2023-12-31', '1d')
        print(f"Multiple symbols data:")
        for symbol, df in data.items():
            print(f"  {symbol}: {df.shape}")
        print()
    except Exception as e:
        print(f"Error fetching multiple symbols: {e}")
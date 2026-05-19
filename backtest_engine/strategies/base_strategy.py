#!/usr/bin/env python3
"""
Base Strategy Class and Strategy Factory using TA-Lib
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import talib
from typing import Dict, List, Optional, Any


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies
    """
    
    def __init__(self):
        self.name = "Base Strategy"
        self.description = "Abstract base strategy"
        self.parameters = {}
        
    @abstractmethod
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for the strategy
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added indicator columns
        """
        pass
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy/sell signals based on indicators
        
        Args:
            data: DataFrame with indicators calculated
            
        Returns:
            DataFrame with 'signal' column (1 = buy, -1 = sell, 0 = hold)
        """
        pass
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters"""
        return self.parameters.copy()
    
    def set_parameters(self, **kwargs):
        """Set strategy parameters"""
        for key, value in kwargs.items():
            if key in self.parameters:
                self.parameters[key] = value
            else:
                # Allow new parameters to be added
                self.parameters[key] = value
    
    def __str__(self):
        return f"{self.name} (params: {self.parameters})"


class StrategyFactory:
    """Factory for creating strategy instances"""
    
    @staticmethod
    def create(strategy_name: str, **kwargs) -> Optional[BaseStrategy]:
        """
        Create a strategy instance by name
        
        Args:
            strategy_name: Name of strategy to create
            **kwargs: Parameters to pass to strategy constructor
            
        Returns:
            Strategy instance or None if not found
        """
        strategy_map = {
            'rsi_bbands': RSIBollingerBandsStrategy,
            'macd_rsi_vol': MACDRSIVolumeStrategy,
            'ema_adx_rsi': EMADXRSIStrategy,
            'bbands_rsi_cci': BBandsRSICCIStrategy
        }
        
        if strategy_name not in strategy_map:
            print(f"Unknown strategy: {strategy_name}")
            print(f"Available strategies: {list(strategy_map.keys())}")
            return None
            
        try:
            strategy_class = strategy_map[strategy_name]
            return strategy_class(**kwargs)
        except Exception as e:
            print(f"Error creating strategy {strategy_name}: {e}")
            return None
    
    @staticmethod
    def get_available_strategies() -> List[str]:
        """Get list of available strategy names"""
        return ['rsi_bbands', 'macd_rsi_vol', 'ema_adx_rsi', 'bbands_rsi_cci']
    
    @staticmethod
    def get_strategy_info(strategy_name: str) -> dict:
        """Get information about a strategy"""
        strategy_map = {
            'rsi_bbands': "RSI + Bollinger Bands Strategy",
            'macd_rsi_vol': "MACD + RSI + Volume Filter Strategy", 
            'ema_adx_rsi': "EMA(50/200) + ADX + RSI Pullback Strategy",
            'bbands_rsi_cci': "Bollinger Bands Width + RSI + CCI Strategy"
        }
        
        if strategy_name not in strategy_map:
            return {"error": f"Unknown strategy: {strategy_name}"}
            
        return {
            "name": strategy_name,
            "description": strategy_map[strategy_name],
            "available": True
        }


# ============================================================================
# STRATEGY IMPLEMENTATIONS
# ============================================================================

class RSIBollingerBandsStrategy(BaseStrategy):
    """RSI + Bollinger Bands Strategy (from stock-monitor-agent skill)"""
    
    def __init__(self, rsi_length: int = 14, bb_length: int = 20, bb_std: float = 2.0,
                 rsi_oversold: int = 30, rsi_overbought: int = 70):
        super().__init__()
        self.name = "RSI + Bollinger Bands"
        self.description = "Buy when RSI < 30 and price < lower Bollinger Band, Sell when RSI > 70 and price > upper Bollinger Band"
        self.parameters = {
            'rsi_length': rsi_length,
            'bb_length': bb_length,
            'bb_std': bb_std,
            'rsi_oversold': rsi_oversold,
            'rsi_overbought': rsi_overbought
        }
        
        # Set instance variables
        self.rsi_length = rsi_length
        self.bb_length = bb_length
        self.bb_std = bb_std
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI and Bollinger Bands using TA-Lib"""
        # Calculate RSI
        data['RSI'] = talib.RSI(data['Close'].values, timeperiod=self.rsi_length)
        
        # Calculate Bollinger Bands
        upper, middle, lower = talib.BBANDS(data['Close'].values, 
                                          timeperiod=self.bb_length, 
                                          nbdevup=self.bb_std, 
                                          nbdevdn=self.bb_std)
        data['BBU'] = upper
        data['BBM'] = middle
        data['BBL'] = lower
        
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate buy/sell signals"""
        # Initialize signal column
        data['signal'] = 0
        
        # Buy signal: RSI < oversold AND Close < Lower Bollinger Band
        buy_condition = (
            (data['RSI'] < self.rsi_oversold) & 
            (data['Close'] < data['BBL'])
        )
        
        # Sell signal: RSI > overbought AND Close > Upper Bollinger Band  
        sell_condition = (
            (data['RSI'] > self.rsi_overbought) & 
            (data['Close'] > data['BBU'])
        )
        
        data.loc[buy_condition, 'signal'] = 1   # Buy
        data.loc[sell_condition, 'signal'] = -1  # Sell
        
        return data


class MACDRSIVolumeStrategy(BaseStrategy):
    """MACD + RSI + Volume Filter Strategy (Model 1 from research)"""
    
    def __init__(self, macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9,
                 rsi_length: int = 14, rsi_low: int = 40, rsi_high: int = 60,
                 volume_multiplier: float = 1.5, volume_length: int = 20):
        super().__init__()
        self.name = "MACD + RSI + Volume Filter"
        self.description = "Buy when MACD bullish crossover, RSI in neutral zone (40-60), and volume > avg*multiplier"
        self.parameters = {
            'macd_fast': macd_fast,
            'macd_slow': macd_slow,
            'macd_signal': macd_signal,
            'rsi_length': rsi_length,
            'rsi_low': rsi_low,
            'rsi_high': rsi_high,
            'volume_multiplier': volume_multiplier,
            'volume_length': volume_length
        }
        
        # Set instance variables
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.rsi_length = rsi_length
        self.rsi_low = rsi_low
        self.rsi_high = rsi_high
        self.volume_multiplier = volume_multiplier
        self.volume_length = volume_length
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD, RSI, and volume indicators using TA-Lib"""
        # Calculate MACD
        macd, macdsignal, macdhist = talib.MACD(data['Close'].values, 
                                              fastperiod=self.macd_fast, 
                                              slowperiod=self.macd_slow, 
                                              signalperiod=self.macd_signal)
        data['MACD'] = macd
        data['MACDS'] = macdsignal
        data['MACDHIST'] = macdhist
        
        # Calculate RSI
        data['RSI'] = talib.RSI(data['Close'].values, timeperiod=self.rsi_length)
        
        # Calculate volume average
        data["Volume_MA"] = talib.SMA(data["Volume"].values.astype(float), timeperiod=self.volume_length)
        
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate buy/sell signals"""
        # Initialize signal column
        data['signal'] = 0
        
        # MACD crossover signals
        macd_buy = (data["MACD"] > data["MACDS"]) & (data["MACD"].shift(1) <= data["MACDS"].shift(1))
        macd_sell = (data["MACD"] < data["MACDS"]) & (data["MACD"].shift(1) >= data["MACDS"].shift(1))
        
        # RSI conditions
        rsi_neutral = (data['RSI'] > self.rsi_low) & (data['RSI'] < self.rsi_high)
        
        # Volume condition: volume > volume_multiplier * volume average
        volume_condition = data['Volume'] > (data['Volume_MA'] * self.volume_multiplier)
        
        # Combined strategy conditions
        long_condition = macd_buy & rsi_neutral & volume_condition
        short_condition = macd_sell & rsi_neutral & volume_condition  # For completeness
        
        data.loc[long_condition, 'signal'] = 1   # Buy
        data.loc[short_condition, 'signal'] = -1  # Sell
        
        return data


class EMADXRSIStrategy(BaseStrategy):
    """EMA(50/200) Trend + ADX + RSI Pullback Strategy (Model 2 from research)"""
    
    def __init__(self, ema_fast: int = 50, ema_slow: int = 200, adx_length: int = 14,
                 adx_threshold: int = 25, rsi_length: int = 14, rsi_buy: int = 40,
                 rsi_sell: int = 60):
        super().__init__()
        self.name = "EMA Trend + ADX + RSI Pullback"
        self.description = "Buy in uptrend (EMA_fast > EMA_slow) with ADX > threshold on RSI pullback (<40). Sell in downtrend on RSI >60."
        self.parameters = {
            'ema_fast': ema_fast,
            'ema_slow': ema_slow,
            'adx_length': adx_length,
            'adx_threshold': adx_threshold,
            'rsi_length': rsi_length,
            'rsi_buy': rsi_buy,
            'rsi_sell': rsi_sell
        }
        
        # Set instance variables
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.adx_length = adx_length
        self.adx_threshold = adx_threshold
        self.rsi_length = rsi_length
        self.rsi_buy = rsi_buy
        self.rsi_sell = rsi_sell
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate EMA, ADX, and RSI indicators using TA-Lib"""
        # Calculate EMAs
        data['EMA_Fast'] = talib.EMA(data['Close'].values, timeperiod=self.ema_fast)
        data['EMA_Slow'] = talib.EMA(data['Close'].values, timeperiod=self.ema_slow)
        
        # Calculate ADX
        data['ADX'] = talib.ADX(data['High'].values, data['Low'].values, data['Close'].values, 
                              timeperiod=self.adx_length)
        
        # Calculate RSI
        data['RSI'] = talib.RSI(data['Close'].values, timeperiod=self.rsi_length)
        
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate buy/sell signals"""
        # Initialize signal column
        data['signal'] = 0
        
        # Trend determination
        is_uptrend = data['EMA_Fast'] > data['EMA_Slow']
        is_downtrend = data['EMA_Fast'] < data['EMA_Slow']
        strong_trend = data['ADX'] > self.adx_threshold
        
        # RSI conditions for pullbacks
        rsi_pullback_buy = data['RSI'] < self.rsi_buy
        rsi_pullback_sell = data['RSI'] > self.rsi_sell
        
        # Combined strategy conditions
        long_condition = is_uptrend & strong_trend & rsi_pullback_buy
        short_condition = is_downtrend & strong_trend & rsi_pullback_sell
        
        data.loc[long_condition, 'signal'] = 1   # Buy
        data.loc[short_condition, 'signal'] = -1  # Sell
        
        return data


class BBandsRSICCIStrategy(BaseStrategy):
    """Bollinger Bands Width + RSI + CCI Strategy (Model 3 from research)"""
    
    def __init__(self, bb_length: int = 20, bb_std: float = 2.0, rsi_length: int = 14,
                 cci_length: int = 20, squeeze_threshold: float = 0.80):
        super().__init__()
        self.name = "BBands Width + RSI + CCI"
        self.description = "Buy when breaking out of Bollinger Bands squeeze in direction of CCI/RSI momentum"
        self.parameters = {
            'bb_length': bb_length,
            'bb_std': bb_std,
            'rsi_length': rsi_length,
            'cci_length': cci_length,
            'squeeze_threshold': squeeze_threshold
        }
        
        # Set instance variables
        self.bb_length = bb_length
        self.bb_std = bb_std
        self.rsi_length = rsi_length
        self.cci_length = cci_length
        self.squeeze_threshold = squeeze_threshold  # ratio vs BB_Width rolling mean (0.8 = 80%)
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands, RSI, and CCI indicators using TA-Lib"""
        # Calculate Bollinger Bands
        upper, middle, lower = talib.BBANDS(data['Close'].values, 
                                          timeperiod=self.bb_length, 
                                          nbdevup=self.bb_std, 
                                          nbdevdn=self.bb_std)
        data['BBU'] = upper
        data['BBM'] = middle
        data['BBL'] = lower
        
        # Calculate Bollinger Bands width + rolling mean for relative squeeze detection
        data['BB_Width'] = (data['BBU'] - data['BBL']) / data['BBM']
        data['BB_Width_SMA'] = data['BB_Width'].rolling(window=self.bb_length).mean()
        
        # Calculate RSI
        data['RSI'] = talib.RSI(data['Close'].values, timeperiod=self.rsi_length)
        
        # Calculate CCI
        data['CCI'] = talib.CCI(data['High'].values, data['Low'].values, data['Close'].values, 
                              timeperiod=self.cci_length)
        
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate buy/sell signals"""
        # Initialize signal column
        data['signal'] = 0
        
        # Relative squeeze: BB_Width below squeeze_threshold * its own rolling mean
        # Adapts to the stock's natural volatility — no fixed absolute threshold needed
        is_squeeze = data['BB_Width'] < (data['BB_Width_SMA'] * self.squeeze_threshold)

        # Squeeze in any of the last 5 bars (breakout often happens a few bars after squeeze)
        was_in_squeeze = is_squeeze.rolling(5).max().shift(1).fillna(False).astype(bool)

        # Trend direction from CCI
        cci_uptrend = data['CCI'] > 100
        cci_downtrend = data['CCI'] < -100

        # Momentum from RSI (avoid extreme overbought/oversold for breakouts)
        rsi_not_overbought = data['RSI'] < 70
        rsi_not_oversold = data['RSI'] > 30

        # Breakout conditions
        bullish_breakout = (data['Close'] > data['BBU']) & cci_uptrend & rsi_not_overbought
        bearish_breakout = (data['Close'] < data['BBL']) & cci_downtrend & rsi_not_oversold

        # Combined: breakout after a recent squeeze
        long_condition = bullish_breakout & was_in_squeeze
        short_condition = bearish_breakout & was_in_squeeze
        
        data.loc[long_condition, 'signal'] = 1   # Buy
        data.loc[short_condition, 'signal'] = -1  # Sell
        
        return data
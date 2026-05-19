#!/usr/bin/env python3
"""
Risk Management Module
Handles position sizing, stop-loss, and risk controls
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import talib as ta

logger = logging.getLogger(__name__)


class PositionSizer:
    """Handles position sizing calculations"""
    
    @staticmethod
    def fixed_fractional(capital: float, risk_per_trade: float, 
                        entry_price: float, stop_loss_price: float) -> float:
        """
        Calculate position size based on fixed fractional risk
        
        Args:
            capital: Available capital
            risk_per_trade: Risk per trade as decimal (e.g., 0.02 for 2%)
            entry_price: Entry price per share
            stop_loss_price: Stop loss price per share
            
        Returns:
            Number of shares to trade
        """
        if entry_price <= 0 or stop_loss_price <= 0:
            return 0
            
        risk_per_share = abs(entry_price - stop_loss_price)
        if risk_per_share <= 0:
            return 0
            
        risk_amount = capital * risk_per_trade
        position_size = risk_amount / risk_per_share
        
        # Ensure we don't exceed available capital
        max_shares_by_capital = capital / entry_price
        position_size = min(position_size, max_shares_by_capital)
        
        return max(0, position_size)
    
    @staticmethod
    def fixed_dollar(capital: float, dollar_amount: float, 
                    entry_price: float) -> float:
        """
        Calculate position size based on fixed dollar amount
        
        Args:
            capital: Available capital
            dollar_amount: Dollar amount to invest per trade
            entry_price: Entry price per share
            
        Returns:
            Number of shares to trade
        """
        if entry_price <= 0:
            return 0
            
        position_size = dollar_amount / entry_price
        max_shares_by_capital = capital / entry_price
        position_size = min(position_size, max_shares_by_capital)
        
        return max(0, position_size)
    
    @staticmethod
    def percent_of_equity(capital: float, percent: float, 
                         entry_price: float) -> float:
        """
        Calculate position size as percentage of equity
        
        Args:
            capital: Available capital
            percent: Percentage of equity to use (e.g., 0.1 for 10%)
            entry_price: Entry price per share
            
        Returns:
            Number of shares to trade
        """
        if entry_price <= 0:
            return 0
            
        dollar_amount = capital * percent
        return PositionSizer.fixed_dollar(capital, dollar_amount, entry_price)
    
    @staticmethod
    def kelly_criterion(win_rate: float, avg_win: float, 
                       avg_loss: float, capital: float, 
                       entry_price: float) -> float:
        """
        Calculate position size using Kelly Criterion
        
        Args:
            win_rate: Probability of winning (0-1)
            avg_win: Average winning trade amount
            avg_loss: Average losing trade amount (positive number)
            capital: Available capital
            entry_price: Entry price per share
            
        Returns:
            Number of shares to trade
        """
        if avg_loss <= 0:
            return 0
            
        # Kelly formula: f* = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = loss_rate
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - win_rate
        
        kelly_fraction = (b * p - q) / b if b > 0 else 0
        
        # Limit Kelly fraction to prevent overbetting
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
        
        dollar_amount = capital * kelly_fraction
        return PositionSizer.fixed_dollar(capital, dollar_amount, entry_price)
    
    @staticmethod
    def calculate_position_size(capital: float, entry_price: float, 
                               commission: float = 0.001, slippage: float = 0.0005,
                               risk_per_trade: float = 0.02,
                               use_atr: bool = False, atr_multiplier: float = 2.0,
                               data: Optional[pd.DataFrame] = None) -> float:
        """
        Main position sizing function with multiple options
        
        Args:
            capital: Available capital
            entry_price: Entry price per share
            commission: Commission rate (decimal)
            slippage: Slippage rate (decimal)
            risk_per_trade: Risk per trade as decimal (default 2%)
            use_atr: Whether to use ATR for stop loss calculation
            atr_multiplier: ATR multiplier for stop loss distance
            data: DataFrame with OHLCV data (required if use_atr=True)
            
        Returns:
            Number of shares to trade
        """
        if use_atr and data is not None and len(data) > 0:
            # Calculate ATR-based stop loss
            if 'ATRr_14' not in data.columns:
                # Calculate ATR if not present
                atr = ta.atr(data['High'], data['Low'], data['Close'], length=14)
                data['ATRr_14'] = atr
            
            atr_value = data['ATRr_14'].iloc[-1] if len(data) > 0 else entry_price * 0.02
            stop_loss_price = entry_price - (atr_value * atr_multiplier)
            
            # Ensure stop loss is reasonable
            max_stop_loss = entry_price * 0.1  # Max 10% stop loss
            if abs(entry_price - stop_loss_price) > max_stop_loss:
                stop_loss_price = entry_price - np.sign(entry_price - stop_loss_price) * max_stop_loss
        else:
            # Fixed percentage stop loss
            stop_loss_price = entry_price * (1 - risk_per_trade)
        
        return PositionSizer.fixed_fractional(capital, risk_per_trade, entry_price, stop_loss_price)


class StopLossManager:
    """Handles stop-loss mechanisms"""
    
    @staticmethod
    def fixed_percentage_stop(entry_price: float, percentage: float, 
                             is_long: bool = True) -> float:
        """
        Calculate fixed percentage stop loss
        
        Args:
            entry_price: Entry price
            percentage: Stop loss percentage (e.g., 0.05 for 5%)
            is_long: True for long position, False for short
            
        Returns:
            Stop loss price
        """
        if is_long:
            return entry_price * (1 - percentage)
        else:
            return entry_price * (1 + percentage)
    
    @staticmethod
    def atr_stop(data: pd.DataFrame, atr_length: int = 14, 
                atr_multiplier: float = 2.0, is_long: bool = True) -> float:
        """
        Calculate ATR-based stop loss
        
        Args:
            data: DataFrame with OHLCV data
            atr_length: ATR lookback period
            atr_multiplier: ATR multiplier for stop distance
            is_long: True for long position, False for short
            
        Returns:
            Stop loss price
        """
        if len(data) == 0:
            return 0
            
        # Calculate ATR
        atr = ta.atr(data['High'], data['Low'], data['Close'], length=atr_length)
        atr_value = atr.iloc[-1] if len(atr) > 0 else 0
        
        if is_long:
            stop_price = data['Close'].iloc[-1] - (atr_value * atr_multiplier)
        else:
            stop_price = data['Close'].iloc[-1] + (atr_value * atr_multiplier)
            
        return max(0, stop_price)
    
    @staticmethod
    def trailing_stop(highest_price: float, trailing_percentage: float,
                     is_long: bool = True) -> float:
        """
        Calculate trailing stop loss
        
        Args:
            highest_price: Highest price reached since entry
            trailing_percentage: Trailing percentage (e.g., 0.05 for 5%)
            is_long: True for long position, False for short
            
        Returns:
            Trailing stop price
        """
        if is_long:
            return highest_price * (1 - trailing_percentage)
        else:
            return highest_price * (1 + trailing_percentage)
    
    @staticmethod
    def paraboli_sar(data: pd.DataFrame, acceleration: float = 0.02,
                    maximum: float = 0.2) -> List[float]:
        """
        Calculate Parabolic SAR for trailing stop
        
        Args:
            data: DataFrame with OHLCV data
            acceleration: Acceleration factor
            maximum: Maximum acceleration factor
            
        Returns:
            List of SAR values
        """
        if len(data) < 2:
            return [data['Close'].iloc[-1]] if len(data) > 0 else [0]
            
        # Simplified SAR calculation
        high = data['High'].values
        low = data['Low'].values
        close = data['Close'].values
        
        sar = close.copy()
        trend = 1  # 1 for uptrend, -1 for downtrend
        af = acceleration
        ep = high[0] if trend == 1 else low[0]  # Extreme point
        
        for i in range(1, len(close)):
            if trend == 1:  # Uptrend
                sar[i] = sar[i-1] + af * (ep - sar[i-1])
                # Reverse if price drops below SAR
                if low[i] < sar[i]:
                    trend = -1
                    sar[i] = ep
                    af = acceleration
                    ep = low[i]
                else:
                    # Update extreme point
                    if high[i] > ep:
                        ep = high[i]
                        af = min(af + acceleration, maximum)
            else:  # Downtrend
                sar[i] = sar[i-1] + af * (sar[i-1] - ep)
                # Reverse if price rises above SAR
                if high[i] > sar[i]:
                    trend = 1
                    sar[i] = ep
                    af = acceleration
                    ep = high[i]
                else:
                    # Update extreme point
                    if low[i] < ep:
                        ep = low[i]
                        af = min(af + acceleration, maximum)
        
        return sar.tolist()


class RiskManager:
    """Main risk management coordinator"""
    
    def __init__(self, max_portfolio_risk: float = 0.06, 
                 max_position_risk: float = 0.02,
                 max_correlation: float = 0.7,
                 max_sectors: int = 3):
        """
        Initialize RiskManager
        
        Args:
            max_portfolio_risk: Maximum portfolio risk at any time (default 6%)
            max_position_risk: Maximum risk per position (default 2%)
            max_correlation: Maximum correlation between positions (default 70%)
            max_sectors: Maximum number of sectors to invest in (default 3)
        """
        self.max_portfolio_risk = max_portfolio_risk
        self.max_position_risk = max_position_risk
        self.max_correlation = max_correlation
        self.max_sectors = max_sectors
        self.positions = {}  # Track current positions
        self.sector_exposure = {}  # Track sector exposure
        
    def can_open_position(self, symbol: str, sector: str, 
                         position_value: float, portfolio_value: float,
                         current_positions: Dict[str, dict]) -> Tuple[bool, str]:
        """
        Check if a new position can be opened based on risk rules
        
        Args:
            symbol: Stock symbol
            sector: Stock sector
            position_value: Value of proposed position
            portfolio_value: Total portfolio value
            current_positions: Dictionary of current positions
            
        Returns:
            Tuple of (can_open, reason)
        """
        # Check position size limit
        position_risk = position_value / portfolio_value if portfolio_value > 0 else 0
        if position_risk > self.max_position_risk:
            return False, f"Position risk {position_risk:.2%} exceeds max {self.max_position_risk:.2%}"
        
        # Check portfolio risk limit
        current_risk = sum(pos.get('value', 0) for pos in current_positions.values()) / portfolio_value if portfolio_value > 0 else 0
        new_risk = current_risk + position_risk
        if new_risk > self.max_portfolio_risk:
            return False, f"Portfolio risk {new_risk:.2%} exceeds max {self.max_portfolio_risk:.2%}"
        
        # Check sector concentration
        sector_value = sum(pos.get('value', 0) for sym, pos in current_positions.items() 
                          if getattr(pos, 'sector', None) == sector)
        new_sector_value = sector_value + position_value
        sector_pct = new_sector_value / portfolio_value if portfolio_value > 0 else 0
        
        # Simple sector limit (could be enhanced with actual sector data)
        max_sector_pct = 0.3  # Max 30% in any sector
        if sector_pct > max_sector_pct:
            return False, f"Sector exposure {sector_pct:.2%} exceeds max {max_sector_pct:.2%}"
        
        # Check correlation (simplified - would need actual correlation matrix)
        # For now, just limit number of positions
        if len(current_positions) >= 10:  # Max 10 positions
            return False, f"Maximum number of positions ({len(current_positions)}) reached"
        
        return True, "OK"
    
    def calculate_portfolio_risk(self, positions: Dict[str, dict], 
                                portfolio_value: float) -> float:
        """
        Calculate current portfolio risk
        
        Args:
            positions: Dictionary of current positions
            portfolio_value: Total portfolio value
            
        Returns:
            Portfolio risk as decimal
        """
        if portfolio_value <= 0:
            return 0
            
        total_risk = sum(pos.get('risk_amount', 0) for pos in positions.values())
        return total_risk / portfolio_value
    
    def get_position_size_adjustment(self, symbol: str, volatility: float,
                                   base_volatility: float = 0.02) -> float:
        """
        Calculate position size adjustment based on volatility
        
        Args:
            symbol: Stock symbol
            volatility: Current volatility (e.g., ATR/price)
            base_volatility: Baseline volatility for normal sizing
            
        Returns:
            Volatility adjustment multiplier
        """
        if volatility <= 0:
            return 1.0
            
        # Inversely proportional to volatility - higher vol = smaller position
        adjustment = base_volatility / volatility
        
        # Bound the adjustment
        return max(0.5, min(adjustment, 2.0))


# Convenience functions
def calculate_position_size(capital: float, entry_price: float, 
                           risk_per_trade: float = 0.02,
                           commission: float = 0.001, slippage: float = 0.0005) -> float:
    """
    Convenience function for position sizing
    
    Args:
        capital: Available capital
        entry_price: Entry price per share
        risk_per_trade: Risk per trade as decimal
        commission: Commission rate
        slippage: Slippage rate
        
    Returns:
        Number of shares to trade
    """
    return PositionSizer.calculate_position_size(
        capital, entry_price, commission, slippage, risk_per_trade
    )


def calculate_stop_loss(entry_price: float, method: str = 'fixed_pct',
                       percentage: float = 0.02, data: Optional[pd.DataFrame] = None,
                       atr_length: int = 14, atr_multiplier: float = 2.0,
                       is_long: bool = True) -> float:
    """
    Convenience function for stop loss calculation
    
    Args:
        entry_price: Entry price
        method: 'fixed_pct' or 'atr'
        percentage: Stop loss percentage (for fixed_pct method)
        data: DataFrame with OHLCV data (for ATR method)
        atr_length: ATR lookback period
        atr_multiplier: ATR multiplier
        is_long: True for long position
        
    Returns:
        Stop loss price
    """
    if method == 'fixed_pct':
        return StopLossManager.fixed_percentage_stop(entry_price, percentage, is_long)
    elif method == 'atr' and data is not None:
        return StopLossManager.atr_stop(data, atr_length, atr_multiplier, is_long)
    else:
        # Default to fixed percentage
        return StopLossManager.fixed_percentage_stop(entry_price, 0.02, is_long)


if __name__ == "__main__":
    # Test the risk management module
    print("Testing Risk Management Module...")
    
    # Test position sizing
    capital = 100000
    entry_price = 150.0
    stop_price = 140.0
    
    pos_size = PositionSizer.fixed_fractional(capital, 0.02, entry_price, stop_price)
    print(f"Fixed fractional position size: {pos_size:.2f} shares")
    print(f"Position value: ${pos_size * entry_price:,.2f}")
    print(f"Risk amount: ${pos_size * (entry_price - stop_price):,.2f}")
    
    # Test Kelly criterion
    kelly_size = PositionSizer.kelly_criterion(0.6, 200, 100, capital, entry_price)
    print(f"Kelly criterion position size: {kelly_size:.2f} shares")
    
    # Test stop loss
    fixed_stop = StopLossManager.fixed_percentage_stop(entry_price, 0.05, True)
    print(f"Fixed 5% stop loss: ${fixed_stop:.2f}")
    
    print("Risk management module test completed!")
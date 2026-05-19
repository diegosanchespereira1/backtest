#!/usr/bin/env python3
"""
Streamlit Dashboard for Stock Backtesting Results
Visualizes backtesting performance with charts, metrics, and trade analysis
"""

import streamlit as st
import pandas as pd
import requests
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os

# Add the backtest engine to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backtest_engine'))

from data_handler import DataHandler
from strategies.base_strategy import StrategyFactory
from risk_management.position_sizer import PositionSizer, StopLossManager
from analytics.performance import PerformanceAnalyzer
from visualization.report_generator import ReportGenerator

# Page configuration
st.set_page_config(
    page_title="Stock Strategy Backtesting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .positive { color: #28a745; }
    .negative { color: #dc3545; }
    .neutral { color: #6c757d; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300, show_spinner=False)
def _search_yahoo(query: str):
    """Query Yahoo Finance search API and return EQUITY results."""
    try:
        r = requests.get(
            "https://query1.finance.yahoo.com/v1/finance/search",
            params={"q": query, "quotesCount": 8, "newsCount": 0, "enableFuzzyQuery": True},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=5,
        )
        return [
            q for q in r.json().get("quotes", [])
            if q.get("quoteType") in ("EQUITY", "ETF", "FUND")
        ]
    except Exception:
        return []


def main():
    st.markdown('<h1 class="main-header">📈 Stock Strategy Backtesting Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Strategy selection
    strategy_options = {
        'rsi_bbands': 'RSI + Bollinger Bands',
        'macd_rsi_vol': 'MACD + RSI + Volume Filter',
        'ema_adx_rsi': 'EMA Trend + ADX + RSI Pullback',
        'bbands_rsi_cci': 'BBands Width + RSI + CCI'
    }
    
    selected_strategy = st.sidebar.selectbox(
        "Select Strategy",
        options=list(strategy_options.keys()),
        format_func=lambda x: strategy_options[x],
        index=0
    )

    _sidebar_desc = {
        'rsi_bbands':    "**Entry:** RSI < oversold **and** price < Lower BB\n\n**Exit:** RSI > overbought **and** price > Upper BB\n\n*Mean-reversion: buys dips at the lower band when momentum is oversold.*",
        'macd_rsi_vol':  "**Entry:** MACD crosses **above** signal, RSI in 40–60 neutral zone, volume > avg × multiplier\n\n**Exit:** MACD crosses **below** signal with same filters\n\n*Momentum + volume confirmation: avoids extremes, only trades confirmed breakouts.*",
        'ema_adx_rsi':   "**Entry:** EMA_fast > EMA_slow (uptrend), ADX > threshold (strong trend), RSI < buy level (pullback)\n\n**Exit:** EMA_fast < EMA_slow (downtrend), ADX strong, RSI > sell level\n\n*Trend-following with pullback entries: rides the trend, enters on temporary weakness.*",
        'bbands_rsi_cci':"**Entry:** Price breaks **above** upper BB after a squeeze (low BB Width), CCI > 100, RSI < 70\n\n**Exit:** Price breaks **below** lower BB after squeeze, CCI < −100, RSI > 30\n\n*Volatility-breakout: waits for compression, trades the expansion with CCI/RSI confirmation.*",
    }
    with st.sidebar.expander("ℹ️ How this strategy works"):
        st.markdown(_sidebar_desc[selected_strategy])
    
    # Symbol selection — search by name or ticker
    st.sidebar.subheader("🔍 Symbol Search")

    if "selected_symbols" not in st.session_state:
        st.session_state.selected_symbols = ["AAPL", "MSFT", "GOOGL"]

    search_q = st.sidebar.text_input(
        "Search by name or ticker",
        placeholder="e.g. McDonald's, NVDA, Apple...",
        key="sym_search_input",
    )

    if search_q and len(search_q) >= 2:
        with st.sidebar:
            with st.spinner("Searching…"):
                _results = _search_yahoo(search_q)
        if _results:
            st.sidebar.caption("Click to add:")
            for _r in _results[:6]:
                _sym   = _r.get("symbol", "")
                _name  = _r.get("shortname") or _r.get("longname") or _sym
                _exch  = _r.get("exchDisp", "")
                _label = f"**{_sym}** — {_name[:30]} _{_exch}_"
                if st.sidebar.button(f"➕ {_sym}  {_name[:25]}", key=f"add_{_sym}_{search_q}"):
                    if _sym not in st.session_state.selected_symbols:
                        st.session_state.selected_symbols.append(_sym)
                    st.rerun()
        else:
            st.sidebar.caption("No results found.")

    # Show and manage selected symbols
    st.sidebar.caption("**Selected symbols:**")
    _to_remove = []
    if st.session_state.selected_symbols:
        _cols = st.sidebar.columns([4, 1])
        for _sym in st.session_state.selected_symbols:
            _c1, _c2 = st.sidebar.columns([4, 1])
            _c1.markdown(f"• `{_sym}`")
            if _c2.button("✕", key=f"rm_{_sym}"):
                _to_remove.append(_sym)
    else:
        st.sidebar.info("No symbols selected.")

    for _sym in _to_remove:
        st.session_state.selected_symbols.remove(_sym)
        st.rerun()

    symbols = st.session_state.selected_symbols
    
    # Date range
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=365*2),
            max_value=datetime.now() - timedelta(days=30)
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now() - timedelta(days=1),
            max_value=datetime.now()
        )
    
    # Timeframe selection
    st.sidebar.subheader("Timeframe")
    timeframe_options = {
        '1d':   '1 Day',
        '1wk':  '1 Week',
        '240m': '240 Minutes (4H)',
        '60m':  '60 Minutes (1H)',
    }
    selected_timeframe = st.sidebar.selectbox(
        "Candle Interval",
        options=list(timeframe_options.keys()),
        format_func=lambda x: timeframe_options[x],
        index=0
    )
    if selected_timeframe in ('60m', '240m'):
        st.sidebar.caption("⚠️ Intraday: 60m supports up to 730 days; 240m is resampled from 60m data.")

    # Parameters
    st.sidebar.subheader("Strategy Parameters")
    
    # Strategy-specific parameters
    params = {}
    if selected_strategy == 'rsi_bbands':
        params['rsi_length'] = st.sidebar.slider("RSI Length", 5, 30, 14)
        params['bb_length'] = st.sidebar.slider("BB Length", 10, 50, 20)
        params['bb_std'] = st.sidebar.slider("BB Std Dev", 1.0, 3.0, 2.0, 0.1)
        params['rsi_oversold'] = st.sidebar.slider("RSI Oversold", 10, 40, 30)
        params['rsi_overbought'] = st.sidebar.slider("RSI Overbought", 60, 90, 70)
    elif selected_strategy == 'macd_rsi_vol':
        params['macd_fast'] = st.sidebar.slider("MACD Fast", 5, 20, 12)
        params['macd_slow'] = st.sidebar.slider("MACD Slow", 20, 40, 26)
        params['macd_signal'] = st.sidebar.slider("MACD Signal", 5, 20, 9)
        params['rsi_length'] = st.sidebar.slider("RSI Length", 5, 30, 14)
        params['rsi_low'] = st.sidebar.slider("RSI Low Threshold", 20, 50, 40)
        params['rsi_high'] = st.sidebar.slider("RSI High Threshold", 50, 80, 60)
        params['volume_multiplier'] = st.sidebar.slider("Volume Multiplier", 1.0, 3.0, 1.5, 0.1)
        params['volume_length'] = st.sidebar.slider("Volume MA Length", 5, 50, 20)
    elif selected_strategy == 'ema_adx_rsi':
        params['ema_fast'] = st.sidebar.slider("EMA Fast", 10, 50, 50)
        params['ema_slow'] = st.sidebar.slider("EMA Slow", 50, 300, 200)
        params['adx_length'] = st.sidebar.slider("ADX Length", 5, 30, 14)
        params['adx_threshold'] = st.sidebar.slider("ADX Threshold", 15, 40, 25)
        params['rsi_length'] = st.sidebar.slider("RSI Length", 5, 30, 14)
        params['rsi_buy'] = st.sidebar.slider("RSI Buy Threshold", 20, 50, 40)
        params['rsi_sell'] = st.sidebar.slider("RSI Sell Threshold", 50, 80, 60)
    elif selected_strategy == 'bbands_rsi_cci':
        params['bb_length'] = st.sidebar.slider("BB Length", 10, 50, 20)
        params['bb_std'] = st.sidebar.slider("BB Std Dev", 1.0, 3.0, 2.0, 0.1)
        params['rsi_length'] = st.sidebar.slider("RSI Length", 5, 30, 14)
        params['cci_length'] = st.sidebar.slider("CCI Length", 5, 50, 20)
        params['squeeze_threshold'] = st.sidebar.slider("Squeeze Ratio (vs BB Width avg)", 0.3, 1.0, 0.8, 0.05)
    
    # Risk management
    st.sidebar.subheader("Risk Management")
    initial_capital = st.sidebar.number_input("Initial Capital ($)", 1000, 1000000, 100000, step=1000)
    commission = st.sidebar.slider("Commission (%)", 0.0, 1.0, 0.1, 0.01) / 100
    slippage = st.sidebar.slider("Slippage (%)", 0.0, 1.0, 0.05, 0.01) / 100
    risk_per_trade = st.sidebar.slider("Risk per Trade (%)", 0.5, 5.0, 2.0, 0.1) / 100
    
    # Run button
    run_backtest = st.sidebar.button("🚀 Run Backtest", type="primary")
    
    # Main content area
    if run_backtest and symbols:
        with st.spinner("Running backtest... Please wait."):
            try:
                # Initialize components
                data_handler = DataHandler()
                strategy = StrategyFactory.create(selected_strategy, **params)
                
                if strategy is None:
                    st.error(f"Failed to create strategy: {selected_strategy}")
                    return
                
                # Fetch data
                _fetch_interval = '60m' if selected_timeframe == '240m' else selected_timeframe
                data = data_handler.fetch_data(
                    symbols,
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d'),
                    interval=_fetch_interval,
                )
                # Resample 60m → 240m (4H) — yfinance has no native 4H interval
                if selected_timeframe == '240m':
                    for _sym in list(data.keys()):
                        _df = data[_sym]
                        data[_sym] = _df.resample('240min').agg({
                            'Open': 'first', 'High': 'max', 'Low': 'min',
                            'Close': 'last', 'Volume': 'sum',
                        }).dropna(subset=['Open', 'Close'])
                
                if not data:
                    st.error("No data retrieved for the specified symbols and date range.")
                    return
                
                # Run backtest for each symbol
                results = {}
                all_trades = []
                portfolio_values_list = []
                
                for symbol in symbols:
                    if symbol not in data:
                        continue
                        
                    symbol_data = data[symbol].copy()
                    
                    # Calculate indicators
                    symbol_data = strategy.calculate_indicators(symbol_data)
                    symbol_data = strategy.generate_signals(symbol_data)
                    
                    # Simple backtest logic
                    cash = initial_capital / len(symbols)  # Divide capital equally
                    position = 0
                    entry_price = 0
                    stop_loss_price = 0
                    trades = []
                    portfolio_values = []
                    
                    for i in range(1, len(symbol_data)):
                        current_date = symbol_data.index[i]
                        current_price = symbol_data.iloc[i]['Close']
                        prev_signal = symbol_data.iloc[i-1]['signal'] if 'signal' in symbol_data.columns else 0
                        
                        # Calculate portfolio value
                        portfolio_value = cash + (position * current_price)
                        portfolio_values.append({
                            'date': current_date,
                            'portfolio_value': portfolio_value,
                            'symbol': symbol
                        })
                        
                        # Execute trades
                        # Stop loss check (runs before signal checks)
                        if position > 0 and stop_loss_price > 0 and current_price <= stop_loss_price:
                            proceeds = position * current_price * (1 - commission - slippage)
                            cash += proceeds
                            sl_pnl = proceeds - (position * entry_price * (1 + commission + slippage))
                            sl_pnl_pct = (sl_pnl / (position * entry_price * (1 + commission + slippage))) * 100
                            trades.append({
                                'date': current_date,
                                'symbol': symbol,
                                'action': 'SELL',
                                'exit_type': 'STOP_LOSS',
                                'price': current_price,
                                'entry_price': entry_price,
                                'stop_loss': stop_loss_price,
                                'shares': position,
                                'value': position * current_price,
                                'pnl': sl_pnl,
                                'pnl_pct': sl_pnl_pct,
                                'cash_after': cash
                            })
                            position = 0
                            entry_price = 0
                            stop_loss_price = 0

                        if prev_signal == 1 and position <= 0:  # Buy signal
                            # Calculate position size
                            max_shares = cash // current_price
                            if max_shares > 0:
                                position = max_shares
                                cost = position * current_price * (1 + commission + slippage)
                                cash -= cost
                                entry_price = current_price
                                stop_loss_price = current_price * (1 - risk_per_trade * 2)
                                trades.append({
                                    'date': current_date,
                                    'symbol': symbol,
                                    'action': 'BUY',
                                    'price': current_price,
                                    'stop_loss': stop_loss_price,
                                    'shares': position,
                                    'value': position * current_price,
                                    'cash_after': cash
                                })
                        
                        elif prev_signal == -1 and position >= 0:  # Sell signal
                            if position > 0:
                                proceeds = position * current_price * (1 - commission - slippage)
                                cash += proceeds
                                pnl = proceeds - (position * entry_price * (1 + commission + slippage))
                                pnl_pct = (pnl / (position * entry_price * (1 + commission + slippage))) * 100
                                trades.append({
                                    'date': current_date,
                                    'symbol': symbol,
                                    'action': 'SELL',
                                    'exit_type': 'SIGNAL',
                                    'price': current_price,
                                    'entry_price': entry_price,
                                    'stop_loss': stop_loss_price,
                                    'shares': position,
                                    'value': position * current_price,
                                    'pnl': pnl,
                                    'pnl_pct': pnl_pct,
                                    'cash_after': cash
                                })
                                position = 0
                                entry_price = 0
                                stop_loss_price = 0
                    
                    # Final portfolio value
                    final_price = symbol_data.iloc[-1]['Close']
                    final_portfolio_value = cash + (position * final_price)
                    portfolio_values.append({
                        'date': symbol_data.index[-1],
                        'portfolio_value': final_portfolio_value,
                        'symbol': symbol
                    })
                    
                    results[symbol] = {
                        'data': symbol_data,
                        'trades': pd.DataFrame(trades) if trades else pd.DataFrame(),
                        'portfolio_values': pd.DataFrame(portfolio_values),
                        'initial_capital': initial_capital / len(symbols),
                        'final_value': final_portfolio_value,
                        'total_return': (final_portfolio_value - (initial_capital / len(symbols))) / (initial_capital / len(symbols)) if (initial_capital / len(symbols)) > 0 else 0
                    }
                    
                    all_trades.extend(trades)
                    portfolio_values_list.extend(portfolio_values)
                
                # Save results to session_state so display survives widget reruns
                st.session_state.backtest_results = results
                st.session_state.backtrack_all_trades = all_trades
                st.session_state.backtest_portfolio_values = portfolio_values_list

            except Exception as e:
                st.session_state.pop("backtest_results", None)
                st.error(f"An error occurred during backtesting: {str(e)}")
                st.exception(e)
    elif run_backtest:
        st.warning("Please enter at least one symbol to test.")

    # ── Show results (persists across widget reruns via session_state) ──
    if "backtest_results" in st.session_state:
        results = st.session_state.backtest_results
        all_trades = st.session_state.backtrack_all_trades
        portfolio_values_list = st.session_state.backtest_portfolio_values

        # Display results
        st.success(f"Backtest completed for {len(results)} symbols!")
        
        # Overall metrics
        total_initial = sum(r['initial_capital'] for r in results.values())
        total_final = sum(r['final_value'] for r in results.values())
        overall_return = (total_final - total_initial) / total_initial if total_initial > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Initial Capital", f"${total_initial:,.2f}")
        with col2:
            st.metric("Final Value", f"${total_final:,.2f}")
        with col3:
            st.metric("Total Return", f"{overall_return:.2%}", 
                     delta=f"{overall_return:.2%}")
        with col4:
            win_trades = len([t for t in all_trades if t.get('pnl', 0) > 0]) if all_trades else 0
            total_trades = len([t for t in all_trades if t['action'] == 'SELL']) if all_trades else 0
            win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0
            st.metric("Win Rate", f"{win_rate:.1f}%")
        
        # Charts
        st.subheader("📊 Performance Charts")
        
        # Equity curve
        if portfolio_values_list:
            portfolio_df = pd.DataFrame(portfolio_values_list)
            # Pivot to get portfolio value over time (sum across symbols)
            equity_pivot = portfolio_df.pivot_table(index='date', columns='symbol', values='portfolio_value', aggfunc='last').ffill().fillna(0)
            equity_pivot['Total'] = equity_pivot.sum(axis=1)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=equity_pivot.index,
                y=equity_pivot['Total'],
                mode='lines',
                name='Total Portfolio',
                line=dict(color='blue', width=2)
            ))
            
            # Add individual symbols
            colors = px.colors.qualitative.Set1
            for i, symbol in enumerate(symbols):
                if symbol in equity_pivot.columns:
                    fig.add_trace(go.Scatter(
                        x=equity_pivot.index,
                        y=equity_pivot[symbol],
                        mode='lines',
                        name=symbol,
                        line=dict(color=colors[i % len(colors)], width=1),
                        opacity=0.7
                    ))
            
            fig.update_layout(
                title="Portfolio Equity Curve",
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                hovermode='x unified',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Drawdown chart
        if 'Total' in equity_pivot.columns:
            rolling_max = equity_pivot['Total'].cummax()
            drawdown = (equity_pivot['Total'] - rolling_max) / rolling_max * 100
            
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=equity_pivot.index,
                y=drawdown,
                mode='lines',
                fill='tozeroy',
                name='Drawdown',
                line=dict(color='red'),
                fillcolor='rgba(255,0,0,0.3)'
            ))
            fig_dd.update_layout(
                title="Portfolio Drawdown",
                xaxis_title="Date",
                yaxis_title="Drawdown (%)",
                height=300
            )
            st.plotly_chart(fig_dd, use_container_width=True)
        
        # Per-symbol price chart with entry / exit / stop-loss signals
        st.subheader("📈 Trade Signals — Price Chart")
        signal_symbol = st.selectbox("Select symbol to inspect", options=list(results.keys()), key="sig_sym")
        chart_type = st.radio("Chart type", ["Line", "Candlestick"], horizontal=True, key="chart_type")
        if signal_symbol and signal_symbol in results:
            sym_data   = results[signal_symbol]['data']
            sym_trades = results[signal_symbol]['trades']

            # Detect available indicators
            has_bb     = 'BBU' in sym_data.columns
            has_ema    = 'EMA_Fast' in sym_data.columns
            has_rsi    = 'RSI' in sym_data.columns
            has_macd   = 'MACD' in sym_data.columns
            has_adx    = 'ADX' in sym_data.columns
            has_cci    = 'CCI' in sym_data.columns
            has_vol_ma = 'Volume_MA' in sym_data.columns

            n_rows = 4 if (has_macd or has_adx or has_cci) else 3
            label4 = 'MACD' if has_macd else ('ADX' if has_adx else 'CCI')
            row_heights = [0.50, 0.15, 0.18, 0.17] if n_rows == 4 else [0.55, 0.20, 0.25]
            subplot_titles = [f'{signal_symbol} — Price', 'Volume', 'RSI'] + ([label4] if n_rows == 4 else [])

            fig_sig = make_subplots(
                rows=n_rows, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=subplot_titles,
                row_heights=row_heights,
            )

            # ── Panel 1: Price + BB / EMA overlays ────────────────
            if has_bb:
                fig_sig.add_trace(go.Scatter(
                    x=sym_data.index, y=sym_data['BBU'],
                    mode='lines', name='BB Upper',
                    line=dict(color='rgba(150,150,150,0.5)', width=1, dash='dot'),
                ), row=1, col=1)
                fig_sig.add_trace(go.Scatter(
                    x=sym_data.index, y=sym_data['BBL'],
                    mode='lines', name='BB Lower',
                    line=dict(color='rgba(150,150,150,0.5)', width=1, dash='dot'),
                    fill='tonexty', fillcolor='rgba(150,150,150,0.07)',
                ), row=1, col=1)
                fig_sig.add_trace(go.Scatter(
                    x=sym_data.index, y=sym_data['BBM'],
                    mode='lines', name='BB Mid',
                    line=dict(color='rgba(150,150,150,0.4)', width=1),
                ), row=1, col=1)

            if has_ema:
                fig_sig.add_trace(go.Scatter(
                    x=sym_data.index, y=sym_data['EMA_Fast'],
                    mode='lines', name='EMA Fast',
                    line=dict(color='orange', width=1.2),
                ), row=1, col=1)
                fig_sig.add_trace(go.Scatter(
                    x=sym_data.index, y=sym_data['EMA_Slow'],
                    mode='lines', name='EMA Slow',
                    line=dict(color='purple', width=1.2),
                ), row=1, col=1)

            if chart_type == "Candlestick":
                fig_sig.add_trace(go.Candlestick(
                    x=sym_data.index,
                    open=sym_data['Open'], high=sym_data['High'],
                    low=sym_data['Low'],   close=sym_data['Close'],
                    name='Price',
                    increasing_line_color='#26a69a',
                    decreasing_line_color='#ef5350',
                ), row=1, col=1)
            else:
                fig_sig.add_trace(go.Scatter(
                    x=sym_data.index, y=sym_data['Close'],
                    mode='lines', name='Close',
                    line=dict(color='#1f77b4', width=1.5),
                ), row=1, col=1)

            # Trade markers on price panel
            if not sym_trades.empty:
                buys      = sym_trades[sym_trades['action'] == 'BUY']
                sells     = sym_trades[sym_trades['action'] == 'SELL']
                sl_exits  = sells[sells['exit_type'] == 'STOP_LOSS'] if 'exit_type' in sells.columns else pd.DataFrame()
                sig_exits = sells[sells['exit_type'] == 'SIGNAL']    if 'exit_type' in sells.columns else sells

                if not buys.empty:
                    cd = buys[['shares', 'stop_loss']].values if 'stop_loss' in buys.columns else buys[['shares']].values
                    ht = '<b>BUY</b><br>%{x}<br>$%{y:.2f}<br>Shares: %{customdata[0]}<br>SL: $%{customdata[1]:.2f}<extra></extra>' if 'stop_loss' in buys.columns else '<b>BUY</b><br>%{x}<br>$%{y:.2f}<extra></extra>'
                    fig_sig.add_trace(go.Scatter(
                        x=buys['date'], y=buys['price'],
                        mode='markers', name='Entry (BUY)',
                        marker=dict(symbol='triangle-up', size=13, color='green'),
                        customdata=cd, hovertemplate=ht,
                    ), row=1, col=1)

                if not sig_exits.empty:
                    cd2 = sig_exits[['pnl', 'pnl_pct']].values if 'pnl' in sig_exits.columns else None
                    ht2 = '<b>SELL</b><br>%{x}<br>$%{y:.2f}<br>P&L: $%{customdata[0]:,.2f} (%{customdata[1]:.2f}%)<extra></extra>' if 'pnl' in sig_exits.columns else '<b>SELL</b><br>%{x}<br>$%{y:.2f}<extra></extra>'
                    fig_sig.add_trace(go.Scatter(
                        x=sig_exits['date'], y=sig_exits['price'],
                        mode='markers', name='Exit (Signal)',
                        marker=dict(symbol='triangle-down', size=13, color='red'),
                        customdata=cd2, hovertemplate=ht2,
                    ), row=1, col=1)

                if 'exit_type' in sells.columns and not sl_exits.empty:
                    cd3 = sl_exits[['pnl', 'pnl_pct', 'stop_loss']].values if 'pnl' in sl_exits.columns else None
                    fig_sig.add_trace(go.Scatter(
                        x=sl_exits['date'], y=sl_exits['price'],
                        mode='markers', name='Stop Loss Hit',
                        marker=dict(symbol='x', size=14, color='orange', line=dict(width=2)),
                        customdata=cd3,
                        hovertemplate='<b>STOP LOSS</b><br>%{x}<br>$%{y:.2f}<br>SL: $%{customdata[2]:.2f}<br>P&L: $%{customdata[0]:,.2f} (%{customdata[1]:.2f}%)<extra></extra>',
                    ), row=1, col=1)

                if 'stop_loss' in buys.columns:
                    for _, buy_row in buys.iterrows():
                        next_sell = sym_trades[(sym_trades['action'] == 'SELL') & (sym_trades['date'] > buy_row['date'])]
                        exit_date = next_sell.iloc[0]['date'] if not next_sell.empty else sym_data.index[-1]
                        fig_sig.add_shape(
                            type='line',
                            x0=buy_row['date'], x1=exit_date,
                            y0=buy_row['stop_loss'], y1=buy_row['stop_loss'],
                            line=dict(color='orange', width=1, dash='dot'),
                            row=1, col=1,
                        )

            # ── Panel 2: Volume ────────────────────────────────────
            vol_colors = ['green' if c >= o else 'red'
                          for c, o in zip(sym_data['Close'], sym_data['Open'])]
            fig_sig.add_trace(go.Bar(
                x=sym_data.index, y=sym_data['Volume'],
                name='Volume', marker_color=vol_colors, opacity=0.5, showlegend=False,
            ), row=2, col=1)
            if has_vol_ma:
                fig_sig.add_trace(go.Scatter(
                    x=sym_data.index, y=sym_data['Volume_MA'],
                    mode='lines', name='Vol MA',
                    line=dict(color='navy', width=1),
                ), row=2, col=1)

            # ── Panel 3: RSI ───────────────────────────────────────
            if has_rsi:
                fig_sig.add_trace(go.Scatter(
                    x=sym_data.index, y=sym_data['RSI'],
                    mode='lines', name='RSI',
                    line=dict(color='purple', width=1.2),
                ), row=3, col=1)
                rsi_ob = params.get('rsi_overbought', params.get('rsi_high',  params.get('rsi_sell', 70)))
                rsi_os = params.get('rsi_oversold',  params.get('rsi_low',   params.get('rsi_buy',  30)))
                for lvl, clr in [(rsi_ob, 'rgba(220,50,50,0.6)'), (rsi_os, 'rgba(50,180,50,0.6)'), (50, 'rgba(128,128,128,0.4)')]:
                    fig_sig.add_hline(y=lvl, line_dash='dot', line_color=clr, line_width=1, row=3, col=1)

            # ── Panel 4: MACD / ADX / CCI ─────────────────────────
            if has_macd:
                hist_clr = ['green' if v >= 0 else 'red' for v in sym_data['MACDHIST'].fillna(0)]
                fig_sig.add_trace(go.Bar(
                    x=sym_data.index, y=sym_data['MACDHIST'],
                    name='MACD Hist', marker_color=hist_clr, opacity=0.5,
                ), row=4, col=1)
                fig_sig.add_trace(go.Scatter(
                    x=sym_data.index, y=sym_data['MACD'],
                    mode='lines', name='MACD', line=dict(color='blue', width=1.2),
                ), row=4, col=1)
                fig_sig.add_trace(go.Scatter(
                    x=sym_data.index, y=sym_data['MACDS'],
                    mode='lines', name='Signal', line=dict(color='orange', width=1.2),
                ), row=4, col=1)
                fig_sig.add_hline(y=0, line_dash='dot', line_color='gray', line_width=1, row=4, col=1)
            elif has_adx:
                fig_sig.add_trace(go.Scatter(
                    x=sym_data.index, y=sym_data['ADX'],
                    mode='lines', name='ADX', line=dict(color='darkblue', width=1.2),
                ), row=4, col=1)
                adx_thresh = params.get('adx_threshold', 25)
                fig_sig.add_hline(y=adx_thresh, line_dash='dot', line_color='red', line_width=1, row=4, col=1)
            elif has_cci:
                fig_sig.add_trace(go.Scatter(
                    x=sym_data.index, y=sym_data['CCI'],
                    mode='lines', name='CCI', line=dict(color='teal', width=1.2),
                ), row=4, col=1)
                for lvl in [100, -100, 0]:
                    fig_sig.add_hline(y=lvl, line_dash='dot',
                                      line_color='rgba(220,50,50,0.6)' if abs(lvl) == 100 else 'rgba(128,128,128,0.4)',
                                      line_width=1, row=4, col=1)

            fig_sig.update_layout(
                title=f'{signal_symbol} — Price, Volume & Indicators',
                hovermode='x unified',
                height=850,
                legend=dict(orientation='h', yanchor='bottom', y=1.01, x=0),
                xaxis_rangeslider_visible=False,
                barmode='overlay',
            )
            fig_sig.update_yaxes(title_text='Price ($)', row=1, col=1)
            fig_sig.update_yaxes(title_text='Volume',   row=2, col=1)
            fig_sig.update_yaxes(title_text='RSI',      row=3, col=1)
            if n_rows == 4:
                fig_sig.update_yaxes(title_text=label4, row=4, col=1)
            st.plotly_chart(fig_sig, use_container_width=True)

        # Trade analysis
        if all_trades:
            st.subheader("📋 Trade Analysis")
            
            trades_df = pd.DataFrame(all_trades)
            sell_trades = trades_df[trades_df['action'] == 'SELL'].copy()
            
            if not sell_trades.empty:
                # Trade P&L distribution
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=sell_trades['pnl_pct'],
                    nbinsx=20,
                    name='Trade P&L (%)',
                    marker_color='lightblue'
                ))
                fig_hist.update_layout(
                    title="Distribution of Trade Returns (%)",
                    xaxis_title="P&L (%)",
                    yaxis_title="Frequency",
                    height=400
                )
                st.plotly_chart(fig_hist, use_container_width=True)
                
                # Paired entry/exit trade table
                st.subheader("Trade Log — Entry / Exit / Stop Loss")
                buy_trades = trades_df[trades_df['action'] == 'BUY'].reset_index(drop=True)
                paired_rows = []
                for _, sell_row in sell_trades.iterrows():
                    entry_dt  = sell_row.get('entry_price', None)
                    # find matching buy before this sell
                    buy_match = buy_trades[buy_trades['date'] <= sell_row['date']]
                    entry_date = buy_match.iloc[-1]['date'] if not buy_match.empty else '—'
                    entry_px   = sell_row.get('entry_price', buy_match.iloc[-1]['price'] if not buy_match.empty else None)
                    exit_type  = sell_row.get('exit_type', 'SIGNAL')
                    stop_px    = sell_row.get('stop_loss', None)
                    pnl_val    = sell_row.get('pnl', 0)
                    pnl_pct_v  = sell_row.get('pnl_pct', 0)
                    paired_rows.append({
                        'Symbol':      sell_row['symbol'],
                        'Entry Date':  str(entry_date)[:10],
                        'Entry Price': f"${entry_px:.2f}" if entry_px else '—',
                        'Exit Date':   str(sell_row['date'])[:10],
                        'Exit Price':  f"${sell_row['price']:.2f}",
                        'Stop Loss':   f"${stop_px:.2f}" if stop_px else '—',
                        'Exit Type':   exit_type,
                        'Shares':      int(sell_row['shares']),
                        'P&L':         f"${pnl_val:,.2f}",
                        'P&L %':       f"{pnl_pct_v:.2f}%",
                    })
                if paired_rows:
                    paired_df = pd.DataFrame(paired_rows)
                    st.dataframe(paired_df, use_container_width=True)
            else:
                st.info("No completed trades (SELL actions) in this period.")
        
        # Individual symbol performance
        st.subheader("📊 Individual Symbol Performance")
        
        perf_data = []
        for symbol, result in results.items():
            perf_data.append({
                'Symbol': symbol,
                'Initial Capital': f"${result['initial_capital']:,.2f}",
                'Final Value': f"${result['final_value']:,.2f}",
                'Total Return': f"{result['total_return']:.2%}",
                'Trades': len(result['trades'][result['trades']['action'] == 'SELL']) if not result['trades'].empty else 0
            })
        
        perf_df = pd.DataFrame(perf_data)
        st.dataframe(perf_df, use_container_width=True)
        
        # ── Strategy description section ──────────────────────────
        st.subheader("📖 Strategy Description")

        _strategy_docs = {
            'rsi_bbands': {
                'title': 'RSI + Bollinger Bands — Mean Reversion',
                'overview': (
                    "Combines two classic oscillators to identify **oversold/overbought extremes** "
                    "at the edges of price volatility bands. Works best in range-bound or moderately "
                    "trending markets where prices revert to the mean."
                ),
                'indicators': [
                    "**RSI (Relative Strength Index):** Measures momentum on a 0–100 scale. "
                    "Values below the oversold threshold signal exhausted selling; values above "
                    "overbought signal exhausted buying.",
                    "**Bollinger Bands:** A middle SMA surrounded by upper/lower bands at ±N standard "
                    "deviations. Price touching the lower band indicates statistical cheapness; "
                    "upper band indicates expensiveness.",
                ],
                'entry': "RSI < oversold threshold **AND** Close < Lower Bollinger Band",
                'exit':  "RSI > overbought threshold **AND** Close > Upper Bollinger Band",
                'strengths': ["Simple and intuitive", "Works well in sideways markets", "Low signal frequency reduces overtrading"],
                'weaknesses': ["Can give false signals in strong trends", "May enter too early during prolonged downtrends"],
            },
            'macd_rsi_vol': {
                'title': 'MACD + RSI + Volume Filter — Momentum with Confirmation',
                'overview': (
                    "Trades **MACD crossovers** only when RSI is in a neutral zone (not already "
                    "extended) and volume is above average — filtering out weak or exhausted moves "
                    "and requiring genuine market participation."
                ),
                'indicators': [
                    "**MACD (Moving Average Convergence Divergence):** Difference between fast and slow "
                    "EMAs. A crossover of the MACD line above its signal line indicates bullish momentum shift.",
                    "**RSI (neutral filter):** Used here not for extremes but to confirm the move is "
                    "starting from a neutral state — avoiding entries into already-extended rallies.",
                    "**Volume Filter:** Compares current volume to a rolling average. High volume confirms "
                    "that institutional participation supports the move.",
                ],
                'entry': "MACD crosses **above** Signal line **AND** RSI between low–high thresholds **AND** Volume > average × multiplier",
                'exit':  "MACD crosses **below** Signal line **AND** same RSI + volume filters apply",
                'strengths': ["Volume confirmation reduces false breakouts", "RSI filter avoids chasing extended moves", "Good for trending markets"],
                'weaknesses': ["MACD lags price — entries can be late", "Neutral RSI filter may miss strong momentum trades"],
            },
            'ema_adx_rsi': {
                'title': 'EMA Trend + ADX + RSI Pullback — Trend Following',
                'overview': (
                    "A **trend-following** strategy that only trades in the direction of the primary trend "
                    "(defined by EMA crossover), when the trend is strong enough (ADX filter), and enters "
                    "on RSI pullbacks rather than chasing breakouts."
                ),
                'indicators': [
                    "**EMA Fast / EMA Slow:** When EMA_fast > EMA_slow the market is in an uptrend; "
                    "below indicates a downtrend. The crossover defines the tradeable direction.",
                    "**ADX (Average Directional Index):** Measures trend *strength* regardless of direction. "
                    "Above the threshold (e.g. 25) confirms the trend is strong enough to trade.",
                    "**RSI (pullback detector):** Used to time entries within the trend. A dip below the "
                    "buy threshold in an uptrend suggests a temporary pullback — a better entry price.",
                ],
                'entry': "EMA_fast > EMA_slow (uptrend) **AND** ADX > threshold **AND** RSI < buy level",
                'exit':  "EMA_fast < EMA_slow (downtrend) **AND** ADX > threshold **AND** RSI > sell level",
                'strengths': ["Rides large directional moves", "ADX filter avoids choppy/sideways periods", "Pullback entries improve risk/reward"],
                'weaknesses': ["Slow to react to trend reversals (lagging EMAs)", "Few signals during consolidation phases"],
            },
            'bbands_rsi_cci': {
                'title': 'Bollinger Bands Width + RSI + CCI — Volatility Breakout',
                'overview': (
                    "Detects periods of **volatility compression** (BB squeeze) and trades the subsequent "
                    "expansion breakout, using CCI and RSI to confirm the direction and avoid "
                    "false breakouts into already-extended conditions."
                ),
                'indicators': [
                    "**Bollinger Bands Width (BB Width):** (Upper − Lower) / Middle. When this ratio falls "
                    "below the squeeze threshold, volatility is historically low — a breakout is likely.",
                    "**CCI (Commodity Channel Index):** Measures price deviation from its average. "
                    "CCI > 100 confirms upward momentum; CCI < −100 confirms downward momentum.",
                    "**RSI (extreme filter):** Prevents entries into already overbought/oversold conditions "
                    "where the breakout is likely to reverse.",
                ],
                'entry': "Price breaks **above** Upper BB (after prior squeeze) **AND** CCI > 100 **AND** RSI < 70",
                'exit':  "Price breaks **below** Lower BB (after prior squeeze) **AND** CCI < −100 **AND** RSI > 30",
                'strengths': ["Captures explosive volatility expansions", "Squeeze filter improves signal quality", "CCI provides directional confidence"],
                'weaknesses': ["Squeezes can resolve with small moves (false breakout)", "Requires multiple conditions — fewer signals"],
            },
        }

        doc = _strategy_docs[selected_strategy]
        st.markdown(f"### {doc['title']}")
        st.info(doc['overview'])

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**🟢 Entry condition**")
            st.success(doc['entry'])
            st.markdown("**💪 Strengths**")
            for s in doc['strengths']:
                st.markdown(f"- {s}")
        with col_b:
            st.markdown("**🔴 Exit condition**")
            st.error(doc['exit'])
            st.markdown("**⚠️ Weaknesses**")
            for w in doc['weaknesses']:
                st.markdown(f"- {w}")

        with st.expander("📐 Indicators explained"):
            for ind in doc['indicators']:
                st.markdown(f"- {ind}")

        with st.expander("⚙️ Active parameters & risk settings"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Strategy parameters**")
                for key, value in params.items():
                    st.markdown(f"- `{key}`: **{value}**")
            with c2:
                st.markdown("**Risk management**")
                st.markdown(f"- Initial Capital: **${initial_capital:,.2f}**")
                st.markdown(f"- Commission: **{commission*100:.2f}%**")
                st.markdown(f"- Slippage: **{slippage*100:.2f}%**")
                st.markdown(f"- Risk per Trade: **{risk_per_trade*100:.1f}%**")
        

    else:
        # Show welcome message when no backtest has been run yet
        # Show welcome message
        st.info("👈 Configure your backtest in the sidebar and click 'Run Backtest' to see results.")
        
        # Show example charts
        st.subheader("📈 Example Visualization")
        
        # Create sample data for demonstration
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.02, len(dates))
        equity = 100000 * np.exp(np.cumsum(returns))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=equity,
            mode='lines',
            name='Sample Equity Curve',
            line=dict(color='green', width=2)
        ))
        fig.update_layout(
            title="Sample Equity Curve (Example)",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
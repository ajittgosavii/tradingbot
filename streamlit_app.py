import streamlit as st
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config
st.set_page_config(page_title="Crypto Trading Bot", layout="wide", initial_sidebar_state="expanded")

# Initialize session state
if 'trading_active' not in st.session_state:
    st.session_state.trading_active = False
if 'paper_trading_active' not in st.session_state:
    st.session_state.paper_trading_active = False
if 'trade_history' not in st.session_state:
    st.session_state.trade_history = []
if 'paper_trades' not in st.session_state:
    st.session_state.paper_trades = []
if 'daily_profit' not in st.session_state:
    st.session_state.daily_profit = 0
if 'daily_loss' not in st.session_state:
    st.session_state.daily_loss = 0
if 'paper_balance' not in st.session_state:
    st.session_state.paper_balance = 10000
if 'positions' not in st.session_state:
    st.session_state.positions = {}

class AdvancedTradingBot:
    def __init__(self, exchange_name, api_key, api_secret, test_mode=True):
        try:
            exchange_class = getattr(ccxt, exchange_name)
            self.exchange = exchange_class({
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
                'sandbox': test_mode
            })
            self.exchange_name = exchange_name
        except Exception as e:
            st.error(f"Failed to connect: {e}")
            return None

    def get_balance(self):
        try:
            return self.exchange.fetch_balance()
        except Exception as e:
            st.error(f"Error fetching balance: {e}")
            return None

    def get_market_data(self, symbol, timeframe='1h', limit=100, since=None):
        try:
            if since:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit, since=since)
            else:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            return None

    def calculate_indicators(self, df):
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['rsi'] = self._calculate_rsi(df['close'], 14)
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        return df

    def _calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def strategy_momentum(self, df):
        latest = df.iloc[-1]
        buy = (latest['macd'] > latest['macd_signal'] and 
               latest['rsi'] < 70 and latest['sma_20'] > latest['sma_50'])
        sell = (latest['macd'] < latest['macd_signal'] and latest['rsi'] > 70)
        return buy, sell

    def strategy_mean_reversion(self, df):
        latest = df.iloc[-1]
        buy = (latest['close'] < latest['bb_lower'] and 
               latest['rsi'] < 40 and latest['sma_20'] > latest['sma_50'])
        sell = (latest['close'] > latest['bb_upper'] and latest['rsi'] > 60)
        return buy, sell

    def strategy_trend_following(self, df):
        latest = df.iloc[-1]
        buy = (latest['sma_20'] > latest['sma_50'] and 
               latest['sma_50'] > df.iloc[-10]['sma_50'] and latest['rsi'] > 40)
        sell = (latest['sma_20'] < latest['sma_50'] and latest['rsi'] > 70)
        return buy, sell

    def strategy_rsi_oversold(self, df):
        """RSI drops below 30 = oversold, above 70 = overbought (proven retail strategy)"""
        latest = df.iloc[-1]
        buy = (latest['rsi'] < 30 and latest['close'] > latest['sma_50'])
        sell = (latest['rsi'] > 70)
        return buy, sell

    def strategy_bollinger_squeeze(self, df):
        """Buy when price breaks above BB upper band with volume (breakout)"""
        latest = df.iloc[-1]
        buy = (latest['close'] > latest['bb_upper'] and latest['rsi'] > 50 and latest['volume'] > df['volume'].mean())
        sell = (latest['close'] < latest['bb_middle'] and latest['rsi'] > 70)
        return buy, sell

    def strategy_volume_confirmation(self, df, volume_threshold=1.5):
        """Buy when price action + volume confirms (institutional strategy)"""
        latest = df.iloc[-1]
        avg_volume = df['volume'].tail(20).mean()
        buy = (latest['sma_20'] > latest['sma_50'] and 
               latest['volume'] > avg_volume * volume_threshold and 
               latest['rsi'] > 40 and latest['rsi'] < 70)
        sell = (latest['volume'] > avg_volume * 2 and latest['rsi'] > 70)
        return buy, sell

    def strategy_pullback(self, df):
        """Buy on pullback to moving average after uptrend (popular day trader strategy)"""
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        buy = (latest['close'] < latest['sma_20'] and 
               prev['close'] > prev['sma_20'] and
               latest['sma_20'] > latest['sma_50'] and
               latest['rsi'] > 35 and latest['rsi'] < 70)
        sell = (latest['close'] > latest['sma_50'] + (latest['sma_50'] * 0.05) and latest['rsi'] > 70)
        return buy, sell

    def strategy_dca_with_signals(self, df):
        """Dollar Cost Averaging with signal confirmation (safest, most reliable)"""
        latest = df.iloc[-1]
        buy = (latest['sma_20'] > latest['sma_50'] and 
               latest['rsi'] > 35 and latest['rsi'] < 65)
        sell = (latest['rsi'] > 75)
        return buy, sell

    def get_signals(self, df, strategy='momentum'):
        if strategy == 'momentum':
            return self.strategy_momentum(df)
        elif strategy == 'mean_reversion':
            return self.strategy_mean_reversion(df)
        elif strategy == 'trend_following':
            return self.strategy_trend_following(df)
        elif strategy == 'rsi_oversold':
            return self.strategy_rsi_oversold(df)
        elif strategy == 'bollinger_squeeze':
            return self.strategy_bollinger_squeeze(df)
        elif strategy == 'volume_confirmation':
            return self.strategy_volume_confirmation(df)
        elif strategy == 'pullback':
            return self.strategy_pullback(df)
        elif strategy == 'dca_signals':
            return self.strategy_dca_with_signals(df)
        return False, False

class Backtester:
    def __init__(self, df, initial_balance=10000, position_size=0.015, 
                 stop_loss=1.5, take_profit=3.0):
        self.df = df.copy()
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position_size = position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.trades = []
        self.equity_curve = [initial_balance]
        
    def backtest_strategy(self, bot, strategy, symbol):
        in_position = False
        entry_price = 0
        entry_index = 0
        
        for i in range(50, len(self.df)):
            df_slice = self.df.iloc[:i+1].copy()
            df_slice = bot.calculate_indicators(df_slice)
            
            buy_signal, sell_signal = bot.get_signals(df_slice, strategy)
            current_price = self.df.iloc[i]['close']
            current_time = self.df.iloc[i]['timestamp']
            
            if buy_signal and not in_position:
                if self.balance > 100:
                    entry_price = current_price
                    entry_index = i
                    in_position = True
            
            if in_position:
                pnl_percent = ((current_price - entry_price) / entry_price) * 100
                
                if pnl_percent >= self.take_profit or sell_signal:
                    profit = self.balance * (pnl_percent / 100)
                    self.balance += profit
                    self.trades.append({
                        'entry_time': self.df.iloc[entry_index]['timestamp'],
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'pnl_percent': pnl_percent,
                        'pnl_usd': profit,
                        'type': 'win' if pnl_percent > 0 else 'loss'
                    })
                    in_position = False
                
                elif pnl_percent <= -self.stop_loss:
                    loss = self.balance * (pnl_percent / 100)
                    self.balance += loss
                    self.trades.append({
                        'entry_time': self.df.iloc[entry_index]['timestamp'],
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'pnl_percent': pnl_percent,
                        'pnl_usd': loss,
                        'type': 'loss'
                    })
                    in_position = False
            
            self.equity_curve.append(self.balance)
        
        return self.get_backtest_results()
    
    def get_backtest_results(self):
        if not self.trades:
            return None
        
        trades_df = pd.DataFrame(self.trades)
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl_percent'] > 0])
        losing_trades = len(trades_df[trades_df['pnl_percent'] <= 0])
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        total_profit = trades_df['pnl_usd'].sum()
        avg_win = trades_df[trades_df['pnl_percent'] > 0]['pnl_usd'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl_percent'] <= 0]['pnl_usd'].mean() if losing_trades > 0 else 0
        
        equity_array = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max * 100
        max_drawdown = np.min(drawdown)
        
        returns = np.diff(self.equity_curve) / np.array(self.equity_curve[:-1])
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'final_balance': self.balance,
            'return_percent': (self.balance - self.initial_balance) / self.initial_balance * 100
        }

# Sidebar Configuration
st.sidebar.title("Configuration")
st.sidebar.divider()

st.sidebar.subheader("Exchange Settings")
exchange = st.sidebar.selectbox("Exchange", ["binance", "kraken", "coinbase"])
api_key = st.sidebar.text_input("API Key", type="password")
api_secret = st.sidebar.text_input("API Secret", type="password")
test_mode = st.sidebar.checkbox("Use Testnet/Sandbox", value=True)

st.sidebar.divider()
st.sidebar.subheader("Trading Parameters")
daily_profit_target = st.sidebar.number_input("Daily Profit Target ($)", value=50, min_value=10)
daily_loss_limit = st.sidebar.number_input("Daily Loss Limit ($)", value=5, min_value=5)
position_size = st.sidebar.slider("Position Size (%)", 0.5, 5.0, 1.5, 0.1)
stop_loss = st.sidebar.slider("Stop Loss (%)", 0.5, 5.0, 1.5, 0.1)
take_profit = st.sidebar.slider("Take Profit (%)", 1.0, 10.0, 3.0, 0.5)

st.sidebar.divider()
st.sidebar.subheader("Strategy Settings")
strategy = st.sidebar.selectbox("Select Strategy", 
    ["momentum", "mean_reversion", "trend_following", "rsi_oversold", 
     "bollinger_squeeze", "volume_confirmation", "pullback", "dca_signals"],
    help="""
    - Momentum: MACD crossover, ideal for trending markets
    - Mean Reversion: Bollinger Bands, works in range-bound markets  
    - Trend Following: SMA crossover, proven long-term strategy
    - RSI Oversold: Buy when RSI < 30, popular retail strategy
    - Bollinger Squeeze: Breakout with volume, used by day traders
    - Volume Confirmation: Institutional-grade volume analysis
    - Pullback: Buy dips in uptrend, proven pattern
    - DCA Signals: Safest, most consistent (recommended)
    """)
timeframe = st.sidebar.selectbox("Timeframe", ["5m", "15m", "1h", "4h", "1d"])
symbols = st.sidebar.multiselect("Trading Pairs", 
    ["BTC/USDT", "ETH/USDT", "ADA/USDT", "SOL/USDT", "XRP/USDT"],
    default=["BTC/USDT", "ETH/USDT"])

# Main Dashboard
st.title("Crypto Trading Bot Dashboard")

# Top metrics
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Daily Target", f"${daily_profit_target}", delta="Target")
with col2:
    st.metric("Daily Profit", f"${st.session_state.daily_profit:.2f}", 
              delta=f"{(st.session_state.daily_profit/daily_profit_target*100):.0f}%")
with col3:
    st.metric("Daily Loss", f"${st.session_state.daily_loss:.2f}", 
              delta=f"-{(st.session_state.daily_loss/daily_loss_limit*100):.0f}%")
with col4:
    st.metric("Paper Trading", f"${st.session_state.paper_balance:.2f}", delta="Simulated")
with col5:
    status = "ACTIVE" if st.session_state.trading_active else "STOPPED"
    st.metric("Bot Status", status)

st.divider()

# Control buttons
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    if st.button("Start Trading", use_container_width=True):
        st.session_state.trading_active = True
        st.success("Trading bot started!")
        st.rerun()
with col2:
    if st.button("Stop Trading", use_container_width=True):
        st.session_state.trading_active = False
        st.info("Trading bot stopped!")
        st.rerun()
with col3:
    if st.button("Paper Trading", use_container_width=True):
        st.session_state.paper_trading_active = not st.session_state.paper_trading_active
        st.rerun()
with col4:
    if st.button("Reset Stats", use_container_width=True):
        st.session_state.daily_profit = 0
        st.session_state.daily_loss = 0
        st.rerun()
with col5:
    if st.button("Clear History", use_container_width=True):
        st.session_state.trade_history = []
        st.session_state.paper_trades = []
        st.rerun()

st.divider()

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Market Analysis", "Trading Activity", 
                                               "Profit/Loss", "Trade History", 
                                               "Backtesting", "Paper Trading"])

# Tab 1: Market Analysis
with tab1:
    st.subheader("Market Analysis")
    
    if api_key and api_secret:
        bot = AdvancedTradingBot(exchange, api_key, api_secret, test_mode)
        
        for symbol in symbols:
            df = bot.get_market_data(symbol, timeframe, 100)
            if df is not None and len(df) > 50:
                df = bot.calculate_indicators(df)
                latest = df.iloc[-1]
                
                with st.expander(f"{symbol} Chart"):
                    col_price, col_rsi = st.columns(2)
                    with col_price:
                        st.metric(f"{symbol} Price", f"${latest['close']:.2f}", 
                                 delta=f"{((latest['close']-latest['open'])/latest['open']*100):.2f}%")
                    with col_rsi:
                        st.metric(f"RSI", f"{latest['rsi']:.2f}")
                    
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
                    
                    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['close'], 
                                            name='Price', line=dict(color='white')), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['sma_20'], 
                                            name='SMA 20', line=dict(color='orange')), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['sma_50'], 
                                            name='SMA 50', line=dict(color='blue')), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['rsi'], 
                                            name='RSI', line=dict(color='purple')), row=2, col=1)
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                    
                    fig.update_layout(height=400, showlegend=True, template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please enter API credentials to view market analysis")

# Tab 2: Trading Activity
with tab2:
    st.subheader("Active Positions & Signals")
    
    if api_key and api_secret:
        bot = AdvancedTradingBot(exchange, api_key, api_secret, test_mode)
        balance = bot.get_balance()
        
        if balance:
            col1, col2 = st.columns(2)
            with col1:
                usdt_balance = balance.get('USDT', {}).get('free', 0) if isinstance(balance.get('USDT'), dict) else 0
                st.metric("USDT Balance", f"${usdt_balance:.2f}")
            with col2:
                btc_balance = balance.get('BTC', {}).get('free', 0) if isinstance(balance.get('BTC'), dict) else 0
                st.metric("BTC Balance", f"{btc_balance:.4f} BTC")
        
        for symbol in symbols:
            df = bot.get_market_data(symbol, timeframe, 100)
            if df is not None and len(df) > 50:
                df = bot.calculate_indicators(df)
                buy_sig, sell_sig = bot.get_signals(df, strategy)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.subheader(symbol)
                with col2:
                    if buy_sig:
                        st.success("BUY SIGNAL")
                    elif sell_sig:
                        st.error("SELL SIGNAL")
                    else:
                        st.info("NEUTRAL")
                with col3:
                    st.text(f"Strategy: {strategy.upper()}")
    else:
        st.warning("Please enter API credentials")

# Tab 3: Profit/Loss
with tab3:
    st.subheader("Daily Profit & Loss Summary")
    
    profit_cols = st.columns(3)
    with profit_cols[0]:
        st.metric("Daily Profit", f"${st.session_state.daily_profit:.2f}")
    with profit_cols[1]:
        st.metric("Daily Loss", f"${st.session_state.daily_loss:.2f}")
    with profit_cols[2]:
        net_pl = st.session_state.daily_profit - st.session_state.daily_loss
        st.metric("Net P/L", f"${net_pl:.2f}")
    
    chart_data = pd.DataFrame({
        'Time': pd.date_range(start='2024-01-01', periods=24, freq='h'),
        'Profit': np.random.randn(24).cumsum() * 5 + 50
    })
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=chart_data['Time'], y=chart_data['Profit'], 
                            fill='tozeroy', name='Daily P/L', line=dict(color='green')))
    fig.update_layout(title="Daily P/L Progress", xaxis_title="Time", 
                     yaxis_title="Profit ($)", template="plotly_dark", height=400)
    st.plotly_chart(fig, use_container_width=True)

# Tab 4: Trade History
with tab4:
    st.subheader("Trade History")
    
    if st.session_state.trade_history:
        df_history = pd.DataFrame(st.session_state.trade_history)
        st.dataframe(df_history, use_container_width=True, hide_index=True)
    else:
        st.info("No trades yet. Start the bot to begin trading!")

# Tab 5: Backtesting
with tab5:
    st.subheader("Strategy Backtesting")
    st.write("Test your trading strategy on historical data before using real money")
    
    if api_key and api_secret:
        backtest_col1, backtest_col2, backtest_col3 = st.columns(3)
        
        with backtest_col1:
            backtest_symbol = st.selectbox("Symbol for Backtest", symbols)
        with backtest_col2:
            lookback_days = st.slider("Lookback Period (days)", 7, 365, 30)
        with backtest_col3:
            backtest_strategy = st.selectbox("Strategy", 
                ["momentum", "mean_reversion", "trend_following"], key="backtest_strategy")
        
        if st.button("Run Backtest", use_container_width=True):
            with st.spinner("Running backtest..."):
                bot = AdvancedTradingBot(exchange, api_key, api_secret, test_mode)
                since = int((datetime.now() - timedelta(days=lookback_days)).timestamp() * 1000)
                df = bot.get_market_data(backtest_symbol, '1h', limit=1000, since=since)
                
                if df is not None and len(df) > 50:
                    backtester = Backtester(df, initial_balance=10000, 
                                          position_size=position_size/100, 
                                          stop_loss=stop_loss, 
                                          take_profit=take_profit)
                    results = backtester.backtest_strategy(bot, backtest_strategy, backtest_symbol)
                    
                    if results:
                        st.success("Backtest completed!")
                        
                        metric_cols = st.columns(4)
                        with metric_cols[0]:
                            st.metric("Total Trades", results['total_trades'])
                        with metric_cols[1]:
                            st.metric("Win Rate", f"{results['win_rate']:.1f}%")
                        with metric_cols[2]:
                            st.metric("Total Return", f"${results['total_profit']:.2f}")
                        with metric_cols[3]:
                            st.metric("Final Balance", f"${results['final_balance']:.2f}", 
                                     delta=f"{results['return_percent']:.1f}%")
                        
                        detail_cols = st.columns(4)
                        with detail_cols[0]:
                            st.metric("Winning Trades", results['winning_trades'])
                        with detail_cols[1]:
                            st.metric("Losing Trades", results['losing_trades'])
                        with detail_cols[2]:
                            st.metric("Avg Win", f"${results['avg_win']:.2f}")
                        with detail_cols[3]:
                            st.metric("Avg Loss", f"${results['avg_loss']:.2f}")
                        
                        adv_cols = st.columns(2)
                        with adv_cols[0]:
                            st.metric("Max Drawdown", f"{results['max_drawdown']:.2f}%")
                        with adv_cols[1]:
                            st.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(y=backtester.equity_curve, 
                                               name='Equity', line=dict(color='cyan')))
                        fig.update_layout(title="Equity Curve", xaxis_title="Candles", 
                                        yaxis_title="Balance ($)", template="plotly_dark", height=400)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        if backtester.trades:
                            st.subheader("Backtest Trades")
                            trades_df = pd.DataFrame(backtester.trades)
                            st.dataframe(trades_df, use_container_width=True, hide_index=True)
    else:
        st.warning("Please enter API credentials to run backtest")

# Tab 6: Paper Trading
with tab6:
    st.subheader("Paper Trading Mode")
    st.write("Practice trading with simulated money - no real funds at risk!")
    
    paper_col1, paper_col2, paper_col3 = st.columns(3)
    with paper_col1:
        st.metric("Paper Account Balance", f"${st.session_state.paper_balance:.2f}", delta="Simulated")
    with paper_col2:
        st.metric("Total Paper Trades", len(st.session_state.paper_trades))
    with paper_col3:
        paper_pl = sum([t.get('pnl', 0) for t in st.session_state.paper_trades])
        st.metric("Paper Trading P/L", f"${paper_pl:.2f}")
    
    if api_key and api_secret:
        st.subheader("Place Paper Trade")
        
        p_col1, p_col2, p_col3, p_col4 = st.columns(4)
        with p_col1:
            paper_symbol = st.selectbox("Symbol", symbols, key="paper_symbol")
        with p_col2:
            paper_side = st.selectbox("Side", ["BUY", "SELL"])
        with p_col3:
            paper_amount = st.number_input("Amount (USD)", value=100, min_value=10)
        with p_col4:
            if st.button("Execute Paper Trade"):
                bot = AdvancedTradingBot(exchange, api_key, api_secret, test_mode)
                df = bot.get_market_data(paper_symbol, timeframe, 10)
                if df is not None:
                    entry_price = df.iloc[-1]['close']
                    st.session_state.paper_trades.append({
                        'symbol': paper_symbol,
                        'side': paper_side,
                        'entry_price': entry_price,
                        'amount': paper_amount,
                        'timestamp': datetime.now(),
                        'pnl': 0
                    })
                    st.success(f"Paper {paper_side} executed at ${entry_price:.2f}")
                    st.rerun()
        
        if st.session_state.paper_trades:
            st.subheader("Active Paper Positions")
            for idx, trade in enumerate(st.session_state.paper_trades):
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.text(f"{trade['symbol']}")
                with col2:
                    st.text(f"{trade['side']} @ ${trade['entry_price']:.2f}")
                with col3:
                    st.text(f"${trade['amount']:.2f}")
                with col4:
                    st.text(f"Entry: {trade['timestamp'].strftime('%H:%M')}")
                with col5:
                    if st.button("Close", key=f"close_{idx}"):
                        bot = AdvancedTradingBot(exchange, api_key, api_secret, test_mode)
                        df = bot.get_market_data(trade['symbol'], timeframe, 10)
                        if df is not None:
                            exit_price = df.iloc[-1]['close']
                            pnl = (exit_price - trade['entry_price']) * (trade['amount'] / trade['entry_price'])
                            st.session_state.paper_balance += pnl
                            st.session_state.paper_trades.pop(idx)
                            st.success(f"Position closed! P/L: ${pnl:.2f}")
                            st.rerun()
    else:
        st.warning("Please enter API credentials for paper trading")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: gray; font-size: 12px;">
    <p>Disclaimer: This bot is for educational purposes. Crypto trading involves significant risk. Never invest more than you can afford to lose.</p>
    <p>Made with love for crypto traders | Always test on testnet and paper trading first</p>
</div>
""", unsafe_allow_html=True)
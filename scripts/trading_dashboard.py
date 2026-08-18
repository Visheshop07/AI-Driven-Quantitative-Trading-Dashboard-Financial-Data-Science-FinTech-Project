#!/usr/bin/env python3
"""
AI Quant Trader
"""

import json
import time
import pandas as pd
from pathlib import Path
import argparse

try:
    import yfinance as yf
except ImportError:
    print("Yfinance not installed. Please run: uv add yfinance")
    exit(1)

try:
    import gradio as gr
except ImportError:
    print("Gradio not installed. Please run: uv add gradio")
    exit(1)

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("Plotly not installed. Please run: uv add plotly")
    exit(1)

class TradingDashboard:
    """Professional trading dashboard"""

    def get_live_price(self, symbol):
        try:
          stock = yf.Ticker(symbol)
          #df = stock.history(period="5d")
          df = stock.history(period="1d", interval="1m")
          current_price = float(df["Close"].iloc[-1]) if not df.empty else 0

          if df.empty:
            return None

          return float(df["Close"].iloc[-1])

        except Exception as e:
            print(f"Error fetching price: {e}")
            return None
    
    def get_stock_data(self, symbol):
        import yfinance as yf

        df = yf.download(symbol, period="5d", interval="5m", auto_adjust=False, progress=False)

        # 🔥 FIX: Flatten columns
        if isinstance(df.columns, tuple) or hasattr(df.columns, "levels"):
            df.columns = [col[0] for col in df.columns]

        df.reset_index(inplace=True)

        return df

    def add_indicators(self, df):
        import pandas as pd

        # Moving Averages
        df["MA20"] = df["Close"].rolling(20).mean()
        df["MA50"] = df["Close"].rolling(50).mean()

        # RSI
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()

        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))

        return df

    def create_chart(self, symbol):
        import plotly.graph_objects as go

        try:
            if not symbol:
                symbol = "AAPL"

            symbol = symbol.upper()

            df = self.get_stock_data(symbol)

            # ✅ CHECK 1: Empty data
            if df is None or df.empty:
                fig = go.Figure()
                fig.update_layout(
                    template="plotly_dark",
                    height=600,
                    plot_bgcolor="#0f172a",
                    paper_bgcolor="#0f172a",
                    annotations=[
                        dict(
                            text=f"❌ Share '{symbol}' not available",
                            x=0.5,
                            y=0.5,
                            xref="paper",
                            yref="paper",
                            showarrow=False,
                            font=dict(size=20, color="red")
                        )
                    ]
                )
                return fig

            # ✅ FIX MultiIndex (VERY IMPORTANT)
            if hasattr(df.columns, "levels"):
                df.columns = [col[0] for col in df.columns]

            # ✅ CHECK 2: Required column
            if "Datetime" not in df.columns:
                return go.Figure()

            df = self.add_indicators(df)

            fig = go.Figure()

            # Candlestick
            fig.add_trace(go.Candlestick(
                x=df["Datetime"],
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                increasing=dict(line=dict(color="#00ff88")),
                decreasing=dict(line=dict(color="#ff4d4d"))
            ))

            # Moving averages
            fig.add_trace(go.Scatter(
                x=df["Datetime"], y=df["MA20"],
                line=dict(color="#FFD700", width=2),
                name="MA20"
            ))

            fig.add_trace(go.Scatter(
                x=df["Datetime"], y=df["MA50"],
                line=dict(color="#00BFFF", width=2),
                name="MA50"
            ))

            # Volume
            if "Volume" in df.columns:
                fig.add_trace(go.Bar(
                    x=df["Datetime"],
                    y=df["Volume"],
                    marker_color="rgba(100,149,237,0.3)",
                    yaxis="y2",
                    name="Volume"
                ))

            fig.update_layout(
                title=f"📈 {symbol} Live Trading Chart",
                template="plotly_dark",
                height=600,

                plot_bgcolor="#0f172a",
                paper_bgcolor="#0f172a",
                font=dict(color="white"),

                xaxis=dict(
                    showgrid=True,
                    gridcolor="rgba(255,255,255,0.05)"
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor="rgba(255,255,255,0.05)"
                ),

                xaxis_rangeslider_visible=False,
                hovermode="x unified",

                yaxis2=dict(
                    overlaying='y',
                    side='right',
                    showgrid=False,
                    title="Volume"
                )
            )

            return fig

        except Exception as e:
            print("ERROR:", e)

            fig = go.Figure()
            fig.update_layout(
                template="plotly_dark",
                height=600,
                annotations=[
                    dict(
                        text=f"⚠️ Error loading '{symbol}'",
                        x=0.5,
                        y=0.5,
                        showarrow=False,
                        font=dict(size=20, color="orange")
                    )
                ]
            )
            return fig

    def __init__(self, results_dir="data/results"):
        import threading
        self.algo_engine_lock = threading.Lock()
        self.portfolio_lock = threading.Lock()
        self.results_dir = Path(results_dir)
        self.portfolio_file = Path("data/portfolio.json")
        self.initialize_algo_engine()
        self.max_orders_per_cycle = 2
        self.max_orders_per_session = 10
        self.orders_this_session = 0
        self.trade_cooldown_minutes = 30
        self.last_trade_time = {}
        

    def load_latest_results(self):
        """Load the latest analysis results"""
        try:
            step_files = {
                'step1': self._find_latest_file('step1_fetch_*_raw_data.json'),
                'step2': self._find_latest_file('step2_process_*_processed_data.json'),
                'step3': self._find_latest_file('step3_features_*_engineered_features.json'),
                'step4': self._find_latest_file('step4_predictions_*_llm_predictions.json'),
                'step5': self._find_latest_file('step5_signals_*_trading_signals.json'),
                'step6': self._find_latest_file('step6_orders_*_portfolio_orders.json')
            }

            results = {}
            for step, file_path in step_files.items():
                if file_path and file_path.exists():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        results[step] = json.load(f)
                else:
                    results[step] = None

            return results
        except Exception as e:
            print(f"Error loading results: {e}")
            return {}

    def _find_latest_file(self, pattern):
       """Find the latest matching file"""
       files = list(self.results_dir.glob(pattern))
       if files:
          return max(files, key=lambda x: x.stat().st_mtime)
       return None

    def load_portfolio_data(self):
        """Load portfolio data"""
        try:
            if self.portfolio_file.exists():
                 with open(self.portfolio_file, 'r', encoding='utf-8') as f:
                   return json.load(f)

            # 🔥 If file does not exist → create default
            return {"cash": 100000, "positions": {}}

        except Exception as e:
            print(f"Error loading portfolio: {e}")

            # 🔥 Even if error → return safe default
            return {"cash": 100000, "positions": {}}
        
    def save_portfolio_data(self, data):
        """Save portfolio data safely (atomic write)"""
        with self.portfolio_lock:

            try:
                temp_file = self.portfolio_file.with_suffix(".tmp")

                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4)

                temp_file.replace(self.portfolio_file)

            except Exception as e:
                print(f"Error saving portfolio: {e}")

    def add_realized_pnl(self, portfolio, pnl):
        portfolio["realized_pnl"] = (
            portfolio.get("realized_pnl", 0) + pnl
        )            

    def load_orders(self):
        try:
            if Path("data/orders.json").exists():
                with open("data/orders.json", "r") as f:
                  return json.load(f)
            return []
        except:
            return []

    def save_orders(self, orders):
        with open("data/orders.json", "w") as f:
            json.dump(orders, f, indent=4)  

    def create_orders_dashboard(self):
        orders = self.load_orders()

        if not orders:
           return None, "No orders yet", "📭 No trades executed"

        df = pd.DataFrame(orders)

        total_trades = len(df)
        buy_trades = len(df[df["type"].isin(["BUY", "ALGO BUY"])])
        sell_trades = len(df[df["type"].isin(["SELL", "ALGO SELL"])])

        stats = f"""
        📊 Total Trades: {total_trades}  
        🟢 Buy Orders: {buy_trades}  
        🔴 Sell Orders: {sell_trades}
       """

        summary = f"📈 Last Trade: {df.iloc[-1]['symbol']} ({df.iloc[-1]['type']})"

        return df, stats, summary               
    
    def calculate_change(self, df, period):
        """Calculate percentage change over a period"""
        try:
            if len(df) <= period:
               return "N/A"

            current = df["Close"].iloc[-1]
            past = df["Close"].iloc[-(period+1)]

            change = ((current - past) / past) * 100
            return round(change, 2)
        except:
            return "N/A"

    def create_market_overview(self, results, symbol=None, volume_filter="all"):
        """Create real-time market overview using yfinance"""

        # 🔥 Trending stocks (default)
        TRENDING = ["AAPL", "MSFT", "TSLA", "NVDA", "GOOGL",
    "META", "AMZN", "NFLX", "AMD", "INTC"]
        if symbol and symbol.strip() != "":
            # Support multiple stocks
            symbols = [s.strip().upper() for s in symbol.split(",")]
        else:
         symbols = TRENDING

        overview_data = []

        for sym in symbols:
            try:
               stock = yf.Ticker(sym)
               df = stock.history(period="1mo", interval="1h")

               if df.empty:
                continue

               volume = int(df["Volume"].iloc[-1])

               # 🔥 Volume filter
               if volume_filter == "high" and volume < 10000000:
                continue


               currency = stock.info.get("currency", "")
                
               if currency == "INR":
                     price = f"₹{df['Close'].iloc[-1]:,.2f}"
               elif currency == "USD":
                      price = f"${df['Close'].iloc[-1]:,.2f}"
               else:
                price = f"{df['Close'].iloc[-1]:,.2f}"
               overview_data.append({
                  "Symbol": sym,
                  "Price": price,
                  "1H %": f"{self.calculate_change(df, 1)}%",
                  "4H %": f"{self.calculate_change(df, 4)}%",
                  "24H %": f"{self.calculate_change(df, 24)}%",
                  "1W %": f"{self.calculate_change(df, 24*5)}%",
                  "Volume": f"{volume:,}"
                })

            except Exception as e:
               print(f"Error fetching {sym}: {e}")

        df = pd.DataFrame(overview_data)
        return df, f"📊 Market Overview - {len(df)} stocks (Live)"

    def create_price_chart(self, symbol, results):
        """Create price chart"""
        if not results.get('step3') or symbol not in results['step3']:
            return None

        data = results['step3'][symbol]['data']
        df = pd.DataFrame(data)

        if df.empty:
            return None

        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(f'{symbol} Price Trend', 'Volume', 'Technical Indicators'),
            row_heights=[0.5, 0.2, 0.3]
        )

        # Price candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=["Datetime"],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price',
                increasing_line_color='#00ff88',
                decreasing_line_color='#ff4444'
            ),
            row=1, col=1
        )

        # 🔥 FIXED VOLUME
        if "Volume" in df.columns:
            fig.add_trace(
                go.Bar(
                    x=df["Datetime"],
                    y=df["Volume"],
                    name="Volume",
                    marker_color="lightblue",
                    opacity=0.7
                )
            )

        # Moving averages
        if 'ma_20' in df.columns:
            fig.add_trace(
                go.Scatter(x=["Datetime"], y=df['ma_20'], name='MA20',
                          line=dict(color='orange', width=2)),
                row=1, col=1
            )
        if 'ma_50' in df.columns:
            fig.add_trace(
                go.Scatter(x=["Datetime"], y=df['ma_50'], name='MA50',
                          line=dict(color='blue', width=2)),
                row=1, col=1
            )


        # RSI indicator
        if 'rsi' in df.columns:
            fig.add_trace(
                go.Scatter(x=["Datetime"], y=df['rsi'], name='RSI',
                          line=dict(color='purple', width=2)),
                row=3, col=1
            )
            # RSI overbought/oversold lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

        fig.update_layout(
            title=f'{symbol} Technical Analysis',
            xaxis_rangeslider_visible=False,
            height=800,
            showlegend=True,
            template='plotly_dark'
        )

        return fig

    def create_ai_analysis(self, symbols):

        import yfinance as yf
        import pandas as pd
        import numpy as np
        import plotly.graph_objects as go
        from sklearn.linear_model import LinearRegression

        symbols = [s.strip().upper() for s in symbols.split(",") if s.strip()]

        def get_trend(df, threshold=0.03):
            if df.empty or "Close" not in df.columns:
                return "No Data", 0

            start = float(df["Close"].iloc[0])
            end = float(df["Close"].iloc[-1])

            change = (end - start) / start

            if change > threshold:
                return "🟢 Bullish", change
            elif change < -threshold:
                return "🔴 Bearish", change
            else:
                return "⚪ Sideways", change
            
        def get_projection_percent(df, future_periods=10):
            import numpy as np
            from sklearn.linear_model import LinearRegression

            if df.empty or "Close" not in df.columns:
                return 0

            df = df.dropna().copy()

            X = np.arange(len(df)).reshape(-1, 1)
            y = df["Close"].values

            model = LinearRegression()
            model.fit(X, y)

            future_X = np.arange(len(df), len(df) + future_periods).reshape(-1, 1)
            future_pred = model.predict(future_X)

            current_price = float(df["Close"].iloc[-1])
            projected_price = float(future_pred[-1])

            return (projected_price - current_price) / current_price    


        def create_projection_chart(df, title, future_periods=10, freq="B"):
            if df.empty or "Close" not in df.columns:
                return go.Figure()

            df = df.dropna().copy()

            X = np.arange(len(df)).reshape(-1, 1)
            y = df["Close"].values

            model = LinearRegression()
            model.fit(X, y)

            future_X = np.arange(len(df), len(df) + future_periods).reshape(-1, 1)
            future_pred = model.predict(future_X)

            future_dates = pd.date_range(
                start=df.index[-1],
                periods=future_periods + 1,
                freq=freq
            )[1:]

            fig = go.Figure()

            # Historical Price
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df["Close"],
                mode="lines",
                name="Historical Price",
                line=dict(color="cyan", width=2)
            ))

            # Projection
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=future_pred,
                mode="lines",
                name="Trend Projection",
                line=dict(color="orange", width=2, dash="dash")
            ))

            fig.update_layout(
                title=title,
                template="plotly_dark",
                height=320,
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(orientation="h")
            )

            return fig


        analysis_data = []

        intraday_fig = go.Figure()
        short_fig = go.Figure()
        long_fig = go.Figure()

        for symbol in symbols:
            try:
                df_intraday = yf.download(
                    symbol,
                    period="5d",
                    interval="15m",
                    progress=False,
                    auto_adjust=True
                )

                df_short = yf.download(
                    symbol,
                    period="3mo",
                    interval="1d",
                    progress=False,
                    auto_adjust=True
                )

                df_long = yf.download(
                    symbol,
                    period="1y",
                    interval="1d",
                    progress=False,
                    auto_adjust=True
                )

                if (
                    df_intraday.empty or
                    df_short.empty or
                    df_long.empty
                ):
                    print(f"Skipping invalid/no-data symbol: {symbol}")
                    continue

                # Fix MultiIndex if present
                for df in [df_intraday, df_short, df_long]:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)

                intraday_score = get_projection_percent(df_intraday, 8)
                short_score = get_projection_percent(df_short, 10)
                long_score = get_projection_percent(df_long, 30)

                intraday_signal = "🟢 Bullish" if intraday_score > 0.01 else "🔴 Bearish" if intraday_score < -0.01 else "⚪ Sideways"

                short_signal = "🟢 Bullish" if short_score > 0.03 else "🔴 Bearish" if short_score < -0.03 else "⚪ Sideways"

                long_signal = "🟢 Bullish" if long_score > 0.08 else "🔴 Bearish" if long_score < -0.08 else "⚪ Sideways"

                overall_score = round(
                    (intraday_score * 30) +
                    (short_score * 30) +
                    (long_score * 40),
                    2
                )

                analysis_data.append({
                    "Stock": symbol,
                    "Intraday": intraday_signal,
                    "Intraday Projection %": f"{intraday_score:.2%}",
                    "Short Term": short_signal,
                    "Swing Projection %": f"{short_score:.2%}",
                    "Long Term": long_signal,
                    "Investment Projection %": f"{long_score:.2%}",
                    "Score": overall_score
                })

                # Use first stock for charts (clean UI)
                if symbol == symbols[0]:
                    intraday_fig = create_projection_chart(
                        df_intraday,
                        f"{symbol} Intraday Projection",
                        future_periods=8,
                        freq="15min"
                    )

                    short_fig = create_projection_chart(
                        df_short,
                        f"{symbol} Short-Term Projection",
                        future_periods=10,
                        freq="B"
                    )

                    long_fig = create_projection_chart(
                        df_long,
                        f"{symbol} Long-Term Projection",
                        future_periods=30,
                        freq="B"
                    )

            except Exception as e:
                print(f"Error analyzing {symbol}: {e}")


        df_analysis = pd.DataFrame(analysis_data)

        if not df_analysis.empty:

            intraday_df = df_analysis[["Stock", "Intraday", "Intraday Projection %"]]
            short_df = df_analysis[["Stock", "Short Term", "Swing Projection %"]]
            long_df = df_analysis[["Stock", "Long Term", "Investment Projection %"]]

            # SINGLE STOCK SUMMARY
            if len(df_analysis) == 1:

                if df_short.empty or "Close" not in df_short.columns:
                    current_price = "N/A"
                else:
                    current_price = float(df_short["Close"].iloc[-1])

                volatility = df_short["Close"].pct_change().std() * 100

                if volatility < 1:
                    risk_level = "🟢 Low"
                elif volatility < 2.5:
                    risk_level = "🟡 Medium"
                else:
                    risk_level = "🔴 High"

                bullish_count = sum([
                    intraday_signal == "🟢 Bullish",
                    short_signal == "🟢 Bullish",
                    long_signal == "🟢 Bullish"
                ])

                bearish_count = sum([
                    intraday_signal == "🔴 Bearish",
                    short_signal == "🔴 Bearish",
                    long_signal == "🔴 Bearish"
                ])

                confidence = round(
                    max(bullish_count, bearish_count) / 3 * 100,
                    2
                )

                summary = f"""

        - **Stock:** {df_analysis.iloc[0]['Stock']}
        - **AI Score:** {df_analysis.iloc[0]['Score']:.2f}
        - **Current Price:** ${current_price:.2f}
        - **Risk Level:** {risk_level}
        - **Confidence:** {confidence:.0f}%

        > Projection charts show trend-based model estimates, not guaranteed future prices.
        """

            # MULTIPLE STOCK SUMMARY
            else:

                summary = f"""



        ### ✅ AI Analysis Complete

        - **Stocks Analyzed:** {len(df_analysis)}
        - **Average Score:** {df_analysis['Score'].mean():.2f}
        - **Top Rated Stock:** {df_analysis.loc[df_analysis['Score'].idxmax(), 'Stock']}

        > Projection charts show trend-based model estimates, not guaranteed future prices.
        """

            return (
                intraday_df,
                intraday_fig,
                short_df,
                short_fig,
                long_df,
                long_fig,
                summary
            )

        else:
            empty_df = pd.DataFrame({"Error": ["No valid stock data"]})
            empty_fig = go.Figure()

            return (
                empty_df,
                empty_fig,
                empty_df,
                empty_fig,
                empty_df,
                empty_fig,
                "❌ No Data Found"
            )

            
        
        

    def get_market_news(self, query=""):

        import feedparser
        from datetime import datetime
        from urllib.parse import quote

        try:
            # Default live market news if search empty
            if not query or str(query).strip() == "":
                query = "Global Stock Market 📉📈"

            encoded_query = quote(f"{query} stock")

            url = (
                f"https://news.google.com/rss/search?"
                f"q={encoded_query}"
                f"&hl=en-IN&gl=IN&ceid=IN:en"
            )

            feed = feedparser.parse(url)

            news_html = ""

            positive_words = ["gain", "rise", "bull", "growth", "profit", "surge"]
            negative_words = ["fall", "loss", "crash", "drop", "decline", "slump"]

            if not feed.entries:
                return (
                    "<p style='color:orange;'>No news found.</p>",
                    "⚠ No Results"
                )

            for entry in feed.entries[:10]:

                title = entry.title
                link = entry.link
                source = entry.get("source", {}).get("title", "Google News")

                sentiment = "⚪ Neutral"
                title_lower = title.lower()

                if any(w in title_lower for w in positive_words):
                    sentiment = "🟢 Bullish"
                elif any(w in title_lower for w in negative_words):
                    sentiment = "🔴 Bearish"

                news_html += f"""
                <div style="
                    padding:12px;
                    margin-bottom:10px;
                    border-radius:10px;
                    background:#111827;
                    border:1px solid #1f2937;
                ">
                    <div style="font-size:12px;">{sentiment}</div>

                    <a href="{link}" target="_blank"
                    style="color:#60a5fa; font-weight:500; text-decoration:none;">
                        {title}
                    </a>

                    <div style="color:#9ca3af; font-size:12px; margin-top:5px;">
                        {source}
                    </div>
                </div>
                """

            time = datetime.now().strftime("%H:%M:%S")

            return news_html, f"📰 Results for '{query}' at {time}"

        except Exception as e:
            return f"<p style='color:red;'>Error: {str(e)}</p>", "❌ Failed"
        
    def initialize_algo_engine(self):

        self.algo_running = False
        self.max_qty_per_stock = 10

        portfolio_data = self.load_portfolio_data()

        self.portfolio_cash = portfolio_data.get("cash", 0)

        # Restore Algo Positions From portfolio.json
        self.algo_positions = []

        for stock, pos in portfolio_data.get("algo_positions", {}).items():
            self.algo_positions.append({
                "Stock": stock,
                "Entry Price": pos["entry_price"],
                "Qty": pos["quantity"],
                "Target": pos["target"],
                "Stop Loss": pos["stop_loss"],
                "Signal Strength": pos.get("signal_strength", 0)
            })

        self.algo_trade_history = []

        self.algo_pnl = 0

        print("Restored Algo Positions:", self.algo_positions)

    def start_algo(self, strategy_code, selected_market):

        self.orders_this_session = 0

        self.save_strategy(strategy_code)

        self.selected_market = selected_market

        portfolio_data = self.load_portfolio_data()

        self.portfolio_cash = portfolio_data.get("cash", 0)

        self.current_strategy_code = strategy_code
        self.algo_running = True

        return f"🟢 Algo Started ({selected_market})"

    def stop_algo(self):
        self.algo_running = False
        return "🔴 Algo Stopped"  

    def execute_algo_trade(self, stock, price, signal_strength):

        print(f"\nTrying Trade -> {stock} | Price={price} | Strength={signal_strength}")

        allocation_pct = max(
            min(signal_strength * 0.1, 0.10),
            0.05
        )

        qty = min(
            int((self.portfolio_cash * allocation_pct) / price),
            self.max_qty_per_stock
        )

        print("Portfolio Cash:", self.portfolio_cash)
        print("Calculated Qty:", qty)

        if qty <= 0:
            print("Skipped: Qty <= 0")
            return False

        invested = qty * price

        if invested > self.portfolio_cash:
            print("Skipped: Not Enough Cash")
            return

        existing_pos = next(
            (pos for pos in self.algo_positions if pos["Stock"] == stock),
            None
        )

        portfolio = self.load_portfolio_data()
        portfolio.setdefault("algo_positions", {})

        # ==============================
        # ADD TO EXISTING POSITION
        # ==============================
        if existing_pos:

            total_qty = existing_pos["Qty"] + qty

            new_avg = (
                (existing_pos["Entry Price"] * existing_pos["Qty"]) +
                (price * qty)
            ) / total_qty

            existing_pos["Qty"] = total_qty
            existing_pos["Entry Price"] = new_avg

            existing_pos["Target"] = round(new_avg * 1.01, 2)
            existing_pos["Stop Loss"] = round(new_avg * 0.995, 2)

            self.portfolio_cash -= invested

            portfolio["algo_positions"][stock] = {
                "quantity": total_qty,
                "entry_price": new_avg,
                "target": existing_pos["Target"],
                "stop_loss": existing_pos["Stop Loss"],
                "signal_strength": signal_strength
            }

            portfolio["cash"] = self.portfolio_cash

            self.save_portfolio_data(portfolio)

            orders = self.load_orders()
            orders.append({
                "type": "ALGO BUY",
                "symbol": stock,
                "qty": qty,
                "price": price,
                "time": str(pd.Timestamp.now())
            })
            self.save_orders(orders)

            print("Added Qty To Existing Position")
            return True

        # ==============================
        # NEW POSITION CHECK
        # ==============================
        # if len(self.algo_positions) >= self.max_open_positions:
        #     print("Skipped: Max Open Positions Reached")
        #     return

        if self.orders_this_session >= self.max_orders_per_session:
            print("Skipped: Session Order Limit Reached")
            return False
        
        now = pd.Timestamp.now()

        last_trade = self.last_trade_time.get(stock)

        if last_trade:

            minutes_since_trade = (
                now - last_trade
            ).total_seconds() / 60

            if minutes_since_trade < self.trade_cooldown_minutes:
                print(f"Skipped: Cooldown Active For {stock}")
                return False

        self.portfolio_cash -= invested

        orders = self.load_orders()
        orders.append({
            "type": "ALGO BUY",
            "symbol": stock,
            "qty": qty,
            "price": price,
            "time": str(pd.Timestamp.now())
        })
        self.save_orders(orders)

        new_position = {
            "Stock": stock,
            "Entry Price": price,
            "Qty": qty,
            "Target": round(price * 1.01, 2),
            "Stop Loss": round(price * 0.995, 2),
            "Signal Strength": signal_strength
        }

        self.algo_positions.append(new_position)

        portfolio["algo_positions"][stock] = {
            "quantity": qty,
            "entry_price": price,
            "target": new_position["Target"],
            "stop_loss": new_position["Stop Loss"],
            "signal_strength": signal_strength
        }

        portfolio["cash"] = self.portfolio_cash

        self.save_portfolio_data(portfolio)

        self.last_trade_time[stock] = pd.Timestamp.now()

        self.orders_this_session += 1

        print("TRADE EXECUTED SUCCESSFULLY")

        return True


    def update_algo_positions(self):

        closed_positions = []

        portfolio = self.load_portfolio_data()

        for pos in self.algo_positions.copy():

            current_price = self.get_live_price(pos["Stock"])

            if current_price is None:
                continue

            if current_price >= pos["Target"]:
                reason = "Target Hit"

            elif current_price <= pos["Stop Loss"]:
                reason = "Stop Loss Hit"

            else:
                continue

            pnl = (current_price - pos["Entry Price"]) * pos["Qty"]
            self.add_realized_pnl(portfolio, pnl)

            self.portfolio_cash += current_price * pos["Qty"]

            portfolio["cash"] = self.portfolio_cash

            if pos["Stock"] in portfolio.get("algo_positions", {}):
                del portfolio["algo_positions"][pos["Stock"]]

            orders = self.load_orders()

            orders.append({
                "type": "ALGO SELL",
                "symbol": pos["Stock"],
                "qty": pos["Qty"],
                "price": current_price,
                "pnl": pnl,
                "time": str(pd.Timestamp.now())
            })

            self.save_orders(orders)

            self.algo_pnl += pnl

            self.algo_trade_history.append({
                **pos,
                "Exit Price": current_price,
                "PnL": pnl,
                "Exit Reason": reason
            })

            closed_positions.append(pos["Stock"])

        # Safe Removal By Symbol
        self.algo_positions = [
            pos for pos in self.algo_positions
            if pos["Stock"] not in closed_positions
        ]

        self.save_portfolio_data(portfolio)
            
    def run_algo_engine(self):

        import yfinance as yf
        import pandas as pd
        import numpy as np
        import time

        if not self.algo_running:
            return

        orders_this_cycle = 0

        stock_universe = self.load_market_universe(self.selected_market)

        if not stock_universe:
            print("No Stocks In Selected Universe")
            return

        for stock in stock_universe:

            if orders_this_cycle >= self.max_orders_per_cycle:
                print("Cycle Order Limit Reached")
                break

            if any(
                p["Stock"].strip().upper() == stock.strip().upper()
                for p in self.algo_positions
            ):
                continue

            if not self.algo_engine_lock.acquire(blocking=False):
                print("Algo Engine Already Running — Skipping Duplicate Trigger")
                return

            try:
                df = yf.download(
                    stock,
                    period="30d",
                    interval="5m",
                    auto_adjust=False,
                    progress=False
                )

                if df.empty or len(df) < 50:
                    continue

                # Flatten MultiIndex if needed
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                close = df["Close"].astype(float)
                high = df["High"].astype(float)
                low = df["Low"].astype(float)
                open_ = df["Open"].astype(float)
                volume = df["Volume"].astype(float)

                # -----------------------------
                # Dynamic Indicator Functions
                # -----------------------------

                def ema(period):
                    return close.ewm(span=period).mean().iloc[-1]

                def sma(period):
                    return close.rolling(period).mean().iloc[-1]

                def rsi(period=14):
                    delta = close.diff()

                    gain = delta.clip(lower=0).rolling(period).mean()
                    loss = (-delta.clip(upper=0)).rolling(period).mean()

                    rs = gain / loss
                    return (100 - (100 / (1 + rs))).iloc[-1]

                def macd():
                    fast = close.ewm(span=12).mean()
                    slow = close.ewm(span=26).mean()
                    return (fast - slow).iloc[-1]

                def macd_signal():
                    fast = close.ewm(span=12).mean()
                    slow = close.ewm(span=26).mean()
                    macd_line = fast - slow
                    return macd_line.ewm(span=9).mean().iloc[-1]

                def atr(period=14):
                    tr1 = high - low
                    tr2 = abs(high - close.shift())
                    tr3 = abs(low - close.shift())

                    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

                    return tr.rolling(period).mean().iloc[-1]

                def volume_sma(period):
                    return volume.rolling(period).mean().iloc[-1]

                # -----------------------------
                # Strategy Scope
                # -----------------------------

                strategy_scope = {
                    "close": close.iloc[-1],
                    "open": open_.iloc[-1],
                    "high": high.iloc[-1],
                    "low": low.iloc[-1],
                    "volume": volume.iloc[-1],

                    "ema": ema,
                    "sma": sma,
                    "rsi": rsi,
                    "macd": macd,
                    "macd_signal": macd_signal,
                    "atr": atr,
                    "volume_sma": volume_sma,

                    "np": np,
                    "pd": pd
                }

                try:
                    safe_globals = {"__builtins__": {}}

                    exec(
                        self.current_strategy_code,
                        safe_globals,
                        strategy_scope
                    )

                    buy_signal = bool(strategy_scope.get("buy", False))

                except Exception as e:
                    print(f"Strategy Error ({stock}): {e}")
                    continue

                if buy_signal:

                    strength = min(
                        abs(ema(9) - ema(21)) / max(ema(21), 1) * 100,
                        1.0
                    )

                    trade_success = self.execute_algo_trade(
                        stock=stock,
                        price=close.iloc[-1],
                        signal_strength=strength
                    )

                    if trade_success:
                        orders_this_cycle += 1

                time.sleep(0.3)

                self.algo_engine_lock.release() 

            except Exception as e:
                print(f"Algo Scan Error {stock}: {e}")   

            


    def load_market_universe(self, market_name):

        market_files = {
            "India": "data/universes/india_nifty50.csv",
            "USA": "data/universes/usa_sp500.csv",
            "UK": "data/universes/uk_ftse100.csv",
            "Japan": "data/universes/japan_nikkei225.csv",
            "Germany": "data/universes/germany_dax.csv",
            "France": "data/universes/france_cac40.csv",
            "China": "data/universes/china_sse50.csv",
            "Hong Kong": "data/universes/hongkong_hsi.csv",
            "Canada": "data/universes/canada_tsx60.csv",
            "Australia": "data/universes/australia_asx200.csv"
        }

        file_path = market_files.get(market_name)

        if not file_path:
            return []

        df = pd.read_csv(file_path)

        return df["Symbol"].dropna().tolist()            



    def refresh_algo_dashboard(self):

        if self.algo_running:
            self.run_algo_engine()

        self.update_algo_positions()

        positions_df = pd.DataFrame(self.algo_positions)
        orders = self.load_orders()

        algo_orders = [
            o for o in orders
            if o.get("type") in ["ALGO BUY", "ALGO SELL"]
        ]

        history_df = pd.DataFrame(algo_orders)

        pnl_text = f"$ {self.algo_pnl:.2f}"

        return (
                    "🟢 Algo Running" if self.algo_running else "🔴 Algo Stopped",
                    positions_df,
                    gr.update(
                        choices=self.get_algo_position_symbols(),
                        value=None
                    ),
                    history_df,
                    pnl_text
                )     


    def save_strategy(self, code):
        with open("data/strategy.py", "w") as f:
            f.write(code)

    def load_strategy(self):
        path = Path("data/strategy.py")

        if path.exists():
            return path.read_text()

        return "buy = ema_9 > ema_21 and rsi < 60"    


    def partial_close_algo_position(self, stock, qty_to_sell):

        portfolio = self.load_portfolio_data()



        if stock not in portfolio.get("algo_positions", {}):
            return "Position Not Found"

        pos = portfolio["algo_positions"][stock]

        current_qty = pos["quantity"]

        if qty_to_sell <= 0:
            return "Invalid Quantity"

        if qty_to_sell > current_qty:
            return "Cannot Sell More Than Held Qty"

        live_price = self.get_live_price(stock)

        proceeds = qty_to_sell * live_price

        self.portfolio_cash += proceeds

        remaining_qty = current_qty - qty_to_sell

        pnl = (live_price - pos["entry_price"]) * qty_to_sell

        self.add_realized_pnl(portfolio, pnl)

        if remaining_qty == 0:
            del portfolio["algo_positions"][stock]

            self.algo_positions = [
                p for p in self.algo_positions
                if p["Stock"] != stock
            ]

        else:
            pos["quantity"] = remaining_qty

            for runtime_pos in self.algo_positions:
                if runtime_pos["Stock"] == stock:
                    runtime_pos["Qty"] = remaining_qty
                    break

        portfolio["cash"] = self.portfolio_cash
       
        self.save_portfolio_data(portfolio)

        orders = self.load_orders()

        orders.append({
            "type": "ALGO SELL",
            "symbol": stock,
            "qty": qty_to_sell,
            "price": live_price,
            "pnl": pnl,
            "time": str(pd.Timestamp.now()),
            "reason": "Partial Exit"
        })

        self.save_orders(orders)

        return f"Sold {qty_to_sell} Qty of {stock}"      


    def close_all_algo_positions(self):

        portfolio = self.load_portfolio_data()

        algo_positions = portfolio.get("algo_positions", {})

        if not algo_positions:
            return "No Algo Positions To Close"

        for stock, pos in list(algo_positions.items()):

            live_price = self.get_live_price(stock)

            if live_price is None:
                continue

            qty = pos["quantity"]
            entry_price = pos["entry_price"]

            proceeds = qty * live_price

            pnl = (live_price - entry_price) * qty

            self.portfolio_cash += proceeds

            self.add_realized_pnl(portfolio, pnl)

            orders = self.load_orders()

            orders.append({
                "type": "ALGO SELL",
                "symbol": stock,
                "qty": qty,
                "price": live_price,
                "pnl": pnl,
                "time": str(pd.Timestamp.now()),
                "reason": "Close All"
            })

            self.save_orders(orders)

        portfolio["algo_positions"] = {}

        portfolio["cash"] = self.portfolio_cash

        self.algo_positions = []

        self.save_portfolio_data(portfolio)

        return "All Algo Positions Closed Successfully" 
        

    def get_algo_position_symbols(self):
      return [pos["Stock"] for pos in self.algo_positions]


    def create_portfolio_dashboard(self, portfolio_data):

        if not portfolio_data:
            return None, None, None, "No portfolio data"

        cash = portfolio_data.get("cash", 0)
        positions = portfolio_data.get("positions", {})
        algo_positions = portfolio_data.get("algo_positions", {})
        realized_pnl = portfolio_data.get("realized_pnl", 0)

        # Merge Manual + Algo Positions
        all_positions = {}

        for symbol, pos in positions.items():
            all_positions[symbol] = {
                "quantity": pos.get("quantity", 0),
                "avg_price": pos.get("avg_price", 0),
                "type": "Manual"
            }

        for symbol, pos in algo_positions.items():

            if symbol in all_positions:
                existing = all_positions[symbol]

                total_qty = existing["quantity"] + pos["quantity"]

                avg_price = (
                    existing["avg_price"] * existing["quantity"] +
                    pos["entry_price"] * pos["quantity"]
                ) / total_qty

                all_positions[symbol] = {
                    "quantity": total_qty,
                    "avg_price": avg_price,
                    "type": "Manual + Algo"
                }

            else:
                all_positions[symbol] = {
                    "quantity": pos["quantity"],
                    "avg_price": pos["entry_price"],
                    "type": "Algo"
                }

        total_stock_value = 0
        total_pnl = 0
        position_values = []

        import yfinance as yf

        for symbol, pos in all_positions.items():

            quantity = pos["quantity"]
            avg_price = pos["avg_price"]

            stock = yf.Ticker(symbol)
            df = stock.history(period="1d")

            current_price = float(df["Close"].iloc[-1]) if not df.empty else 0

            market_value = quantity * current_price
            pnl = (current_price - avg_price) * quantity

            total_stock_value += market_value
            total_pnl += pnl

            position_values.append({
                "Type": pos["type"],
                "Stock": symbol,
                "Shares": quantity,
                "Current Price": f"${current_price:.2f}",
                "Market Value": f"${market_value:,.2f}",
                "P&L": f"${pnl:,.2f}"
            })

        df_positions = pd.DataFrame(position_values)

        total_value = cash + total_stock_value

        # Asset Allocation Pie Chart
        if total_value > 0:

            fig_pie = go.Figure(data=[go.Pie(
                labels=["Cash", "Stocks", "P&L"],
                values=[cash, total_stock_value, total_pnl],
                hole=0.3,
                textinfo="label+percent+value",
                texttemplate="%{label}<br>%{percent}<br>$%{value:,.0f}"
            )])

            fig_pie.update_layout(
                title="Asset Allocation",
                height=400,
                template="plotly_dark"
            )

        else:
            fig_pie = None

        # Portfolio Overview
        overview_data = {
            "Metric": [
                "Total Assets",
                "Cash",
                "Stock Value",
                "Total PNL",
                "Position Count"
            ],
            "Value": [
                f"${total_value:,.2f}",
                f"${cash:,.2f}",
                f"${total_stock_value:,.2f}",
                f"${total_pnl:,.2f}",
                len(all_positions)
            ]
        }

        df_overview = pd.DataFrame(overview_data)

        summary_text = f"""
    💼 Portfolio Value: ${total_value:,.2f}
    📈 Unrealized P&L: ${total_pnl:,.2f}
    💰 Realized P&L: ${realized_pnl:,.2f}
    📦 Total Positions: {len(all_positions)}
    """

        return df_overview, df_positions, fig_pie, summary_text

    def save_portfolio_cash(self):

        portfolio_data = self.load_portfolio_data()

        if not portfolio_data:
            portfolio_data = {}

        portfolio_data["cash"] = self.portfolio_cash

        with open(self.portfolio_file, "w", encoding="utf-8") as f:
            json.dump(portfolio_data, f, indent=4)

    def create_orders_panel(self, results):
        """Create orders panel"""
        if not results.get('step6'):
            signals = results.get('step5', {})

            orders = []

            for symbol, signal in signals.items():
                action = signal.get("signal")

                if action == "BUY":
                    orders.append({
                        "symbol": symbol,
                        "action": "BUY",
                        "quantity": 2,
                        "estimated_price": 100,
                        "estimated_cost": 200,
                        "reason": "Signal-based BUY"
                    })

                elif action == "SELL":
                    orders.append({
                        "symbol": symbol,
                        "action": "SELL",
                        "quantity": 2,
                        "estimated_price": 100,
                        "estimated_proceeds": 200,
                        "reason": "Signal-based SELL"
                    })

            if not orders:
                return None, None, "No orders generated"

            # Order summary
            order_data = []
            total_cost = 0
            total_proceeds = 0

            for order in orders:
                symbol = order.get('symbol', '')
                action = order.get('action', '')
                quantity = order.get('quantity', 0)
                price = order.get('estimated_price', 0)
                reason = order.get('reason', '')

                if action == 'BUY':
                    cost = order.get('estimated_cost', 0)
                    total_cost += cost
                    value_str = f"${cost:,.2f}"
                    icon = "🟢 Buy"
                else:
                    proceeds = order.get('estimated_proceeds', 0)
                    total_proceeds += proceeds
                    value_str = f"${proceeds:,.2f}"
                    icon = "🔴 Sell"

                order_data.append({
                    'Stock': symbol,
                    'Action': icon,
                    'Quantity': quantity,
                    'Price': f"${price:.2f}",
                    'Amount': value_str,
                    'Reason': reason[:30] + '...' if len(reason) > 30 else reason
                })

            df_orders = pd.DataFrame(order_data)

            # Order statistics
            net_flow = total_proceeds - total_cost
            stats_text = f"""
📊 Order Statistics:
• Total Orders: {len(orders)}
• Buy Cost: ${total_cost:,.2f}
• Sell Proceeds: ${total_proceeds:,.2f}
• Net Cash Flow: ${net_flow:,.2f}
"""

            return df_orders, stats_text, f"🎯 Trading Orders - {len(orders)} orders"
        return None, None, "No orders data"

    def execute_trade(self, symbol, action, qty, results):
        # 🔥 Get real price
        price = 100
        if results.get('step1') and symbol in results['step1']:
            data = results['step1'][symbol]['data']
            if data:
                price = data[-1].get('close', 100)

        # 🔥 Load portfolio
        portfolio = self.load_portfolio_data()
        if not portfolio:
            portfolio = {"cash": 100000, "positions": {}}

        # 🔥 BUY logic
    def buy_stock(self, symbol, qty):
        import yfinance as yf

        stock = yf.Ticker(symbol)
        df = stock.history(period="1d")

        if df.empty:
          return "❌ Invalid stock"

        price = float(df["Close"].iloc[-1])

        portfolio = self.load_portfolio_data()
        if not portfolio:
            portfolio = {"cash": 100000, "positions": {}}

        cost = price * qty

        if portfolio["cash"] < cost:
           return "❌ Not enough cash"

        portfolio["cash"] -= cost

        pos = portfolio["positions"].get(symbol)

        if symbol in portfolio["positions"]:
            pos = portfolio["positions"][symbol]

            total_cost = pos["avg_price"] * pos["quantity"]
            new_cost = price * qty

            pos["quantity"] += qty
            pos["avg_price"] = (total_cost + new_cost) / pos["quantity"]

        else:
            portfolio["positions"][symbol] = {
                "quantity": qty,
                "avg_price": price
            }

        self.save_portfolio_data(portfolio)

        from datetime import datetime

        orders = self.load_orders()

        orders.append({
            "type": "BUY",
            "symbol": symbol,
            "qty": qty,
            "price": price,
            "time": str(datetime.now())
        })

        self.save_orders(orders)

        return f"✅ Bought {qty} shares of {symbol} at {price:.2f}"

        # 🔥 SELL logic (UPDATED)
    def sell_stock(self, symbol, qty):
        import yfinance as yf
        from datetime import datetime

        portfolio = self.load_portfolio_data()

        if symbol not in portfolio["positions"]:
            return "❌ Stock not in portfolio"

        pos = portfolio["positions"][symbol]

        if pos["quantity"] < qty:
            return "❌ Not enough shares"

        avg_price = pos.get("avg_price", 0)

        stock = yf.Ticker(symbol)
        df = stock.history(period="1d")

        if df.empty:
            return "❌ Price fetch failed"

        sell_price = float(df["Close"].iloc[-1])

        # 🔥 CALCULATE REALIZED P&L
        realized_pnl = (sell_price - avg_price) * qty

        # 🔥 UPDATE CASH (THIS AUTOMATICALLY ADDS P&L)
        portfolio["cash"] += sell_price * qty

        # 🔥 STORE REALIZED P&L
        portfolio["realized_pnl"] = portfolio.get("realized_pnl", 0) + realized_pnl

        # 🔥 UPDATE POSITION
        pos["quantity"] -= qty

        if pos["quantity"] == 0:
            del portfolio["positions"][symbol]

        self.save_portfolio_data(portfolio)

        # Save order
        orders = self.load_orders()
        orders.append({
            "type": "SELL",
            "symbol": symbol,
            "qty": qty,
            "price": sell_price,
            "pnl": realized_pnl,  # 🔥 store pnl here too
            "time": str(datetime.now())
        })
        self.save_orders(orders)

        return f"🔴 Sold {qty} shares of {symbol} at {sell_price:.2f} | P&L: {realized_pnl:.2f}"
    

    def buy_and_refresh(self, symbol, qty):
        msg = self.buy_stock(symbol, qty)

        # reload everything
        results = self.load_latest_results()
        portfolio = self.load_portfolio_data()

        portfolio_overview_df, positions_df, portfolio_fig, portfolio_text = self.create_portfolio_dashboard(portfolio)

        return msg, portfolio_overview_df, positions_df, portfolio_fig, portfolio_text
    

    def sell_and_refresh(self,symbol, qty):
        msg = self.sell_stock(symbol, qty)

        # reload everything
        portfolio = self.load_portfolio_data()
        portfolio_overview_df, positions_df, portfolio_fig, portfolio_text = self.create_portfolio_dashboard(portfolio)

        return msg, portfolio_overview_df, positions_df, portfolio_fig, portfolio_text
    





    def create_dashboard(self):
        """Create main dashboard"""

        custom_css = """
        body, .gradio-container {
            background: #0b1220 !important;
            color: #e5e7eb !important;
        }

        /* HEADER */
        .dashboard-header {
            background: linear-gradient(135deg, #4f46e5, #7c3aed);
            padding: 25px;
            border-radius: 16px;
            color: white;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        }

        /* KPI CARDS */
        .metric-card {
            background: linear-gradient(145deg, #111827, #1f2937);
            border-radius: 14px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 8px 25px rgba(0,0,0,0.6);
            transition: 0.3s;
        }

        .metric-card:hover {
            transform: translateY(-5px);
        }

        /* GLASS CARD */
        .glass-card {
            background: rgba(17, 24, 39, 0.8);
            border-radius: 14px;
            padding: 15px;
            border: 1px solid rgba(255,255,255,0.05);
        }

        /* INPUT */
        input {
            background: #111827 !important;
            color: white !important;
        }

        /* TABLE */
        thead {
            background: #1f2937 !important;
        }
        tbody tr:hover {
            background: rgba(99,102,241,0.2);
        }
        """

        with gr.Blocks(
            title="AI Trading System - Professional Dashboard",
            css=custom_css,
            theme=gr.themes.Base()
        ) as demo:

            # Page title
            gr.HTML("""
            <div class="dashboard-header">
                <h1>🚀 AI Trading System - Professional Dashboard</h1>
                <p>Intelligent Quantitative Trading Analysis Platform | AI-driven Market Analysis and Investment Decisions</p>
            </div>
            """)

            with gr.Row():
                with gr.Column():
                    total_capital = gr.HTML()

                with gr.Column():
                    pnl_card = gr.HTML()

                with gr.Column():
                    win_rate = gr.HTML()

                with gr.Column():
                    trades = gr.HTML()

            # Status control panel
            with gr.Row():
                with gr.Column(scale=1):
                    refresh_btn = gr.Button("🔄 Refresh", scale=1)
                    run_analysis_btn = gr.Button("⚡ Analyze", scale=1)
                with gr.Column(scale=3):
                    status_text = gr.Textbox(
                        label="📈 System Status",
                        value="Ready - Click 'Refresh Data' to start analysis",
                        interactive=False,
                        lines=2
                    )

            # Data state
            results_state = gr.State()

            # Main analysis tabs
            with gr.Tabs():

                # Market overview tab

                
                with gr.Tab("📊 Market Overview", id="market_overview"):
                    gr.Markdown("### 📈 Real-time Market Data")
                    with gr.Row():
                        # ✅ 🔍 SEARCH BAR (YOU MISSED THIS)
                    
                        search_input = gr.Textbox(
                            label="🔍 Search Stock (e.g. AAPL)",
                            placeholder="Enter symbol..."
                        )           
                    with gr.Row():
                            
                        # 📊 LEFT SIDE - MARKET TABLE
                        with gr.Column(scale=3):
                            gr.HTML('<div class="glass-card">')

                            market_table = gr.Dataframe(
                                headers=[
                                    "Symbol", "Price", "1H %", "4H %", "24H %", "1W %", "Volume"
                                ],
                                interactive=False,
                                wrap=True
                            )

                            gr.HTML('</div>')

                        # 📋 RIGHT SIDE - SUMMARY
                        with gr.Column(scale=1):
                            gr.HTML('<div class="glass-card">')

                            market_summary = gr.Textbox(
                                label="📊 Market Summary",
                                lines=8
                            )

                            gr.HTML('</div>')
                       
                        

                # Technical analysis tab
                with gr.Tab("📈 Technical Analysis", id="technical_analysis"):

                    gr.Markdown("### 📈 Live Technical Analysis")

                    # 🔍 Search bar
                    stock_search = gr.Textbox(
                        label="🔍 Search Stock",
                        placeholder="AAPL, TSLA, TCS.NS..."
                    )
                    
                    search_btn = gr.Button("🔍 Analyze")
                    gr.Markdown("Try: AAPL | TSLA | NVDA | TCS.NS | RELIANCE.NS")
                    # 📊 Chart
                    price_chart = gr.Plot(label="📊 Live Chart", show_label=False)

                # AI analysis tab
                with gr.Tab("🤖 AI Analysis"):

                    gr.Markdown("## 🤖 Multi-Timeframe AI Market Analysis")

                    ai_symbols = gr.Textbox(
                        label="Enter Stocks",
                        placeholder="RELIANCE.NS, TCS.NS"
                    )

                    analyze_btn = gr.Button("🚀 Analyze AI")

                    with gr.Row():

                        with gr.Column():
                            gr.Markdown("### 📈 Intraday Analysis")
                            intraday_table = gr.Dataframe()
                            intraday_chart = gr.Plot()

                        with gr.Column():
                            gr.Markdown("### 📊 Short-Term Swing")
                            short_table = gr.Dataframe()
                            short_chart = gr.Plot()

                    with gr.Row():

                        with gr.Column():
                            gr.Markdown("### 🏦 Long-Term Investment")
                            long_table = gr.Dataframe()
                            long_chart = gr.Plot()

                        with gr.Column():
                            gr.Markdown("### 🧠 Overall AI Summary")
                            summary_box = gr.Markdown()
                    
                    analyze_btn.click(
                        fn=self.create_ai_analysis,
                        inputs=[ai_symbols],
                        outputs=[
                            intraday_table,
                            intraday_chart,
                            short_table,
                            short_chart,
                            long_table,
                            long_chart,
                            summary_box
                        ]
                    )
                
            
                # Trading signals tab
                with gr.Tab("📰 Market News"):

                    gr.Markdown("## 📰 Live Market News")

                    news_search = gr.Textbox(
                        placeholder="Search stock (e.g. RELIANCE, TCS, NIFTY)",
                        label="🔍 Search News"
                    )

                    news_output = gr.HTML(
                        value="<p style='color:gray;'>Loading latest news...</p>"
                    )

                    news_status = gr.Textbox(label="Status")

                    news_search.change(
                        fn=self.get_market_news,
                        inputs=[news_search],
                        outputs=[news_output, news_status]
                    )


                # Algo Trading Tab
                with gr.Tab("⚡ Algo Trading"):

                    algo_code = gr.Code(
                        value=self.load_strategy(),
                        label="Strategy Logic",
                        language="python"
                    )

                    market_selector = gr.Dropdown(
                        choices=[
                            "India",
                            "USA",
                            "UK",
                            "Japan",
                            "Germany",
                            "France",
                            "China",
                            "Hong Kong",
                            "Canada",
                            "Australia"
                        ],
                        value="India",
                        label="Select Market Universe"
                    )

                    with gr.Row():
                        start_algo_btn = gr.Button("▶ Start Algo")
                        stop_algo_btn = gr.Button("⏹ Stop Algo")

                    algo_status = gr.Textbox(
                        label="Algo Status",
                        value="🔴 Algo Stopped"
                    )

                    live_positions_table = gr.Dataframe(
                        label="Live Positions"
                    )

                    algo_partial_sell_stock = gr.Dropdown(
                        label="Select Algo Position"
                    )

                    algo_partial_sell_qty = gr.Number(
                        label="Qty To Sell",
                        value=1
                    )

                    partial_sell_btn = gr.Button("Partial Exit")

                    close_all_algo_btn = gr.Button(
                        "Close All Algo Positions",
                        variant="stop"
                    )

                    partial_sell_btn.click(
                        fn=self.partial_close_algo_position,
                        inputs=[
                            algo_partial_sell_stock,
                            algo_partial_sell_qty
                        ],
                        outputs=[algo_status]
                    ).then(
                        fn=self.refresh_algo_dashboard,
                        outputs=[
                            algo_status,
                            live_positions_table,
                            algo_partial_sell_stock,
                        
                        ]
                    )

                    close_all_algo_btn.click(
                        fn=self.close_all_algo_positions,
                        outputs=[algo_status]
                    ).then(
                        fn=self.refresh_algo_dashboard,
                        outputs=[
                            algo_status,
                            live_positions_table,
                            algo_partial_sell_stock,
                            
                        ]
                    )

                    

                    trade_history_table = gr.Dataframe(
                        label="Trade History"
                    )

                    pnl_box = gr.Textbox(
                        label="PnL Summary",
                        value="₹ 0.00"
                    )

                    # AUTO REFRESH TIMER
                    algo_timer = gr.Timer(value=60)

                    algo_timer.tick(
                        fn=self.refresh_algo_dashboard,
                        outputs=[
                            algo_status,
                            live_positions_table,
                            algo_partial_sell_stock,
                            trade_history_table,
                            pnl_box
                        ]
                    )      


                    

                # Portfolio tab
                with gr.Tab("💼 Portfolio", id="portfolio"):
                    gr.Markdown("### 💰 Portfolio Management")

                    # Portfolio overview
                    with gr.Row():
                        with gr.Column(scale=1):
                            portfolio_overview = gr.Dataframe(
                                label="Portfolio Overview",
                                interactive=False
                            )
                        with gr.Column(scale=1):
                            portfolio_pie = gr.Plot(label="Asset Allocation", show_label=False)

                    # Position details
                    gr.Markdown("#### 📊 Position Details")
                    positions_table = gr.Dataframe(
                        label="Current Positions",
                        interactive=False
                    )
                    portfolio_summary = gr.Textbox(
                       label="📊 Portfolio Summary",
                       interactive=False,
                       lines=5   # 🔥 ज्यादा space
                    )

                # Trading orders tab
                with gr.Tab("🎯 Trading Orders", id="orders"):
                    gr.Markdown("### 📋 Intelligent Order Generation")
                    orders_table = gr.Dataframe(
                        label="Generated Trading Orders",
                        interactive=False
                    )
                    orders_stats = gr.Markdown()
                    orders_summary = gr.Markdown()

                    gr.Markdown("### 🖱 Manual Trade Execution")

                    # 🔍 Stock Input
                    stock_input = gr.Textbox(
                        label="🔍 Enter Stock (e.g. AAPL, TCS.NS)",
                        placeholder="Type stock name..."
                    )

                   # 🔢 Quantity
                    quantity_input = gr.Number(
                        label="Quantity",
                        value=1
                    )

                  # 🔘 Buttons
                    with gr.Row():
                       buy_btn = gr.Button("🟢 BUY")
                       sell_btn = gr.Button("🔴 SELL")

                      # 📢 Output
                    trade_output = gr.Textbox(label="Trade Status")

            # AUTO REFRESH TIMER
                algo_timer = gr.Timer(value=60)

                algo_timer.tick(
                    fn=self.refresh_algo_dashboard,
                    outputs=[
                        algo_status,
                        live_positions_table,
                        algo_partial_sell_stock,
                        trade_history_table,
                        pnl_box
                    ]
                )      
                    
            # Event handling functions
            def execute_trade_callback(symbol, action, qty, results):
                return self.execute_trade(symbol, action, qty, results)
            

            def buy_and_refresh(symbol, qty):
               msg = self.buy_stock(symbol, qty)

               orders_df, orders_stats_text, orders_summary_text = self.create_orders_dashboard()

               portfolio = self.load_portfolio_data()
               portfolio_overview_df, positions_df, portfolio_fig, portfolio_text = self.create_portfolio_dashboard(portfolio)

               return (
                  msg,
                  portfolio_overview_df,
                  positions_df,
                  portfolio_fig,
                  portfolio_text,
                 orders_df,
                 orders_stats_text,
                 orders_summary_text
            )


            def sell_and_refresh(symbol, qty):
                msg = self.sell_stock(symbol, qty)

                orders_df, orders_stats_text, orders_summary_text = self.create_orders_dashboard()

                portfolio = self.load_portfolio_data()
                portfolio_overview_df, positions_df, portfolio_fig, portfolio_text = self.create_portfolio_dashboard(portfolio)

                return (
                  msg,
                  portfolio_overview_df,
                  positions_df,
                  portfolio_fig,
                  portfolio_text,
                  orders_df,
                  orders_stats_text,
                  orders_summary_text
               )

            def load_and_refresh(symbol_input):
                """Load and refresh all data"""
                results = self.load_latest_results()
                portfolio_data = self.load_portfolio_data()

                if not results:
                    return (
                        "❌ No analysis data found, please run the trading pipeline first",
                        results,
                        None, None, None,  # market
                        None, None, None,   # ai
                        None, None, None,   # signals
                        None, None, None, None,  # portfolio
                        None, None, None,   # orders
                        None  # price_chart
                    )
                
                # Market overview
                market_df, market_text = self.create_market_overview(results,
                   symbol=symbol_input   
                )

                # AI analysis
                

                # Trading signals

                # Portfolio
                portfolio_overview_df, positions_df, portfolio_fig, portfolio_text = self.create_portfolio_dashboard(portfolio_data)

                # Orders
                orders_df, orders_stats_text, orders_text = self.create_orders_panel(results)
                
                # Create initial price chart for default symbol (AAPL)
                # ✅ Step 1: handle empty input
                if not symbol_input:
                    symbol_input = "AAPL"

                # ✅ Step 2: convert to uppercase
                symbol = symbol_input.upper()

                # ✅ Step 3: create chart
                initial_chart = self.create_chart(symbol)

                # ✅ Step 4: return everything
                return (
                    f"✅ Data loaded successfully",
                    results,
                    market_df, market_text,
                    portfolio_overview_df, positions_df, portfolio_fig, portfolio_text,
                    orders_df, orders_stats_text, orders_text,
                    initial_chart
                )

            def update_chart(symbol, results):
                """Update price chart"""
                if not symbol or not results:
                    return None
                return self.create_price_chart(symbol, results)

            # Bind events
            search_input.change(
                load_and_refresh,
                inputs=[search_input],
                outputs=[
                    status_text, results_state,
                    market_table, market_summary,
                    portfolio_overview, positions_table, portfolio_pie, portfolio_summary,
                    orders_table, orders_stats, orders_summary,
                    price_chart
               ]
            )
            refresh_btn.click(
                load_and_refresh,
                inputs=[search_input],
                outputs=[
                    status_text, results_state,
                    market_table, market_summary,
                    portfolio_overview, positions_table, portfolio_pie, portfolio_summary,
                    orders_table, orders_stats, orders_summary,
                    price_chart  # Add price chart to refresh outputs
                ]
            )

            stock_search.submit(
                fn=self.create_chart,
                inputs=[stock_search],
                outputs=[price_chart]
            )

            search_btn.click(
                fn=self.create_chart,
                inputs=[stock_search],
                outputs=[price_chart]
            )

            buy_btn.click(
                fn=buy_and_refresh,
                inputs=[stock_input, quantity_input],
                outputs=[
                     trade_output,
                     portfolio_overview,
                     positions_table,
                     portfolio_pie,
                     portfolio_summary,
                     orders_table,
                     orders_stats,
                     orders_summary
                ]
            )

            sell_btn.click(
                fn=sell_and_refresh,
                inputs=[stock_input, quantity_input],
                outputs=[
                    trade_output,
                    portfolio_overview,
                    positions_table,
                    portfolio_pie,
                    portfolio_summary,
                    orders_table,
                    orders_stats,
                    orders_summary
                ]
           )

            run_analysis_btn.click(
                load_and_refresh,
                inputs=[search_input],
                outputs=[
                    status_text, results_state,
                    market_table, market_summary,
                    portfolio_overview, positions_table, portfolio_pie, portfolio_summary,
                    orders_table, orders_stats, orders_summary,
                    price_chart  # Add price chart to refresh outputs
                ]
            )

            start_algo_btn.click(
                fn=self.start_algo,
                inputs=[algo_code, market_selector],
                outputs=[algo_status]
            )

            stop_algo_btn.click(
                fn=self.stop_algo,
                outputs=[algo_status]
            )


            demo.load(
                fn=self.refresh_algo_dashboard,
                outputs=[
                    algo_status,
                    live_positions_table,
                    algo_partial_sell_stock,
                    trade_history_table,
                    pnl_box
                ]
            )
             
            demo.load(
                fn=self.create_chart,
                inputs=[stock_search],
                outputs=[price_chart]
                
            ) 

            demo.load(
                fn=self.get_market_news,
                outputs=[news_output, news_status]
            )
           
        return demo

def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="AI Trading System Professional Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python trading_dashboard.py                    # Start dashboard locally
  python trading_dashboard.py --share           # Start dashboard with public sharing
  python trading_dashboard.py --port 8080       # Start on custom port
  python trading_dashboard.py --share --port 8080  # Share on custom port
        """
    )

    parser.add_argument(
        '--share',
        action='store_true',
        help='Enable public sharing (creates public URL via Gradio)'
    )

    parser.add_argument(
        '--port',
        type=int,
        default=7860,
        help='Port number for the dashboard (default: 7860)'
    )

    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host address to bind to (default: 0.0.0.0)'
    )

    parser.add_argument(
        '--no-browser',
        action='store_true',
        help='Do not automatically open browser'
    )

    args = parser.parse_args()

    dashboard = TradingDashboard()
    demo = dashboard.create_dashboard()

    print("🚀 Starting AI Trading System Professional Dashboard...")
    print("📊 Modern interface design, professional trading experience")

    if args.share:
        print("🌐 Public sharing enabled - Dashboard will be accessible via public URL")
    else:
        print("🔒 Local access only - Dashboard accessible at localhost")

    print(f"🔗 Server: {args.host}:{args.port}")

    if not args.no_browser:
        print("🌐 Dashboard will open in browser")

    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        inbrowser=not args.no_browser
    )

if __name__ == "__main__":
    main()
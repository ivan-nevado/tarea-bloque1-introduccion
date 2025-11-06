from .alpha_vantage_base import AlphaVantageBase
from .statistics_analyzer import StatisticsAnalyzer
import pandas as pd
from datetime import datetime
import time

class StockPrices(AlphaVantageBase):
    """Download stock price data from Alpha Vantage"""
    
    def __init__(self):
        super().__init__()
        self.stats_analyzer = StatisticsAnalyzer()
    
    def download_historical_years(self, symbol: str, years: int) -> pd.DataFrame:
        """
        Download historical stock data for specified number of years
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            years: Number of years of historical data
        
        Returns:
            DataFrame with OHLCV data
        """
        
        print(f"Downloading {symbol} stock data ({years} years)...")
        
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'outputsize': 'full'
        }
        
        try:
            data = self.make_request(params)
            
            # Check for API messages
            if 'Information' in data:
                print(f"API Info: {data['Information']}")
                if 'call frequency' in data['Information'].lower():
                    print("Rate limit hit. Waiting 60 seconds...")
                    time.sleep(60)
                    data = self.make_request(params)  # Retry
            
            if 'Time Series (Daily)' not in data:
                available_keys = list(data.keys())
                if 'Information' in data:
                    raise ValueError(f"API returned: {data['Information']}")
                raise ValueError(f"Expected 'Time Series (Daily)' key not found. Available keys: {available_keys}")
            
            time_series = data['Time Series (Daily)']
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            
            # Convert to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert index to datetime
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Filter by years
            if years > 0:
                cutoff_date = datetime.now() - pd.DateOffset(years=years)
                df = df[df.index >= cutoff_date]
            
            if df.empty:
                raise ValueError(f"No data available for {symbol} in the last {years} years")
            
            # Generate filename
            current_date = datetime.now().strftime('%Y%m%d')
            filename = f"{symbol}_stock_{years}y_{current_date}.csv"
            
            # Save to CSV
            filepath = self.save_to_csv(df, filename)
            
            # Generate statistics
            stats = self.stats_analyzer.analyze_price_data(df, symbol)
            if stats:
                stats_filename = f"{symbol}_stock_{years}y_stats_{current_date}.csv"
                self.stats_analyzer.save_statistics(stats, stats_filename)
                self.stats_analyzer.print_summary(stats)
            
            print(f"✓ {symbol}: {len(df)} days saved to {filepath}")
            
            return df
            
        except Exception as e:
            print(f"✗ Error downloading {symbol}: {e}")
            raise
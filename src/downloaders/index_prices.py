from .alpha_vantage_base import AlphaVantageBase
from .statistics_analyzer import StatisticsAnalyzer
import pandas as pd
from datetime import datetime

class IndexPrices(AlphaVantageBase):
    """Download index price data from Alpha Vantage"""
    
    def __init__(self):
        super().__init__()
        self.stats_analyzer = StatisticsAnalyzer()
        
        # Alpha Vantage index symbols
        self.index_symbols = {
            'SPX': 'SPY',    # S&P 500 ETF
            'NDX': 'QQQ',    # NASDAQ 100 ETF
            'RUT': 'IWM',    # Russell 2000 ETF
            'VIX': 'VXX',    # VIX ETF
            'DJI': 'DIA'     # Dow Jones ETF
        }
    
    def download_historical_index_years(self, index: str, years: int) -> pd.DataFrame:
        """
        Download historical index data for specified number of years
        
        Args:
            index: Index symbol (e.g., 'SPX', 'NDX')
            years: Number of years of historical data
        
        Returns:
            DataFrame with OHLCV data
        """
        
        # Convert to Alpha Vantage symbol
        av_symbol = self.index_symbols.get(index, index)
        
        print(f"Downloading {index} index data ({years} years)...")
        
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': av_symbol,
            'outputsize': 'full'
        }
        
        try:
            data = self.make_request(params)
            
            if 'Time Series (Daily)' not in data:
                available_keys = list(data.keys())
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
                raise ValueError(f"No data available for {index} in the last {years} years")
            
            # Generate filename
            current_date = datetime.now().strftime('%Y%m%d')
            filename = f"{index}_index_{years}y_{current_date}.csv"
            
            # Save to CSV
            filepath = self.save_to_csv(df, filename)
            
            # Generate statistics
            stats = self.stats_analyzer.analyze_price_data(df, index)
            if stats:
                stats_filename = f"{index}_index_{years}y_stats_{current_date}.csv"
                self.stats_analyzer.save_statistics(stats, stats_filename)
                self.stats_analyzer.print_summary(stats)
            
            print(f"✓ {index}: {len(df)} days saved to {filepath}")
            
            return df
            
        except Exception as e:
            print(f"✗ Error downloading {index}: {e}")
            raise
import pandas as pd
import numpy as np
from typing import Union, List, Optional

class DataPreprocessor:
    """Clean and preprocess any time series price data for Monte Carlo simulation"""
    
    @staticmethod
    def detect_price_column(df: pd.DataFrame) -> str:
        """Automatically detect the main price column"""
        # Priority order for price columns
        price_columns = [
            'close', 'Close', 'CLOSE',
            'price', 'Price', 'PRICE', 
            'adj_close', 'adjusted_close', 'Adj Close',
            'value', 'Value', 'VALUE'
        ]
        
        for col in price_columns:
            if col in df.columns:
                return col
        
        # If no standard column found, use first numeric column
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            return numeric_cols[0]
        
        raise ValueError("No suitable price column found")
    
    @staticmethod
    def detect_date_column(df: pd.DataFrame) -> Optional[str]:
        """Detect date column if not in index"""
        date_columns = [
            'date', 'Date', 'DATE',
            'timestamp', 'Timestamp', 'TIMESTAMP',
            'time', 'Time', 'TIME'
        ]
        
        for col in date_columns:
            if col in df.columns:
                return col
        
        return None
    
    @staticmethod
    def standardize_dataframe(df: pd.DataFrame, 
                            price_column: Optional[str] = None,
                            date_column: Optional[str] = None) -> pd.DataFrame:
        """
        Standardize any DataFrame to have datetime index and 'close' column
        
        Args:
            df: Input DataFrame
            price_column: Name of price column (auto-detected if None)
            date_column: Name of date column (auto-detected if None)
        
        Returns:
            Standardized DataFrame with datetime index and 'close' column
        """
        df_clean = df.copy()
        
        # Handle date column
        if date_column is None:
            date_column = DataPreprocessor.detect_date_column(df_clean)
        
        if date_column and date_column in df_clean.columns:
            # Move date column to index
            df_clean[date_column] = pd.to_datetime(df_clean[date_column])
            df_clean.set_index(date_column, inplace=True)
        elif not isinstance(df_clean.index, pd.DatetimeIndex):
            # Try to convert existing index to datetime
            try:
                df_clean.index = pd.to_datetime(df_clean.index)
            except:
                # Create artificial date index
                df_clean.index = pd.date_range(start='2020-01-01', periods=len(df_clean), freq='D')
        
        # Handle price column
        if price_column is None:
            price_column = DataPreprocessor.detect_price_column(df_clean)
        
        if price_column not in df_clean.columns:
            raise ValueError(f"Price column '{price_column}' not found")
        
        # Create standardized DataFrame with 'close' column
        result = pd.DataFrame(index=df_clean.index)
        result['close'] = pd.to_numeric(df_clean[price_column], errors='coerce')
        
        # Add other OHLCV columns if available
        column_mapping = {
            'open': ['open', 'Open', 'OPEN'],
            'high': ['high', 'High', 'HIGH'],
            'low': ['low', 'Low', 'LOW'],
            'volume': ['volume', 'Volume', 'VOLUME', 'vol', 'Vol']
        }
        
        for std_col, possible_cols in column_mapping.items():
            for col in possible_cols:
                if col in df_clean.columns:
                    result[std_col] = pd.to_numeric(df_clean[col], errors='coerce')
                    break
        
        # Fill missing OHLCV with close price if not available
        if 'open' not in result.columns:
            result['open'] = result['close']
        if 'high' not in result.columns:
            result['high'] = result['close']
        if 'low' not in result.columns:
            result['low'] = result['close']
        if 'volume' not in result.columns:
            result['volume'] = 1000000  # Default volume
        
        return result
    
    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean data by removing outliers and handling missing values"""
        df_clean = df.copy()
        
        # Remove rows with missing close prices
        df_clean = df_clean.dropna(subset=['close'])
        
        # Remove zero or negative prices
        df_clean = df_clean[df_clean['close'] > 0]
        
        # Remove extreme outliers (prices that change more than 50% in one day)
        if len(df_clean) > 1:
            returns = df_clean['close'].pct_change()
            outlier_mask = (returns.abs() > 0.5) & (returns.notna())
            df_clean = df_clean[~outlier_mask]
        
        # Forward fill missing values in other columns
        df_clean = df_clean.fillna(method='ffill')
        
        # Sort by date
        df_clean = df_clean.sort_index()
        
        return df_clean
    
    @staticmethod
    def validate_data(df: pd.DataFrame, min_periods: int = 30) -> bool:
        """Validate that data is suitable for Monte Carlo simulation"""
        if len(df) < min_periods:
            return False
        
        if 'close' not in df.columns:
            return False
        
        if df['close'].isna().all():
            return False
        
        if (df['close'] <= 0).all():
            return False
        
        return True
    
    @staticmethod
    def process_any_format(data: Union[pd.DataFrame, str, dict], 
                          price_column: Optional[str] = None,
                          date_column: Optional[str] = None) -> pd.DataFrame:
        """
        Process any input format into standardized DataFrame
        
        Args:
            data: DataFrame, CSV file path, or dict
            price_column: Name of price column
            date_column: Name of date column
        
        Returns:
            Cleaned and standardized DataFrame
        """
        # Handle different input types
        if isinstance(data, str):
            # Assume it's a file path
            df = pd.read_csv(data, index_col=0, parse_dates=True)
        elif isinstance(data, dict):
            # Convert dict to DataFrame
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError("Unsupported data format")
        
        # Standardize format
        df_std = DataPreprocessor.standardize_dataframe(df, price_column, date_column)
        
        # Clean data
        df_clean = DataPreprocessor.clean_data(df_std)
        
        # Validate
        if not DataPreprocessor.validate_data(df_clean):
            raise ValueError("Data failed validation - insufficient or invalid price data")
        
        return df_clean
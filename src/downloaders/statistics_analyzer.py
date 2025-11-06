import pandas as pd
import numpy as np
import os
from datetime import datetime

class StatisticsAnalyzer:
    """Generate statistical analysis for financial data"""
    
    def analyze_price_data(self, df: pd.DataFrame, symbol: str) -> dict:
        """Analyze price data and return statistics"""
        if df is None or df.empty:
            return None
        
        # Calculate returns
        df['daily_return'] = df['close'].pct_change()
        
        stats = {
            'symbol': symbol,
            'data_points': len(df),
            'date_range': f"{df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}",
            
            # Price statistics
            'price_mean': df['close'].mean(),
            'price_std': df['close'].std(),
            'price_min': df['close'].min(),
            'price_max': df['close'].max(),
            'price_median': df['close'].median(),
            
            # Return statistics
            'return_mean': df['daily_return'].mean(),
            'return_std': df['daily_return'].std(),
            'return_min': df['daily_return'].min(),
            'return_max': df['daily_return'].max(),
            
            # Volume statistics
            'volume_mean': df['volume'].mean(),
            'volume_std': df['volume'].std(),
            'volume_median': df['volume'].median(),
            
            # Risk metrics
            'volatility_annualized': df['daily_return'].std() * np.sqrt(252),
            'sharpe_ratio': (df['daily_return'].mean() * 252) / (df['daily_return'].std() * np.sqrt(252)) if df['daily_return'].std() > 0 else 0,
            
            # Performance metrics
            'total_return': (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100,
            'max_drawdown': self._calculate_max_drawdown(df['close']),
        }
        
        return stats
    
    def analyze_fundamental_data(self, data: dict, symbol: str) -> dict:
        """Analyze fundamental data and return statistics"""
        if not data or all(v is None for v in data.values()):
            return None
        
        stats = {'symbol': symbol}
        
        # Analyze each statement type
        for statement_type, df in data.items():
            if df is None or df.empty:
                continue
            
            # Get numeric columns only
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 0:
                stats[f'{statement_type}_metrics'] = {
                    'mean': df[numeric_cols].mean().to_dict(),
                    'std': df[numeric_cols].std().to_dict(),
                    'growth_rates': self._calculate_growth_rates(df, numeric_cols)
                }
        
        return stats
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min() * 100
    
    def _calculate_growth_rates(self, df: pd.DataFrame, numeric_cols: list) -> dict:
        """Calculate growth rates for numeric columns"""
        if len(df) < 2:
            return {}
        
        growth_rates = {}
        for col in numeric_cols:
            if df[col].iloc[0] != 0 and not pd.isna(df[col].iloc[0]):
                growth_rate = ((df[col].iloc[-1] / df[col].iloc[0]) ** (1/len(df)) - 1) * 100
                growth_rates[col] = growth_rate
        
        return growth_rates
    
    def save_statistics(self, stats: dict, filename: str):
        """Save statistics to CSV file"""
        if stats is None:
            return
        
        # Flatten nested dictionaries for CSV export
        flattened = self._flatten_dict(stats)
        
        # Convert to DataFrame
        stats_df = pd.DataFrame([flattened])
        
        # Save to CSV
        os.makedirs('data/statistics', exist_ok=True)
        filepath = f'data/statistics/{filename}'
        stats_df.to_csv(filepath, index=False)
        print(f"  📊 Statistics saved: {filepath}")
    
    def _flatten_dict(self, d: dict, parent_key: str = '', sep: str = '_') -> dict:
        """Flatten nested dictionary"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def print_summary(self, stats: dict):
        """Print key statistics summary"""
        if stats is None:
            return
        
        symbol = stats.get('symbol', 'Unknown')
        print(f"\n📈 {symbol} Statistics Summary:")
        
        # Price statistics
        if 'price_mean' in stats:
            print(f"  Price: ${stats['price_mean']:.2f} ± ${stats['price_std']:.2f}")
            print(f"  Total Return: {stats['total_return']:.2f}%")
            print(f"  Volatility: {stats['volatility_annualized']:.2f}%")
            print(f"  Sharpe Ratio: {stats['sharpe_ratio']:.3f}")
            print(f"  Max Drawdown: {stats['max_drawdown']:.2f}%")
        
        # Fundamental statistics
        for key in stats:
            if key.endswith('_metrics'):
                statement_type = key.replace('_metrics', '')
                print(f"  {statement_type.title()}: Available")
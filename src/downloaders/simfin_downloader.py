import simfin as sf
import pandas as pd
import os
from datetime import datetime
from typing import List
from dotenv import load_dotenv
from .statistics_analyzer import StatisticsAnalyzer

load_dotenv()

class SimFinDownloader:
    """SimFin downloader using official library"""
    
    def __init__(self):
        api_key = os.getenv('SIMFIN_API_KEY')
        if not api_key:
            raise Exception("SimFin API key not found in .env file")
        
        sf.set_api_key(api_key)
        sf.set_data_dir('data/simfin_cache/')
    
    def get_income_statement(self, ticker: str, years: int = 5) -> pd.DataFrame:
        """Get income statement for specified years"""
        df = sf.load_income(variant='quarterly', market='us')
        
        if ticker not in df.index:
            raise Exception(f"Ticker {ticker} not found in income data")
        
        ticker_data = df.loc[ticker]
        
        # Filter by years (quarterly data = 4 quarters per year)
        quarters_needed = years * 4
        
        if hasattr(ticker_data.index, 'get_level_values'):
            # Multi-index with dates - get last N quarters
            filtered_data = ticker_data.tail(quarters_needed)
        else:
            # Simple filtering by last N quarters
            filtered_data = ticker_data.tail(quarters_needed)
        
        return filtered_data.to_frame().T if isinstance(filtered_data, pd.Series) else filtered_data
    
    def get_balance_sheet(self, ticker: str, years: int = 5) -> pd.DataFrame:
        """Get balance sheet for specified years"""
        df = sf.load_balance(variant='quarterly', market='us')
        
        if ticker not in df.index:
            raise Exception(f"Ticker {ticker} not found in balance sheet data")
        
        ticker_data = df.loc[ticker]
        
        # Filter by years (quarterly data = 4 quarters per year)
        quarters_needed = years * 4
        
        if hasattr(ticker_data.index, 'get_level_values'):
            # Multi-index with dates - get last N quarters
            filtered_data = ticker_data.tail(quarters_needed)
        else:
            # Simple filtering by last N quarters
            filtered_data = ticker_data.tail(quarters_needed)
        
        return filtered_data.to_frame().T if isinstance(filtered_data, pd.Series) else filtered_data
    
    def get_cash_flow(self, ticker: str, years: int = 5) -> pd.DataFrame:
        """Get cash flow for specified years"""
        df = sf.load_cashflow(variant='quarterly', market='us')
        
        if ticker not in df.index:
            raise Exception(f"Ticker {ticker} not found in cash flow data")
        
        ticker_data = df.loc[ticker]
        
        # Filter by years (quarterly data = 4 quarters per year)
        quarters_needed = years * 4
        
        if hasattr(ticker_data.index, 'get_level_values'):
            # Multi-index with dates - get last N quarters
            filtered_data = ticker_data.tail(quarters_needed)
        else:
            # Simple filtering by last N quarters
            filtered_data = ticker_data.tail(quarters_needed)
        
        return filtered_data.to_frame().T if isinstance(filtered_data, pd.Series) else filtered_data
    
    def download_all_statements(self, ticker: str, years: int = 5):
        """Download all financial statements for a company"""
        print(f"Downloading {years} years of fundamental data for {ticker}...")
        
        results = {}
        
        # Income Statement
        try:
            print(f"  Getting income statement...")
            income = self.get_income_statement(ticker, years)
            filename = f"data/{ticker}_income_{years}y_{datetime.now().strftime('%Y%m%d')}.csv"
            os.makedirs('data', exist_ok=True)
            income.to_csv(filename)
            print(f"  ✓ Income statement saved: {filename}")
            results['income'] = income
        except Exception as e:
            print(f"  ✗ Income statement error: {e}")
            results['income'] = None
        
        # Balance Sheet
        try:
            print(f"  Getting balance sheet...")
            balance = self.get_balance_sheet(ticker, years)
            filename = f"data/{ticker}_balance_{years}y_{datetime.now().strftime('%Y%m%d')}.csv"
            balance.to_csv(filename)
            print(f"  ✓ Balance sheet saved: {filename}")
            results['balance'] = balance
        except Exception as e:
            print(f"  ✗ Balance sheet error: {e}")
            results['balance'] = None
        
        # Cash Flow
        try:
            print(f"  Getting cash flow...")
            cashflow = self.get_cash_flow(ticker, years)
            filename = f"data/{ticker}_cashflow_{years}y_{datetime.now().strftime('%Y%m%d')}.csv"
            cashflow.to_csv(filename)
            print(f"  ✓ Cash flow saved: {filename}")
            results['cashflow'] = cashflow
        except Exception as e:
            print(f"  ✗ Cash flow error: {e}")
            results['cashflow'] = None
        
        # Generate and save statistics for fundamental data
        analyzer = StatisticsAnalyzer()
        stats = analyzer.analyze_fundamental_data(results, ticker)
        if stats:
            stats_filename = f"{ticker}_fundamentals_{years}y_stats_{datetime.now().strftime('%Y%m%d')}.csv"
            analyzer.save_statistics(stats, stats_filename)
            analyzer.print_summary(stats)
        
        return results
    
    def download_multiple_companies(self, tickers: List[str], years: int = 5):
        """Download fundamental data for multiple companies"""
        all_results = {}
        
        for ticker in tickers:
            print(f"\n=== Processing {ticker} ===")
            all_results[ticker] = self.download_all_statements(ticker, years)
        
        return all_results
import concurrent.futures
import time
from typing import List, Dict, Callable, Any
from .stock_prices import StockPrices
from .index_prices import IndexPrices
from .simfin_downloader import SimFinDownloader

class ParallelDownloader:
    """Parallel data downloader using ThreadPoolExecutor"""
    
    def __init__(self, max_workers: int = 3):
        """
        Initialize parallel downloader
        
        Args:
            max_workers: Maximum number of concurrent downloads (respect API limits)
        """
        self.max_workers = max_workers
        self.stock_downloader = StockPrices()
        self.index_downloader = IndexPrices()
        self.simfin_downloader = SimFinDownloader()
    
    def download_stocks_parallel(self, symbols: List[str], years: int) -> Dict[str, Any]:
        """Download multiple stocks in parallel"""
        
        def download_single_stock(symbol: str) -> tuple:
            try:
                print(f"🔄 Downloading {symbol}...")
                df = self.stock_downloader.download_historical_years(symbol, years)
                print(f"✅ {symbol}: {len(df)} days")
                return symbol, df, None
            except Exception as e:
                print(f"❌ {symbol}: {e}")
                return symbol, None, str(e)
        
        results = {}
        errors = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all download tasks
            future_to_symbol = {executor.submit(download_single_stock, symbol): symbol 
                              for symbol in symbols}
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol, data, error = future.result()
                if error:
                    errors[symbol] = error
                else:
                    results[symbol] = data
        
        return {'results': results, 'errors': errors}
    
    def download_indices_parallel(self, indices: List[str], years: int) -> Dict[str, Any]:
        """Download multiple indices in parallel"""
        
        def download_single_index(index: str) -> tuple:
            try:
                print(f"🔄 Downloading {index}...")
                df = self.index_downloader.download_historical_index_years(index, years)
                print(f"✅ {index}: {len(df)} days")
                return index, df, None
            except Exception as e:
                print(f"❌ {index}: {e}")
                return index, None, str(e)
        
        results = {}
        errors = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {executor.submit(download_single_index, index): index 
                             for index in indices}
            
            for future in concurrent.futures.as_completed(future_to_index):
                index, data, error = future.result()
                if error:
                    errors[index] = error
                else:
                    results[index] = data
        
        return {'results': results, 'errors': errors}
    
    def download_fundamentals_parallel(self, tickers: List[str], years: int) -> Dict[str, Any]:
        """Download fundamentals for multiple companies in parallel"""
        
        def download_single_fundamental(ticker: str) -> tuple:
            try:
                print(f"🔄 Downloading {ticker} fundamentals...")
                results = self.simfin_downloader.download_all_statements(ticker, years)
                success_count = len([r for r in results.values() if r is not None])
                print(f"✅ {ticker}: {success_count}/3 statements")
                return ticker, results, None
            except Exception as e:
                print(f"❌ {ticker}: {e}")
                return ticker, None, str(e)
        
        results = {}
        errors = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_ticker = {executor.submit(download_single_fundamental, ticker): ticker 
                              for ticker in tickers}
            
            for future in concurrent.futures.as_completed(future_to_ticker):
                ticker, data, error = future.result()
                if error:
                    errors[ticker] = error
                else:
                    results[ticker] = data
        
        return {'results': results, 'errors': errors}
    
    def download_portfolio_parallel(self, 
                                  stocks: List[str], 
                                  indices: List[str], 
                                  fundamentals: List[str],
                                  years: int) -> Dict[str, Any]:
        """Download complete portfolio data in parallel"""
        
        print(f"🚀 Starting parallel download with {self.max_workers} workers...")
        start_time = time.time()
        
        all_results = {
            'stocks': {'results': {}, 'errors': {}},
            'indices': {'results': {}, 'errors': {}}, 
            'fundamentals': {'results': {}, 'errors': {}}
        }
        
        # Use ThreadPoolExecutor for different data types
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            
            # Submit stock downloads
            if stocks:
                future_stocks = executor.submit(self.download_stocks_parallel, stocks, years)
                futures.append(('stocks', future_stocks))
            
            # Submit index downloads  
            if indices:
                future_indices = executor.submit(self.download_indices_parallel, indices, years)
                futures.append(('indices', future_indices))
            
            # Submit fundamental downloads
            if fundamentals:
                future_fundamentals = executor.submit(self.download_fundamentals_parallel, fundamentals, years)
                futures.append(('fundamentals', future_fundamentals))
            
            # Collect results
            for data_type, future in futures:
                try:
                    result = future.result()
                    all_results[data_type] = result
                except Exception as e:
                    print(f"❌ {data_type} download failed: {e}")
        
        # Summary
        elapsed_time = time.time() - start_time
        total_success = (len(all_results['stocks']['results']) + 
                        len(all_results['indices']['results']) + 
                        len(all_results['fundamentals']['results']))
        total_errors = (len(all_results['stocks']['errors']) + 
                       len(all_results['indices']['errors']) + 
                       len(all_results['fundamentals']['errors']))
        
        print(f"\n⏱️  Parallel download completed in {elapsed_time:.1f} seconds")
        print(f"✅ Successful downloads: {total_success}")
        print(f"❌ Failed downloads: {total_errors}")
        
        return all_results
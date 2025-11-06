"""
Main script for downloading financial data
User can specify number of years via terminal input
"""

from downloaders.stock_prices import StockPrices
from downloaders.index_prices import IndexPrices
from downloaders.simfin_downloader import SimFinDownloader
from downloaders.parallel_downloader import ParallelDownloader
from portfolio.monte_carlo_runner import run_monte_carlo_simulation

def get_user_input():
    """Get user preferences from terminal"""
    print("=" * 50)
    print("FINANCIAL DATA DOWNLOADER")
    print("=" * 50)
    
    # Get number of years
    while True:
        try:
            years = int(input("Enter number of years of data to download: "))
            if years >= 1:
                break
            else:
                print("Please enter a positive number")
        except ValueError:
            print("Please enter a valid number")
    
    # Get data type
    print("\nSelect option:")
    print("1. Stock prices only")
    print("2. Index prices only") 
    print("3. Fundamental data only")
    print("4. All data types")
    print("5. All data types (PARALLEL)")
    print("6. Monte Carlo simulation")
    
    while True:
        try:
            choice = int(input("Enter choice (1-6): "))
            if 1 <= choice <= 6:
                break
            else:
                print("Please enter 1, 2, 3, 4, 5, or 6")
        except ValueError:
            print("Please enter a valid number")
    
    # Get symbols/tickers (skip for Monte Carlo)
    if choice == 6:
        return years, choice, [], [], []
    
    if choice in [1, 4, 5]:  # Stock prices
        stocks = input("Enter stock symbols (comma-separated, e.g., AAPL,MSFT,GOOGL): ").strip()
        stocks = [s.strip().upper() for s in stocks.split(',') if s.strip()]
    else:
        stocks = []
    
    if choice in [2, 4, 5]:  # Index prices
        print("Available indices: SPX, NDX, RUT, VIX, DJI")
        indices = input("Enter index symbols (comma-separated, e.g., SPX,NDX): ").strip()
        indices = [i.strip().upper() for i in indices.split(',') if i.strip()]
    else:
        indices = []
    
    if choice in [3, 4, 5]:  # Fundamental data
        if not stocks:
            fundamentals = input("Enter stock symbols for fundamentals (comma-separated): ").strip()
            fundamentals = [f.strip().upper() for f in fundamentals.split(',') if f.strip()]
        else:
            fundamentals = stocks
    else:
        fundamentals = []
    
    return years, choice, stocks, indices, fundamentals

def download_stock_prices(stocks, years):
    """Download stock price data"""
    if not stocks:
        return
    
    print(f"\n--- Downloading Stock Prices ({years} years) ---")
    downloader = StockPrices()
    
    for stock in stocks:
        try:
            print(f"Downloading {stock}...")
            df = downloader.download_historical_years(stock, years)
            print(f"✓ {stock}: {len(df)} days of data")
        except Exception as e:
            print(f"✗ {stock}: {e}")

def download_index_prices(indices, years):
    """Download index price data"""
    if not indices:
        return
    
    print(f"\n--- Downloading Index Prices ({years} years) ---")
    downloader = IndexPrices()
    
    for index in indices:
        try:
            print(f"Downloading {index}...")
            df = downloader.download_historical_index_years(index, years)
            print(f"✓ {index}: {len(df)} days of data")
        except Exception as e:
            print(f"✗ {index}: {e}")

def download_fundamentals(fundamentals, years):
    """Download fundamental data"""
    if not fundamentals:
        return
    
    print(f"\n--- Downloading Fundamental Data ({years} years) ---")
    downloader = SimFinDownloader()
    
    for ticker in fundamentals:
        try:
            print(f"Downloading {ticker} fundamentals...")
            results = downloader.download_all_statements(ticker, years)
            success_count = len([r for r in results.values() if r is not None])
            print(f"✓ {ticker}: {success_count}/3 statements downloaded")
        except Exception as e:
            print(f"✗ {ticker}: {e}")

def main():
    """Main function"""
    try:
        # Get user input
        years, choice, stocks, indices, fundamentals = get_user_input()
        
        print(f"\n{'='*50}")
        print(f"STARTING DOWNLOAD - {years} YEARS OF DATA")
        print(f"{'='*50}")
        
        # Execute based on choice
        if choice == 1:  # Stock prices only
            download_stock_prices(stocks, years)
        elif choice == 2:  # Index prices only
            download_index_prices(indices, years)
        elif choice == 3:  # Fundamental data only
            download_fundamentals(fundamentals, years)
        elif choice == 4:  # All data types (sequential)
            download_stock_prices(stocks, years)
            download_index_prices(indices, years)
            download_fundamentals(fundamentals, years)
        elif choice == 5:  # All data types (parallel)
            parallel_downloader = ParallelDownloader(max_workers=3)
            results = parallel_downloader.download_portfolio_parallel(stocks, indices, fundamentals, years)
            
            # Print summary
            print(f"\n📊 Parallel Download Summary:")
            for data_type, data in results.items():
                success_count = len(data['results'])
                error_count = len(data['errors'])
                if success_count > 0 or error_count > 0:
                    print(f"  {data_type.title()}: {success_count} success, {error_count} errors")
        elif choice == 6:  # Monte Carlo simulation
            run_monte_carlo_simulation()
            return  # Exit after simulation
        
        print(f"\n{'='*50}")
        print("DOWNLOAD COMPLETE!")
        print("Check the 'data' folder for your files.")
        print(f"{'='*50}")
        
    except KeyboardInterrupt:
        print("\n\nDownload cancelled by user.")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()
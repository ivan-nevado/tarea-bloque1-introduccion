import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from downloaders.stock_prices import StockPrices

def test_stock_prices():
    """Test StockPrices class"""
    print("=== Testing StockPrices ===")
    
    downloader = StockPrices()
    
    # Test single stock download
    try:
        print("Testing AAPL download...")
        df = downloader.download_and_save('AAPL', 'compact')
        print(f"✓ Success: {len(df)} days of AAPL data")
        print(f"  Date range: {df.index.min()} to {df.index.max()}")
        print(f"  Columns: {list(df.columns)}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test historical years download
    try:
        years = 1  # Test with 1 year to avoid API limits
        print(f"\nTesting {years} years of MSFT data...")
        df = downloader.download_historical_years('MSFT', years)
        print(f"✓ Success: {len(df)} days of MSFT data ({years} years)")
    except Exception as e:
        print(f"✗ Error: {e}")

if __name__ == "__main__":
    test_stock_prices()
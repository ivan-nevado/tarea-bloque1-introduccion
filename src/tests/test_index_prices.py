import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from downloaders.index_prices import IndexPrices

def test_index_prices():
    """Test IndexPrices class"""
    print("=== Testing IndexPrices ===")
    
    downloader = IndexPrices()
    
    # Test available indices
    print(f"Available indices: {list(downloader.INDEX_ETFS.keys())}")
    
    # Test single index download
    try:
        print("\nTesting SPX download...")
        df = downloader.download_and_save_index('SPX', 'compact')
        print(f"✓ Success: {len(df)} days of SPX data")
        print(f"  Date range: {df.index.min()} to {df.index.max()}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test historical years download
    try:
        print("\nTesting 1 year of NDX data...")
        df = downloader.download_historical_index_years('NDX', 1)
        print(f"✓ Success: {len(df)} days of NDX data (1 year)")
    except Exception as e:
        print(f"✗ Error: {e}")

if __name__ == "__main__":
    test_index_prices()
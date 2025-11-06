import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from downloaders.simfin_downloader import SimFinDownloader

def test_simfin():
    """Test SimFinDownloader class"""
    print("=== Testing SimFinDownloader ===")
    
    downloader = SimFinDownloader()
    
    # Test single company download
    try:
        print("Testing AAPL fundamental data...")
        results = downloader.download_all_statements('AAPL', 1)
        
        print(f"✓ Results for AAPL:")
        for statement_type, data in results.items():
            if data is not None:
                print(f"  {statement_type}: {data.shape}")
            else:
                print(f"  {statement_type}: No data")
                
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test multiple companies
    try:
        print("\nTesting multiple companies...")
        companies = ['AAPL']
        years = 1
        all_results = downloader.download_multiple_companies(companies, years)
        
        print(f"✓ Downloaded data for {len(all_results)} companies")
        
    except Exception as e:
        print(f"✗ Error: {e}")

if __name__ == "__main__":
    test_simfin()
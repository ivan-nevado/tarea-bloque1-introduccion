import requests
import time
import os
from dotenv import load_dotenv
import pandas as pd

class AlphaVantageBase:
    """Base class for Alpha Vantage API downloads"""
    
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.base_url = 'https://www.alphavantage.co/query'
        self.data_dir = "data"
        os.makedirs(self.data_dir, exist_ok=True)
        
        if not self.api_key:
            raise ValueError("ALPHA_VANTAGE_API_KEY not found in .env file")
    
    def make_request(self, params):
        """Make API request with rate limiting"""
        params['apikey'] = self.api_key
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API errors
            if 'Error Message' in data:
                raise Exception(f"API Error: {data['Error Message']}")
            
            if 'Note' in data:
                raise Exception(f"API Rate Limit: {data['Note']}")
            
            # Rate limiting - wait 12 seconds between calls
            time.sleep(12)
            
            return data
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request failed: {e}")
    
    def save_to_csv(self, data: pd.DataFrame, filename: str) -> str:
        """Save DataFrame to CSV file"""
        filepath = os.path.join(self.data_dir, filename)
        data.to_csv(filepath)
        return filepath
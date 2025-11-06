"""
Flexible data loader for any time series format
"""

import pandas as pd
import numpy as np
from .data_preprocessor import DataPreprocessor
from .portfolio import Portfolio
from typing import Dict, Optional, Union, List

def load_from_excel(file_path: str, 
                   sheet_name: Optional[str] = None,
                   price_column: Optional[str] = None,
                   date_column: Optional[str] = None) -> pd.DataFrame:
    """Load and process data from Excel file"""
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    return DataPreprocessor.process_any_format(df, price_column, date_column)

def load_from_json(file_path: str,
                  price_column: Optional[str] = None,
                  date_column: Optional[str] = None) -> pd.DataFrame:
    """Load and process data from JSON file"""
    df = pd.read_json(file_path)
    return DataPreprocessor.process_any_format(df, price_column, date_column)

def load_from_dict(data: dict,
                  price_column: Optional[str] = None,
                  date_column: Optional[str] = None) -> pd.DataFrame:
    """Load and process data from dictionary"""
    return DataPreprocessor.process_any_format(data, price_column, date_column)


"""
Plotting utilities for the Large Bayesian VAR model
Provides functions to format variable names for enhanced plot readability
"""

import sys
import os

# Add the config directory to the path to import variables
current_dir = os.path.dirname(__file__)
if current_dir not in sys.path:
    sys.path.append(current_dir)

from variables import FRED_VARIABLES

def format_variable_name(variable_code: str) -> str:
    """
    Format variable name for plot titles by combining code with descriptive name
    
    Parameters:
    -----------
    variable_code : str
        The FRED variable code (e.g., 'GDPC1', 'CPIAUCSL')
        
    Returns:
    --------
    str
        Formatted name in the format "CODE (Descriptive Name)"
        If variable not found in dictionary, returns original code
        
    Examples:
    ---------
    >>> format_variable_name('GDPC1')
    'GDPC1 (Real Gross Domestic Product)'
    >>> format_variable_name('UNKNOWN')
    'UNKNOWN'
    """
    if variable_code in FRED_VARIABLES:
        descriptive_name = FRED_VARIABLES[variable_code]['name']
        return f"{variable_code} ({descriptive_name})"
    else:
        # Fallback to original code if not found in dictionary
        return variable_code

def format_variable_names(variable_codes: list) -> list:
    """
    Format multiple variable names for plot labels
    
    Parameters:
    -----------
    variable_codes : list
        List of FRED variable codes
        
    Returns:
    --------
    list
        List of formatted variable names
    """
    return [format_variable_name(code) for code in variable_codes]

def get_variable_description(variable_code: str) -> str:
    """
    Get only the descriptive name of a variable
    
    Parameters:
    -----------
    variable_code : str
        The FRED variable code
        
    Returns:
    --------
    str
        Descriptive name only, or original code if not found
    """
    if variable_code in FRED_VARIABLES:
        return FRED_VARIABLES[variable_code]['name']
    else:
        return variable_code

def get_variable_units(variable_code: str) -> str:
    """
    Get the units for a variable
    
    Parameters:
    -----------
    variable_code : str
        The FRED variable code
        
    Returns:
    --------
    str
        Units description, or empty string if not found
    """
    if variable_code in FRED_VARIABLES:
        return FRED_VARIABLES[variable_code].get('units', '')
    else:
        return ''
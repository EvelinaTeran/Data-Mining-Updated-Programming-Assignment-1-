"""
   Use to create your own functions for reuse 
   across the assignment

   Inside part_1_template_solution.py, 
  
     import new_utils
  
    or
  
     import new_utils as nu
"""
import numpy as np

def scale_data(data):
    if not isinstance(data, np.ndarray):
        raise ValueError("Input data must be a numpy array")
    
    if not np.issubdtype(data.dtype, np.number):
        raise ValueError("Data matrix must contain numerical values")
        
    # Check if all elements are floating point numbers and if not then returns False
    if not np.issubdtype(data.dtype, np.floating):
        return False, data
    
    # Scale the data to be between 0 and 1
    min_val = np.min(data)
    max_val = np.max(data)
    
    if min_val < 0 or max_val > 1:
        data_scaled = (data - min_val) / (max_val - min_val)
    else:
        data_scaled = data
        
    # Check if all elements are scaled between 0 and 1
    if np.min(data_scaled) < 0 or np.max(data_scaled) > 1:
        return False, data_scaled
    
    return True, data_scaled
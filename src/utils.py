
import os
import matplotlib.pyplot as plt

def save_plot(fig, filename):
    """
    Saves a matplotlib figure to the reports/figures directory.
    Returns the absolute path of the saved file.
    """
    
    # Define path
    # Go up from src to root, then into reports/figures
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_dir = os.path.join(base_dir, "reports", "figures")
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    full_path = os.path.join(target_dir, filename)
    fig.savefig(full_path, bbox_inches='tight', dpi=300)
    return full_path

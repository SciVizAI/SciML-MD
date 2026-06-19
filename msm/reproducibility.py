"""
Reproducibility utilities for pipeline runs.

Saves run configurations and seeds for complete reproducibility.
"""
import json
import datetime
from pathlib import Path
import numpy as np


def save_run_config(output_path, config_dict):
    """
    Save run configuration to JSON file.
    
    Args:
        output_path: Path to save config
        config_dict: Dictionary of configuration parameters
    """
    # Add timestamp and version info
    run_info = {
        'timestamp': datetime.datetime.now().isoformat(),
        'config': config_dict
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(run_info, f, indent=2)


def load_run_config(config_path):
    """
    Load run configuration from JSON file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        config_dict: Configuration dictionary
    """
    with open(config_path) as f:
        run_info = json.load(f)
    return run_info['config']


def set_global_seed(seed):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Master random seed
    """
    np.random.seed(seed)
    # Add other library seeds as needed
    try:
        import random
        random.seed(seed)
    except:
        pass


def generate_seed_sequence(master_seed, n_seeds):
    """
    Generate a sequence of random seeds from a master seed.
    
    Args:
        master_seed: Master seed
        n_seeds: Number of seeds to generate
        
    Returns:
        seeds: List of random seeds
    """
    rng = np.random.RandomState(master_seed)
    return rng.randint(0, 2**31 - 1, size=n_seeds).tolist()

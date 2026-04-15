"""Global parameters for the entire project."""
import numpy as np

# Fixed parameters for all analyses
M_FIXED = 4          # Embedding dimension
TAU_FIXED = 15       # Time delay (samples)
TR = 2.0             # Repetition time (seconds)

# Random seed for reproducibility
RANDOM_SEED = 42

# Set global seed
np.random.seed(RANDOM_SEED)

# Color palette for professional plots
COLORS = {
    'primary': ['#2E5EAA', '#5B8C5A', '#D4A76A', '#C1666B', '#4A4A4A'],
    'categorical': ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3'],
    'sequential': ['#f7f7f7', '#d9d9d9', '#bdbdbd', '#969696', '#737373', '#525252', '#252525']
}
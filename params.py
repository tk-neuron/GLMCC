"""
*** YOU NEED TO EDIT THIS FILE WHEN RUNNING THE CODE. ***
this file defines constant parameters
"""

# --- required to customize --- #
DATA_DIR = './datasets/sample15/'  # path to dataset
PREFIX = 'cell'  # file name prefix of dataset
EXTENSION = 'txt'  # file extension of dataset
SAVE_DIR = './results/sample15/'  # save directory of results

# --- optionally customizable parameters --- #
START = 0.0  # start time [ms]
END = 5400.0 * 1000  # end time [ms]
WINDOW = 50.0  # time window when making cross-correlogram
SYNAPTIC_DELAY = [1.0, 2.0, 3.0, 4.0, 5.0]  # synaptic delay range for which GLMCC parameters are optimized
FIRING_RATE_THRESHOLD = 0.5  # firing rate threshold when loading spike trains (to avoid combinatorial explosion
# and low estimation accuracy)

# for glmplot
PLOT_REFERENCE_ID = 1  # reference neuron's id when plotting cc and GLM
PLOT_TARGET_ID = 6  # target neuron's id

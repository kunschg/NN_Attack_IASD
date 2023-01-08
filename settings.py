"""
Module dertining the general variables
"""

import sys
import torch
import subprocess

# defining which device to use, darwin=Mac
if sys.platform=="darwin":
        try:
            device = torch.device("mps") # GPU acceleration for Mac
        except:
            device="cpu"
else:
    device = 'cuda' if torch.cuda.is_available() else 'cpu' # Cuda for GPU
print(f"Will be using device: {device}")
batch_size=64
#storing the root of the git repo
git_dir = subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8')
PGD_EPS = 1e-2
PGD_NB_ITERS = 40
PGD_STEPSIZE = 3/255
PGD_TRAIN_ALPHA = 0.9
FINITE_DIFF_EPS = 1e-5
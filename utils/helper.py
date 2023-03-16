import pickle
from os import listdir

import matplotlib as mpl
import matplotlib.animation
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import mne
import numpy as np
from IPython.display import HTML
from matplotlib.animation import PillowWriter
from numpy.linalg import norm
from tqdm import tqdm, trange

cmap = mpl.colormaps["Set2"].colors


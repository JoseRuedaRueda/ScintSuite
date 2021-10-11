"""Magnetic coils data"""
import Lib.LibPaths as lpath
import numpy as np
import os
from tqdm import tqdm
from scipy.interpolate import interp1d
import requests
import re
import shutil

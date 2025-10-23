from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging
from pathlib import Path
from eda_pipeline import run_eda

warnings.filterwarnings('ignore')

# logging and file setups
base_dir = Path.cwd()
logs_dir = base_dir / 'logs'

Path('logs').mkdir(exist_ok=True)
log_path = logs_dir / 'preprocessing_pipeline.log'

logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s : %(message)s',
        datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)


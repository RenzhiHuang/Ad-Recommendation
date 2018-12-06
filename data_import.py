import pandas as pd
import numpy as np

def data_import(path):
	raw_data = pd.read_csv(path, header = None)
	raw_data = raw_data.values
	return raw_data
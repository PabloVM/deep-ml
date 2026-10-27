import numpy as np
def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
	mean = np.mean(data,axis=0)
	std = np.std(data,axis=0)
	std = np.where(std==0,1,std)
	standardized_data = np.round((data - mean) / std,4)
	normalized_data = np.round((data - np.min(data,axis=0)) / (np.max(data,axis=0) - np.min(data,axis=0)),4)
	return standardized_data.tolist(), normalized_data.tolist()
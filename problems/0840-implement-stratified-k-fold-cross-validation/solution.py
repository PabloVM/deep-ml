import numpy as np

def stratified_kfold_indices(y, n_splits):
	"""
	Generate train/test indices for stratified K-fold cross-validation.

	Args:
		y: 1D array-like of integer class labels
		n_splits: number of folds

	Returns:
		A list of [train_indices, test_indices] pairs, one per fold.
	"""
	y = np.array(y)
	class_idx = [np.where(y==c)[0] for c in np.unique(y)]

	folds = [[[],[]] for _ in range(n_splits)]


	for c in class_idx:
		start = 0
		for fold in range(n_splits):
			size = len(c) // n_splits + (1 if fold < len(c) % n_splits else 0)
			end = start + size

			test = c[start:end]
			folds[fold][0].append(test)
			train = np.concatenate([c[:start],c[end:]])
			folds[fold][1].append(train)
			start=end
	splits = []
	for fold in folds:
		splits.append([np.sort(np.concatenate(fold[1])).tolist(),np.sort(np.concatenate(fold[0])).tolist()])
			
	return splits
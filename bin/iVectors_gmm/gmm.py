import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn import mixture
from sklearn.metrics import roc_auc_score
from Keras_model import *
from Denoise_AE import *

def get_machine_ids(machines, mode):
	mid_dict = {}
	if mode == 'd':
		path = '../../saved_iVectors/ivector_mfcc_100'
		folder = 'test'
		for m in machines:
			file_list = os.listdir(os.path.join(path, m, folder))
			id_list = list(set([int(file.split('_')[2]) for file in file_list]))
			mid_dict[m] = id_list
	elif mode == 'e':
		path = '../../saved_iVectors/ivector_mfcc_100'
		folder = 'test_eval'
		for m in machines:
			file_list = os.listdir(os.path.join(path, m, folder))
			id_list = list(set([int(file.split('_')[1]) for file in file_list]))
			mid_dict[m] = id_list
	return mid_dict

def read_train(m, mid, mode):
	X = []
	if mode == 'd':
		path = '../../saved_iVectors/ivector_mfcc_100/{}/train/'.format(m)
	elif mode == 'e':
		path = '../../saved_iVectors/ivector_mfcc_100/{}/train_eval/'.format(m)
	files = os.listdir(path)
	files = [f for f in files if int(f.split('_')[2]) == mid]
	files.sort()
	for f in files:
		iv = pd.read_csv(path + f, names = ['iv'])
		X.append(list(iv['iv']))
	# X = TRAIN_DENOISE(np.array(X))
	# X = get_model(X)
	return np.array(X)

def read_test(m, mid, mode):
	X, y = [], []
	if mode == 'd':
		path = '../../saved_iVectors/ivector_mfcc_100/{}/test/'.format(m)
		files = os.listdir(path)
		files = [f for f in files if int(f.split('_')[2]) == mid]
		norm_files = [f for f in files if f[0] == 'n']
		anom_files = [f for f in files if f[0] == 'a']
		norm_files.sort()
		anom_files.sort()
		for f in norm_files:
			iv = pd.read_csv(path + f, names = ['iv'])
			X.append(list(iv['iv']))
			y.append(0)
		for f in anom_files:
			iv = pd.read_csv(path + f, names = ['iv'])
			X.append(list(iv['iv']))
			y.append(1)
		# X = TRAIN_DENOISE(np.array(X))
		# y = TRAIN_DENOISE(np.array(y))
		# X = get_model(X)
		# return X,y
		return np.array(X), y
	elif mode == 'e':
		path = '../../saved_iVectors/ivector_mfcc_100/{}/test_eval/'.format(m)
		files = os.listdir(path)
		files = [f for f in files if int(f.split('_')[1]) == mid]
		files.sort()
		for f in files:
			iv = pd.read_csv(path + f, names = ['iv'])
			X.append(list(iv['iv']))
		files = [f[:-4]+'.wav' for f in files]
		# X = TRAIN_DENOISE(np.array(X))
		# X = get_model(X)
		return X, files

def GMM(X_train, X_test, y_test):
	# X_train = TRAIN_DENOISE(X_train,X_test)
	# X_train = get_model(X_train,X_test)
	# clf = mixture.GaussianMixture(n_components = 10, covariance_type='full', random_state = 42).fit(X_train)
	# y_pred = clf.score_samples(X_test)
	scaler = preprocessing.StandardScaler().fit(X_train)
	X_train_transformed = scaler.transform(X_train)
	clf = svm.SVC(C=1).fit(X_train_transformed, X_train)
	X_test_transformed = scaler.transform(X_test)
	y_pred= clf.score(X_test_transformed, y_test)
	y_pred_iv = -1 * y_pred
	return y_pred_iv


def main(mode):

	machines = [
		'ToyCar', 'ToyConveyor', 'fan',
		'pump'
		, 'slider', 'valve'
	]

	if mode == 'd':

		mid_dict = get_machine_ids(machines, mode)
		anom_scores_ensemble = {}
		results = {'Machine':[], 'Mid':[], 'AUC':[], 'pAUC':[]}

		for m in machines:

			anom_scores_ensemble[m] = {}
			avg  = {'AUC':[], 'pAUC':[]}

			for mid in mid_dict[m]:

				anom_scores_ensemble[m][mid] = {}

				# X_train_old = read_train(m, mid, mode)
				# X_train_dim = X_train_old.ndim
				# X_train =get_model(X_train_dim)
				# X_test_old, Y_test_old = read_test(m, mid, mode)
				# X_test_dim = X_test_old.ndim
				# Y_test_dim = Y_test_old.ndim
				# X_test = get_model(X_test_dim)
				# Y_test = get_model(X_test_dim)
				# X_train = np.array(X_train)
				# X_test = np.array(X_test)
				# print(X_train.shape)
				X_train = read_train(m, mid, mode)
				X_test, y_test = read_test(m, mid, mode)
				# X_train,Y_train = TRAIN_DENOISE(X_train)
				# X_test = TRAIN_DENOISE(np.array(X_test))
				# y_test = TRAIN_DENOISE(np.array(y_test))
				y_pred_iv = GMM(X_train, X_test, y_test)

				AUC = roc_auc_score(y_test, y_pred_iv)
				pAUC = roc_auc_score(y_test, y_pred_iv, max_fpr = 0.1)
				anom_scores_ensemble[m][mid]['iv'] = y_pred_iv

				results['Machine'].append(m)
				results['Mid'].append(mid)
				results['AUC'].append(AUC)
				results['pAUC'].append(pAUC)

				avg['AUC'].append(AUC)
				avg['pAUC'].append(pAUC)

			results['Machine'].append(m)
			results['Mid'].append('Average')
			results['AUC'].append(np.mean(avg['AUC']))
			results['pAUC'].append(np.mean(avg['pAUC']))

		results = pd.DataFrame(results)
		results.to_csv('iVectors_gmm_dev_data_results.csv')
		print(results)
		with open('../ensemble/individual_scores/dev/iVectors_gmm_dev_data.pickle', 'wb') as file:
			pickle.dump(anom_scores_ensemble, file)

	elif mode == 'e':

		mid_dict = get_machine_ids(machines, mode)
		anom_scores_ensemble = {}

		for m in machines:

			anom_scores_ensemble[m] = {}

			for mid in mid_dict[m]:

				anom_scores_ensemble[m][mid] = {}
				anom_scores = {'file':[], 'anomaly_score':[]}

				# X_train_old = read_train(m, mid, mode)
				# X_train_dim = X_train_old.ndim
				# X_train = get_model(X_train_dim)
				# X_test_old, eval_files_old = read_test(m, mid, mode)
				# X_test_dim = X_test_old.ndim
				# X_test = get_model(X_test_dim)
				# eval_files = get_model(X_test_dim)
				X_train = read_train(m, mid, mode)
				X_test, eval_files = read_test(m, mid, mode)

				y_pred_iv = GMM(X_train, X_test)
				anom_scores['file'] = eval_files
				anom_scores['anomaly_score'] = y_pred_iv
				anom_scores_ensemble[m][mid]['iv'] = y_pred_iv

				submission_file = pd.DataFrame(anom_scores)
				submission_file.to_csv('../../task2/Tiwari_IITKGP_task2_1/anomaly_score_{}_id_0{}.csv'.format(m, mid), header = False, index = False)

		with open('../ensemble/individual_scores/eval/iVectors_gmm_eval_data.pickle', 'wb') as file:
			pickle.dump(anom_scores_ensemble, file)

if __name__ == '__main__':
	n = len(sys.argv)
	if n < 2:
		print('Please enter dev/eval mode')
	mode = sys.argv[1]

	if mode == 'd' or mode == 'e':
		main(mode)
	else:
		print('Invalid mode')

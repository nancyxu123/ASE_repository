####################################################
####################################################
# COMS 4771 Machine Learning Assignment 5 Problem 2
# Name:
# UNI: 
####################################################
import pickle
import numpy as np

def my_linear_kernel(X, Y):
        return np.dot(X, Y.T)

def predictSVM(X_test):
	"""This function takes a dataset and predict the letter for each data point.

	Parameters
	----------
	dataset: M X 128 numpy array
		A dataset represented by numpy-array

	Returns
	-------
	M x 1 numpy array
		Returns a numpy array of letter that each data point represents
	"""
	# Your code here
	clf2 = pickle.load(open('linearkernelmodel.pkl', 'rb'))
	predictions = clf2.predict(X_test)
	return predictions
                

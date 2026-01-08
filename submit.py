import numpy as np
import sklearn
from sklearn.metrics.pairwise import polynomial_kernel

# You are allowed to import any submodules of sklearn e.g. metrics.pairwise to construct kernel Gram matrices
# You are not allowed to use other libraries such as scipy, keras, tensorflow etc

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_kernel, my_decode etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here

################################
# Non Editable Region Starting #
################################
def my_kernel( X1, Z1, X2, Z2 ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to compute Gram matrices for your proposed kernel
	# Your kernel matrix will be used to train a kernel ridge regressor
  poly = polynomial_kernel(Z1,Z2,degree=3,coef0 = 1)
  X_comb = X1 @ X2.T
  G = X_comb * poly + 1
  return G


################################
# Non Editable Region Starting #
################################
def my_decode( w ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to invert a PUF linear model to get back delays
	# w is a single 1089-dim vector (last dimension being the bias term)
	# The output should be eight 32-dimensional vectors
	
 # Step 1: reshape to 33×33
	M = w.reshape(33, 33)

	# Step 2: rank-1 factorization via SVD
	U, S, Vt = np.linalg.svd(M, full_matrices=False)
	root = np.sqrt(S[0])
	w1 = root * U[1:, 0]
	w2 = root * Vt[0, 1:]

	# Step 4: build A using broadcasting (faster than kron)
	base = np.array([1, -1, -1, 1])
	idx = np.arange(32)
	mask = idx[:, None] >= idx[None, :]      # True where j <= i
	A = mask.repeat(4, axis=1) * np.tile(base, 32)

	# Step 5: fast pseudoinverse solve using SVD(A)
	Ua, Sa, VaT = np.linalg.svd(A, full_matrices=False)

	# x = V * (U^T * b / S)
	temp1 = Ua.T @ w1
	temp2 = Ua.T @ w2
	temp1 = temp1 / Sa
	temp2 = temp2 / Sa

	d1 = VaT.T @ temp1
	d2 = VaT.T @ temp2

	# Step 6: non-negativity
	d1 = np.maximum(d1, 0)
	d2 = np.maximum(d2, 0)

	# Step 7: vectorized splitting
	a, b, c, d = d1.reshape(32, 4).T
	p, q, r, s = d2.reshape(32, 4).T


	return a, b, c, d, p, q, r, s

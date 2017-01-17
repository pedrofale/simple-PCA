import numpy as np
import settings as globalv

def centerData(data):
	print '\nCentering data...'
	mean_vector = np.mean(data, axis = 1).reshape(data.shape[0], 1)
	centered_data = data - mean_vector

	if (globalv.print_results):
		print 'Mean vector: \n', mean_vector
		print 'Centered data: \n', centered_data
	return centered_data, mean_vector

def correlationMatrix(centered_data):
	print '\nComputing correlation matrix...'
	D = globalv.D
	N = globalv.N

	correlation_matrix = np.zeros((D, D))
	for i in range(N):
		correlation_matrix += (centered_data[:,i].reshape(D,1)).dot(centered_data[:,i].reshape(D,1).T)
	correlation_matrix = correlation_matrix/N

	if (globalv.print_results):
		print correlation_matrix
	
	return correlation_matrix

def eigenDecomposition(corr):
	print '\nEigendecomposition of correlation matrix...'
	D = globalv.D
	N = globalv.N

	eigvals, eigvecs = np.linalg.eig(corr)

	for i in range(len(eigvals)):
	    eigv = eigvecs[:,i].reshape(1,D).T
	    np.testing.assert_array_almost_equal(corr.dot(eigv), eigvals[i] * eigv, decimal=6, err_msg='', verbose=True)

	if (globalv.print_results):	
		print 'Eigenvalues: \n', eigvals
		print 'Eigenvectors: \n', eigvecs 
	
	return eigvals, eigvecs

def readUserNumComponents(msg):	
	D = globalv.D
	N = globalv.N

	numComponents = raw_input(msg)
	try:
	    # Try to convert the user input to an integer
	    numComponents = int(numComponents)
	    if numComponents > D or numComponents < 0:
	    	print 'Invalid number of components'
	    	sys.exit(-3)
	except ValueError:
		# Catch the exception if the input was not a number
		numComponents = 1
	print 'Using %d components' %numComponents
	return numComponents

def computeProjectionMatrix(eigvals, eigvecs, r):
	print '\nComputing the projection matrix...'
	D = globalv.D
	N = globalv.N

	# List of (eigenvalue, eigenvector) tuples
	eig_pairs = [(np.abs(eigvals[i]), eigvecs[:,i]) for i in range(len(eigvals))]

	# Sort the (eigenvalue, eigenvector) tuples from high to low (reverse=True)
	eig_pairs.sort(key=lambda tup: tup[0], reverse=True)

	# Projection matrix
	proj_matrix = np.zeros((D,r))
	for i in range(r):
		proj_matrix[:,i] = eig_pairs[i][1].reshape(1, D)

	if (globalv.print_results):
		print 'Projection matrix: \n', proj_matrix
	
	return proj_matrix

def computePrincipalComponents(centered_data, proj_matrix):
	print '\nComputing principal components...'
	p = (proj_matrix.T).dot(centered_data)

	if (globalv.print_results):	
		print 'Principal components: \n', p
	
	return p

def reconstructData(princ_comps, proj_matrix, mean_vector):
	print '\nReconstructing the data...'
	reconstructed_data = (proj_matrix).dot(princ_comps) + mean_vector
	
	if (globalv.print_results):
		print 'Reconstructed data: \n', reconstructed_data
	
	return reconstructed_data

def computeRecError(data, reconstructed_data):
	print '\nComputing reconstruction error...'
	D = globalv.D
	N = globalv.N

	error = data - reconstructed_data
	error_var = 0
	for i in range(N):
		error_var += np.linalg.norm(error[:,i])**2
	error_var = error_var/N

	if (globalv.print_results):
		print 'Error: \n', error
	print 'Error variance: ', error_var
	
	return error
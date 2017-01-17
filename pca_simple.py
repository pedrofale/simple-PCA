import numpy as np
import sys
import pca_utils as pca
import settings as globalv

def readDataFromFile():
	print '\nReading data...'
	data = 0

	if len(sys.argv) > 1:
		fname = sys.argv[1]
		try:
			data = np.loadtxt(fname)
		except IOError:
			print 'Error reading data from', fname
			sys.exit(-1)
	else:
		print 'Usage: python', sys.argv[0], '[filename].txt'
		sys.exit(-2)

	# Shape data as DxN, D dimensions, N patterns
	data = data.T
	print 'Data: \n', data

	globalv.D = data.shape[0]
	globalv.N = data.shape[1]

	return data

def main():
	globalv.init(True)
	data = readDataFromFile()
	centered_data, mean_vector = pca.centerData(data)
	corr = pca.correlationMatrix(centered_data)
	eigvals, eigvecs = pca.eigenDecomposition(corr)
	r = pca.readUserNumComponents('\nHow many components (0 <= int <= %d)?\n' %globalv.D)
	proj_matrix = pca.computeProjectionMatrix(eigvals, eigvecs, r)
	princ_comps = pca.computePrincipalComponents(centered_data, proj_matrix)
	reconstructed_data = pca.reconstructData(princ_comps, proj_matrix, mean_vector)
	error = pca.computeRecError(data, reconstructed_data)

if __name__ == "__main__":
    main()
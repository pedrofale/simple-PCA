import numpy as np 
from PIL import Image
import settings as globalv
import pca_utils as pca
from matplotlib import pyplot as plt

globalv.init(False)

globalv.N = 14
N = globalv.N
imgsize = 50
globalv.D = imgsize**2
D = globalv.D

img = np.zeros((imgsize,imgsize))
data = np.zeros((D,N))

plt.figure(1)
for i in range(N):
	img = np.asarray(Image.open('faces/' + str(i+1) +'.jpg').convert('L'))
	data[:,i] = np.ravel(img)

	plt.subplot(2,N,i+1)
	plt.axis('off')
	plt.imshow(img, cmap='gray', interpolation='none')


# The loaded images form a data vector of dimensions DxN, D=76^2, N = 6
# that is, 6 patterns of 76^2-dimensional data
# Because there are less patterns than dimensions, the maximum number of uncorrelated
# components we can keep is 6-1 = 5, because there are no more directions of data variability

# Do PCA:
centered_data, mean_vector = pca.centerData(data)
corr = pca.correlationMatrix(centered_data)
eigvals, eigvecs = pca.eigenDecomposition(corr)
r = pca.readUserNumComponents('\nHow many components (0 <= int <= %d)?\n' %D)
proj_matrix = pca.computeProjectionMatrix(eigvals, eigvecs, r)
princ_comps = pca.computePrincipalComponents(centered_data, proj_matrix)

reconstructed_data = pca.reconstructData(princ_comps, proj_matrix, mean_vector)

for i in range(N):
	rec_data = reconstructed_data[:,i].reshape(imgsize, imgsize)
	plt.subplot(2,N,N+i+1)
	plt.axis('off')
	plt.imshow(rec_data, cmap='gray', interpolation='none')

plt.show()

error = pca.computeRecError(data, reconstructed_data)
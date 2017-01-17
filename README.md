# simple-PCA

This repository serves as a way of complementing my study of the workings of Principal Component Analysis (PCA). 

## `pca_simple.py` 
A script that reads a .txt file which contains data. Each data pattern is assumed to be a line in the file. The different dimensions of the data are assumed to be separated by a space or a tab. `data.txt` is an example data file.
The script will then perform PCA on the read data and print out the results from each computation, as such:

1. The read data;
2. The mean of the data and the centered data;
3. The empirical correlation matrix of the centered data; 
4. The eigenvalues and corresponding eigenvectors of the correlation matrix;
5. The number of components to keep (received as user input);
6. The projection matrix (the *r* eigenvectors with highest eigenvalue);
7. The resulting principal components (the projection of the data on the lower-dimensional subspace spanned by the columns of the projection matrix);
8. The reconstructed data;
9. The reconstruction error and its variance.

Usage: `python pca_simple.py [filename].txt`


## `image_pca.py` 
A script that uses the same functions as `pca_simple.py` but applies them to image compression. It is setup to load .jpg images from the `/faces/` folder.

Usage: `python image_pca.py`

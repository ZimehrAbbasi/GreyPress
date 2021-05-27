##
#
# @author: Zimehr Abbasi
# @date: 2021
# A basic compressor which uses singular value decomposition to compress image files
#
##

from PIL import Image
import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt
from matplotlib.image import imsave


class LinearPress:

    def __init__(self, image_dir, eigenvalues_to_keep=None):
        self.image_dir = image_dir  # Image Directory
        self.image = None  # Stores the image file as ann array
        self.eigenvalues_to_keep = eigenvalues_to_keep  # number of eigenvalues to keep
        # array with the average of the RGB values instead of the tuple
        self.single_val_array = None
        self.rotate = False  # for functionality of the rotation
        self.final = None  # Compressed image

    '''
    This function takes in as parameters 2 vectors and calculates the projection of the first vector on the second vector
    v1: numpy array
    v2: numpy array
    '''

    def calc_projection(self, v1, v2):

        # Return 0 if the length is 0
        if(np.dot(v2, v2) == 0):
            return 0

        # Calculate the projection
        w = np.dot(v1, v2)/np.dot(v2, v2) * v2

        # return the projection
        return w

    '''
    This function takes in as a parameter a single vector and returns the norm of the vector
    vector: numpy array
    '''

    def norm(self, vector):

        # return if the norm is 0
        if len(vector) == 0:
            return 0

        # Calculate the square of the norm of the inputted vector
        running_sum = 0
        for x in range(len(vector)):
            running_sum += vector[x]**2

        # return norm of the vector
        return np.sqrt(running_sum)

    '''
    This fucntion takes in as parameters a eigenvector matrix and performs the gram Schmidt process on it, returning the orthogonal eigenvectors
    W: 2D numpy array
    '''

    def gram_schmidt(self, W):
        # Set the first vector to v
        v = [W[0]]
        for y in range(1, len(W)):
            vtemp = W[y]
            temp = 0

            # projection calculations for vector y
            for x in range(1, y+1):
                temp += self.calc_projection(vtemp, W[x-1])
            vtemp -= temp

            # Append the new orthonormal array to the
            v.append(vtemp)

        return np.array(v)

    '''
    This function takes in as a parameters an eigenvalue array and the corresponding eigenvector arrays and sorts them in decreasing order based on the eigenvalues
    A: numpy array
    B: 2D numpy array
    '''

    def same_sort(self, A, B):

        # Sort the eigenvectors and eigenvalues

        for x in range(len(A)):
            for y in range(len(A)):
                if(A[x] > A[y]):
                    A[x], A[y] = A[y], A[x]
                    for i in range(len(B)):
                        B[x][i], B[y][i] = B[y][i], B[x][i]

        return A, B

    '''
    This function is the main process of the entire compressor class for an RGB image
    '''

    def process3D(self):
        # The columns and rows for the original shape
        or_cols = self.single_val_array[0].shape[1]
        or_rows = self.single_val_array[0].shape[0]

        # columns and rows for the new shape
        cols = self.single_val_array[0].shape[1]
        rows = self.single_val_array[0].shape[0]

        # Checking if it is vertical
        if rows > cols:
            for i in range(3):
                self.single_val_array[i] = self.single_val_array[i].T
            cols = self.single_val_array[0].shape[1]
            rows = self.single_val_array[0].shape[0]
            self.rotate = True

        self.rgb = []

        for j in range(3):

            # Calculating A^T * A
            A_TA = np.matmul(
                self.single_val_array[j].T, self.single_val_array[j])

            # Find the eigenvalues and eigenvectors
            eigval_ATA, eigvec_ATA = LA.eig(A_TA)

            # Conduct Gram schmidt process
            eigvec_ATA = self.gram_schmidt(eigvec_ATA.T)

            # Sort the values
            eigval_ATA, eigvec_ATA = self.same_sort(eigval_ATA, eigvec_ATA)

            # initiate the U matrix
            U = []
            for i in range(rows):
                temp = np.matmul(self.single_val_array[j], eigvec_ATA[i])
                temp = temp/np.sqrt(eigval_ATA[i])
                U.append(temp)

            U = np.array(U)

            # Find the single values of the Sigma matrix
            for x in range(min(cols, rows)):
                try:
                    eigval_ATA[x] = np.sqrt(
                        eigval_ATA[x]) if eigval_ATA[x] >= 0 else eigval_ATA[x]
                except:
                    pass

            # Discard extra eigenvalues if necessary
            if(self.eigenvalues_to_keep < min(cols, rows)):
                for x in range(min(cols, rows)):
                    if x > self.eigenvalues_to_keep:
                        eigval_ATA[x] = 0

            # Instantiiate Sigma array
            Sigma = np.zeros((rows, cols))

            # Create Diagnal Sigma array with single values
            for x in range(min(cols, rows)):
                Sigma[x][x] = eigval_ATA[x]

            # Calculate the final vector matrix
            final_image = np.matmul(np.matmul(U.T, Sigma), eigvec_ATA)
            final_image = final_image.astype(np.int64)

            # Rotate if necessary
            if self.rotate:
                final_image = final_image.T

            self.rgb.append(final_image)

        # Create image matrix
        self.final = []
        for y in range(or_rows):
            mid = []
            for x in range(or_cols):
                mid.append([self.rgb[c][y][x] for c in range(3)])
            self.final.append(mid)

        # Convert matrix to array
        self.final = np.array(self.final)

    '''
    This function is the main process of the entire compressor class
    '''

    def process(self):

        # The columns and rows for the original shape
        or_cols = self.single_val_array.shape[1]
        or_rows = self.single_val_array.shape[0]

        # columns and rows for the new shape
        cols = self.single_val_array.shape[1]
        rows = self.single_val_array.shape[0]

        # Checking if it is vertical
        if rows > cols:
            self.single_val_array = self.single_val_array.T
            cols = self.single_val_array.shape[1]
            rows = self.single_val_array.shape[0]
            self.rotate = True

            # Calculating A^T * A
        A_TA = np.matmul(self.single_val_array.T, self.single_val_array)

        # Find the eigenvalues and eigenvectors
        eigval_ATA, eigvec_ATA = LA.eig(A_TA)

        # Conduct Gram schmidt process
        eigvec_ATA = self.gram_schmidt(eigvec_ATA.T)

        # Sort the values
        eigval_ATA, eigvec_ATA = self.same_sort(eigval_ATA, eigvec_ATA)

        # initiate the U matrix
        U = []
        for i in range(rows):
            temp = np.matmul(self.single_val_array, eigvec_ATA[i])
            temp = temp/np.sqrt(eigval_ATA[i])
            U.append(temp)

        U = np.array(U)

        # Find the single values of the Sigma matrix
        for x in range(min(cols, rows)):
            try:
                eigval_ATA[x] = np.sqrt(
                    eigval_ATA[x]) if eigval_ATA[x] >= 0 else eigval_ATA[x]
            except:
                pass

        # Discard extra eigenvalues if necessary
        if(self.eigenvalues_to_keep < min(cols, rows)):
            for x in range(min(cols, rows)):
                if x > self.eigenvalues_to_keep:
                    eigval_ATA[x] = 0

        # Instantiiate Sigma array
        Sigma = np.zeros((rows, cols))

        # Create Diagnal Sigma array with single values
        for x in range(min(cols, rows)):
            Sigma[x][x] = eigval_ATA[x]

        # Calculate the final vector matrix
        final_image = np.matmul(np.matmul(U.T, Sigma), eigvec_ATA)
        final_image = final_image.astype(np.int64)

        # Rotate if necessary
        if self.rotate:
            final_image = final_image.T

        # Create image matrix
        self.final = []
        for y in range(or_rows):
            mid = []
            for x in range(or_cols):
                mid.append([final_image[y][x] for c in range(3)])
            self.final.append(mid)

        # Convert matrix to array
        self.final = np.array(self.final)

    '''
    This function shows the original and the compressed image
    '''

    def show(self):

        # Create 2 subplots, for the original and for the compressed
        f, axarr = plt.subplots(2, 1)
        f.set_figheight(8)
        f.set_figwidth(8)
        axarr[0].set_title("Un-Compressed")
        axarr[0].imshow(self.image, interpolation='nearest')
        axarr[1].set_title("Compressed")
        axarr[1].imshow(self.final, interpolation='nearest')
        plt.show()

    '''
    This function begins the process of compression
    '''

    def start(self):
        # Read file
        self.image = np.asarray(Image.open(self.image_dir))

        # Save the width and the height of the image
        width = self.image.shape[0]
        height = self.image.shape[1]

        if len(self.image.shape) == 2:

            self.single_val_array = np.array([[[0 for y in range(height)]
                                               for x in range(width)]for i in range(3)])
            for x in range(width):
                for y in range(height):
                    self.single_val_array[x][y] = self.image[x][y]

            self.single_val_array = np.array(
                self.single_val_array).astype(np.float64)

            # Start the process
            self.process()

        else:

            # Instantiate the single value array
            self.single_val_array = np.array([[[0 for y in range(height)]
                                               for x in range(width)]for i in range(3)])

            # Set each value in the single value array to the average of the 3 RGB values in the image array
            for i in range(3):
                for x in range(width):
                    for y in range(height):
                        self.single_val_array[i][x][y] = self.image[x][y][i]

            # convert it into an array
            self.single_val_array = np.array(
                self.single_val_array).astype(np.float64)

            # Start the process
            self.process3D()


if __name__ == '__main__':
    to_keep = 25
    compressor = LinearPress("images/rgb.png", to_keep)
    compressor.start()
    compressor.show()

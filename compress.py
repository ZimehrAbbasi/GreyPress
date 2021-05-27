from PIL import Image
import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt


class Compressor:

    def __init__(self, image_dir, eigenvalues_to_keep=None):
        self.image_dir = image_dir
        self.image = None
        self.eigenvalues_to_keep = eigenvalues_to_keep
        self.single_val_array = None
        self.rotate = False
        self.final = None

    def calc_projection(self, v1, v2):

        if(np.dot(v2, v2) == 0):
            return 0

        w = np.dot(v1, v2)/np.dot(v2, v2) * v2
        return w

    def gram_schmidt(self, W):
        v = [W[0]]
        for y in range(1, len(W)):
            vtemp = W[y]
            temp = 0
            for x in range(1, y+1):
                temp += self.calc_projection(vtemp, W[x-1])
            vtemp -= temp
            v.append(vtemp)

        return np.array(v)

    def same_sort(self, A, B):
        for x in range(len(A)):
            for y in range(len(A)):
                if(A[x] > A[y]):
                    A[x], A[y] = A[y], A[x]
                    for i in range(len(B)):
                        B[x][i], B[y][i] = B[y][i], B[x][i]

        return A, B

    def process(self):
        or_cols = self.single_val_array.shape[1]
        or_rows = self.single_val_array.shape[0]
        cols = self.single_val_array.shape[1]
        rows = self.single_val_array.shape[0]

        if rows > cols:
            self.single_val_array = self.single_val_array.T
            cols = self.single_val_array.shape[1]
            rows = self.single_val_array.shape[0]
            self.rotate = True

        A_TA = np.matmul(self.single_val_array.T, self.single_val_array)
        eigval_ATA, eigvec_ATA = LA.eig(A_TA)
        eigvec_ATA = self.gram_schmidt(eigvec_ATA.T)
        eigval_ATA, eigvec_ATA = self.same_sort(eigval_ATA, eigvec_ATA)

        U = []
        for i in range(rows):
            temp = np.matmul(self.single_val_array, eigvec_ATA[i])
            temp = temp/np.sqrt(eigval_ATA[i])
            U.append(temp)

        U = np.array(U)

        for x in range(min(cols, rows)):
            try:
                eigval_ATA[x] = np.sqrt(
                    eigval_ATA[x]) if eigval_ATA[x] >= 0 else eigval_ATA[x]
            except:
                pass

        if(self.eigenvalues_to_keep < min(cols, rows)):
            for x in range(min(cols, rows)):
                if x > self.eigenvalues_to_keep:
                    eigval_ATA[x] = 0

        Sigma = np.zeros((rows, cols))

        for x in range(min(cols, rows)):
            Sigma[x][x] = eigval_ATA[x]

        final_image = np.matmul(np.matmul(U.T, Sigma), eigvec_ATA)
        final_image = final_image.astype(np.int64)

        if self.rotate:
            final_image = final_image.T

        self.final = []
        for y in range(or_rows):
            mid = []
            for x in range(or_cols):
                mid.append([final_image[y][x] for c in range(3)])
            self.final.append(mid)

        self.final = np.array(self.final)

    def show(self):

        plt.figure()
        # subplot(r,c) provide the no. of rows and columns
        f, axarr = plt.subplots(2, 1)
        axarr[0].set_title("Un-Compressed")
        axarr[0].imshow(self.image, interpolation='nearest')
        axarr[1].set_title("Compressed")
        axarr[1].imshow(self.final, interpolation='nearest')
        plt.show()

    def start(self):
        self.image = np.asarray(Image.open(self.image_dir))
        width = self.image.shape[0]
        height = self.image.shape[1]

        self.single_val_array = np.array([[0 for y in range(height)]
                                          for x in range(width)])

        for x in range(width):
            for y in range(height):
                self.single_val_array[x][y] = sum(self.image[x][y]) // 3

        self.single_val_array = np.array(
            self.single_val_array).astype(np.float64)

        self.process()


if __name__ == '__main__':
    compressor = Compressor("images/lion.png", 100)
    compressor.start()
    compressor.show()

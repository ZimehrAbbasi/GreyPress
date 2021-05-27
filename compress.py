from PIL import Image
import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt


def mult(v1, v2):

    if(np.dot(v2, v2) == 0):
        return 0

    w = np.dot(v1, v2)/np.dot(v2, v2) * v2
    return w


def gs(W):
    v = [W[0]]
    for y in range(1, len(W)):
        vtemp = W[y]
        temp = 0
        for x in range(1, y+1):
            temp += mult(vtemp, W[x-1])
        vtemp -= temp
        v.append(vtemp)

    return np.array(v)


def same_sort(A, B):
    for x in range(len(A)):
        for y in range(len(A)):
            if(A[x] > A[y]):
                A[x], A[y] = A[y], A[x]
                for i in range(len(B)):
                    B[x][i], B[y][i] = B[y][i], B[x][i]

    return A, B


image = Image.open('svdtest.png')
data = np.asarray(image)

width = data.shape[0]
height = data.shape[1]

image_single = np.array([[0 for y in range(height)]
                         for x in range(width)])

for x in range(width):
    for y in range(height):
        image_single[x][y] = sum(data[x][y])//3


print(image_single)
image_single = np.array(image_single).astype(np.float64)
cols = image_single.shape[1]
rows = image_single.shape[0]
# STEPS
# 1: Find A.T*A and A*A.T

A_TA = np.matmul(image_single.T, image_single)
AA_T = np.matmul(image_single, image_single.T)

# 2: Find the eigenvalues and eigenvectors for each

eigval_ATA, eigvec_ATA = LA.eig(A_TA)
# eigval_AAT, eigvec_AAT = LA.eig(AA_T)
print("Finding orthogonal Eigenvectors")
eigvec_ATA = gs(eigvec_ATA.T)
# eigvec_AAT = gs(eigvec_AAT.T)
print("Sorting eigenvalues")
eigval_ATA, eigvec_ATA = same_sort(eigval_ATA, eigvec_ATA)
# eigval_AAT, eigvec_AAT = same_sort(eigval_AAT, eigvec_AAT)

print("Finding U")
U = []
for i in range(len(eigvec_ATA)):
    temp = np.matmul(image_single, eigvec_ATA[i])
    temp = temp/np.sqrt(eigval_ATA[i])
    U.append(temp)

U = np.array(U)
# print(U.shape)

# for i in range(len(eigvec_ATA), rows):
#     temp = np.zeros((len(eigvec_ATA[0]), 1))
#     U.append(temp)

print("Removing eigenvalues")
# 3: Find which values to remove from eigenvalues
for x in range(min(cols, rows)):
    try:
        eigval_ATA[x] = np.sqrt(
            eigval_ATA[x]) if eigval_ATA[x] >= 0 else eigval_ATA[x]
    except:
        pass

av = sum(eigval_ATA)/max(cols, rows)
for x in range(min(cols, rows)):
    if(eigval_ATA[x] < av):
        eigval_ATA[x] = 0

    # 4: Create Diagonal matrix

Sigma = np.zeros((rows, cols))

for x in range(min(cols, rows)):
    Sigma[x][x] = eigval_ATA[x]


final_image = np.matmul(np.matmul(U.T, Sigma), eigvec_ATA)
final_image = final_image.astype(np.int64)

final = []
for y in range(rows):
    mid = []
    for x in range(cols):
        mid.append([final_image[y][x] for c in range(3)])
    final.append(mid)

final = np.array(final)

plt.figure()
# subplot(r,c) provide the no. of rows and columns
f, axarr = plt.subplots(2, 1)
axarr[0].set_title("Un-Compressed")
axarr[0].imshow(data, interpolation='nearest')
axarr[1].set_title("Compressed")
axarr[1].imshow(final, interpolation='nearest')
plt.show()

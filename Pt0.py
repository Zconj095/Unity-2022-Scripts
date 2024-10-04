# Load library
import numpy as np

# Create a vector as a row
vector_row = np.array([1, 2, 3])

# Create a vector as a column
vector_column = np.array([[1],
                          [2],
                          [3]])


# Load library
import numpy as np

# Create a matrix
matrix = np.array([[1, 2],
                   [1, 2],
                   [1, 2]])

matrix_object = np.mat([[1, 2],
                        [1, 2],
                        [1, 2]])

matrix([[1, 2],
        [1, 2],
        [1, 2]])

# Load libraries
import numpy as np
from scipy import sparse

# Create a matrix
matrix = np.array([[0, 0],
                   [0, 1],
                   [3, 0]])

# Create compressed sparse row (CSR) matrix
matrix_sparse = sparse.csr_matrix(matrix)

# View sparse matrix
print(matrix_sparse)

# Create larger matrix
matrix_large = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                         [3, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

# Create compressed sparse row (CSR) matrix
matrix_large_sparse = sparse.csr_matrix(matrix_large)

# View original sparse matrix
print(matrix_sparse)

# View larger sparse matrix
print(matrix_large_sparse)

# Load library
import numpy as np

# Generate a vector of shape (1,5) containing all zeros
vector = np.zeros(shape=5)

# View the matrix
print(vector)

array([0., 0., 0., 0., 0.])

# Generate a matrix of shape (3,3) containing all ones
matrix = np.full(shape=(3,3), fill_value=1)

# View the vector
print(matrix)

array([[1., 1., 1.],
       [1., 1., 1.],
       [1., 1., 1.]])

# Load library
import numpy as np

# Create row vector
vector = np.array([1, 2, 3, 4, 5, 6])

# Create matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Select third element of vector
vector[2]

# Select second row, second column
matrix[1,1]

# Select all elements of a vector
vector[:]
#array([1, 2, 3, 4, 5, 6])

# Select everything up to and including the third element
vector[:3]
#array([1, 2, 3])
# Select everything after the third element
vector[3:]

# Select the last element
vector[-1]

# Reverse the vector
vector[::-1]
#array([6, 5, 4, 3, 2, 1])

# Select the first two rows and all columns of a matrix
matrix[:2,:]
#array([[1, 2, 3],
#       [4, 5, 6]])

# Select all rows and the second column
matrix[:,1:2]

#array([[2],
#       [5],
#       [8]])


# Load library
import numpy as np

# Create matrix
matrix = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12]])

# View number of rows and columns
matrix.shape
#(3, 4)

# View number of elements (rows * columns)
matrix.size
#12
# View number of dimensions
matrix.ndim
#2

# Load library
import numpy as np

# Create matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Create function that adds 100 to something
add_100 = lambda i: i + 100

# Create vectorized function
vectorized_add_100 = np.vectorize(add_100)

# Apply function to all elements in matrix
vectorized_add_100(matrix)
#array([[101, 102, 103],
#       [104, 105, 106],
#       [107, 108, 109]])

# Add 100 to all elements
matrix + 100

#array([[101, 102, 103],
#       [104, 105, 106],
#       [107, 108, 109]])

# Load library
import numpy as np

# Create matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Return maximum element
np.max(matrix)

# Return minimum element
np.min(matrix)

# Find maximum element in each column
np.max(matrix, axis=0)

# Find maximum element in each row
np.max(matrix, axis=1)

# Load library
import numpy as np

# Create matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Return mean
np.mean(matrix)

# Return variance
np.var(matrix)

# Return standard deviation
np.std(matrix)

# Find the mean value in each column
np.mean(matrix, axis=0)

# Load library
import numpy as np

# Create 4x3 matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9],
                   [10, 11, 12]])

# Reshape matrix into 2x6 matrix
matrix.reshape(2, 6)

matrix.size

matrix.reshape(1, -1)

matrix.reshape(12)

# Load library
import numpy as np

# Create matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Transpose matrix
matrix.T

# Transpose vector
np.array([1, 2, 3, 4, 5, 6]).T

# Transpose row vector
np.array([[1, 2, 3, 4, 5, 6]]).T

# Load library
import numpy as np

# Create matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Flatten matrix
matrix.flatten()

matrix.reshape(1, -1)

# Create one matrix
matrix_a = np.array([[1, 2],
                     [3, 4]])

# Create a second matrix
matrix_b = np.array([[5, 6],
                     [7, 8]])

# Create a list of matrices
matrix_list = [matrix_a, matrix_b]

# Flatten the entire list of matrices
np.ravel(matrix_list)

# Load library
import numpy as np

# Create matrix
matrix = np.array([[1, 1, 1],
                   [1, 1, 10],
                   [1, 1, 15]])

# Return matrix rank
np.linalg.matrix_rank(matrix)

# Load library
import numpy as np

# Create matrix
matrix = np.array([[1, 2, 3],
                   [2, 4, 6],
                   [3, 8, 9]])

# Return diagonal elements
matrix.diagonal()

# Return diagonal one above the main diagonal
matrix.diagonal(offset=1)

# Return diagonal one below the main diagonal
matrix.diagonal(offset=-1)

# Load library
import numpy as np

# Create matrix
matrix = np.array([[1, 2, 3],
                   [2, 4, 6],
                   [3, 8, 9]])

# Return trace
matrix.trace()

# Return diagonal and sum elements
sum(matrix.diagonal())

# Load library
import numpy as np

# Create two vectors
vector_a = np.array([1,2,3])
vector_b = np.array([4,5,6])

# Calculate dot product
np.dot(vector_a, vector_b)

# Calculate dot product
vector_a @ vector_b

# Load library
import numpy as np

# Create matrix
matrix_a = np.array([[1, 1, 1],
                     [1, 1, 1],
                     [1, 1, 2]])

# Create matrix
matrix_b = np.array([[1, 3, 1],
                     [1, 3, 1],
                     [1, 3, 8]])

# Add two matrices
np.add(matrix_a, matrix_b)

# Subtract two matrices
np.subtract(matrix_a, matrix_b)

# Add two matrices
matrix_a + matrix_b

# Load library
import numpy as np

# Create matrix
matrix_a = np.array([[1, 1],
                     [1, 2]])

# Create matrix
matrix_b = np.array([[1, 3],
                     [1, 2]])

# Multiply two matrices
np.dot(matrix_a, matrix_b)

# Multiply two matrices
matrix_a @ matrix_b

# Multiply two matrices element-wise
matrix_a * matrix_b

# Load library
import numpy as np

# Create matrix
matrix = np.array([[1, 4],
                   [2, 5]])

# Calculate inverse of matrix
np.linalg.inv(matrix)

# Multiply matrix and its inverse
matrix @ np.linalg.inv(matrix)

# Load library
import numpy as np

# Set seed
np.random.seed(0)

# Generate three random floats between 0.0 and 1.0
np.random.random(3)

# Generate three random integers between 0 and 10
np.random.randint(0, 11, 3)

# Draw three numbers from a normal distribution with mean 0.0
# and standard deviation of 1.0
np.random.normal(0.0, 1.0, 3)

# Draw three numbers from a logistic distribution with mean 0.0 and scale of 1.0
np.random.logistic(0.0, 1.0, 3)

# Draw three numbers greater than or equal to 1.0 and less than 2.0
np.random.uniform(1.0, 2.0, 3)



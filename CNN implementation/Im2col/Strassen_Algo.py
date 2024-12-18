import cupy as cp 

def next_power_of_2(x):
    return 1 if x == 0 else 2**(x - 1).bit_length()

def pad_matrix(matrix, size):
    padded_matrix = cp.zeros((size, size))
    padded_matrix[:matrix.shape[0], :matrix.shape[1]] = matrix
    return padded_matrix

def split(matrix):
    row, col = matrix.shape
    row2, col2 = row//2, col//2
    return matrix[:row2, :col2], matrix[:row2, col2:], matrix[row2:, :col2], matrix[row2:, col2:]

def strassen(x, y):
    # Base case when size of matrices is 1x1
    if len(x) == 1:
        return x * y

    # Splitting the matrices into quadrants
    a, b, c, d = split(x)
    e, f, g, h = split(y)

    # Computing the 7 products, recursively (p1, p2...p7)
    p1 = strassen(a, f - h)  
    p2 = strassen(a + b, h)        
    p3 = strassen(c + d, e)        
    p4 = strassen(d, g - e)        
    p5 = strassen(a + d, e + h)        
    p6 = strassen(b - d, g + h)  
    p7 = strassen(a - c, e + f)  

    # Computing the values of the 4 quadrants of the final matrix c
    c11 = p5 + p4 - p2 + p6  
    c12 = p1 + p2           
    c21 = p3 + p4            
    c22 = p1 + p5 - p3 - p7  

    # Combining the 4 quadrants into a single matrix by stacking horizontally and vertically.
    c = cp.vstack((cp.hstack((c11, c12)), cp.hstack((c21, c22)))) 

    return c

def strassen_main(A, B):
    # Determine the size to pad the matrices to
    n = next_power_of_2(max(A.shape[0], A.shape[1], B.shape[0], B.shape[1]))
    
    # Pad the matrices
    A_padded = pad_matrix(A, n)
    B_padded = pad_matrix(B, n)

    # Perform Strassen multiplication
    C_padded = strassen(A_padded, B_padded)

    # Return the result trimmed to the original size
    return C_padded[:A.shape[0], :B.shape[1]]

# Example usage:
A = cp.random.randn(100, 100)
B = cp.random.randn(100, 100)
result = strassen_main(A, B)


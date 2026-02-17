import numpy as np

MAT_SIZE = 2000

def generate_and_save_data(N=MAT_SIZE, filename="data.txt"):
    M = np.random.randn(N, N)
    A = M.T @ M + np.eye(N) * 0.1
    # A = np.ones((N, N)) + np.eye(N)
    indices = np.arange(N)
    u = np.sin(2 * np.pi * indices / N)
    b = A @ u
    
    with open(filename, "w") as f:
        f.write(str(N) +"\n")
        for row in A:
            f.write(" ".join(map(str, row)) + "\n")
        for val in b:
            f.write(str(val) + "\n")
        for val in u:
            f.write(str(val) + "\n")
            
    return u


u_true = generate_and_save_data()
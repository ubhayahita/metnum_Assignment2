import numpy as np

A = np.array([[ 1.3, -0.3, -0.2, -0.4],
              [-0.6,  0.8,  0.0,  0.0],
              [-0.3,  0.0,  0.2,  0.0],
              [-0.1,  0.0,  0.0,  0.4]])
b = np.array([25.0, 0.0, 0.0, 0.0])

X0 = np.zeros(4)   # initial guess
TOL = 1e-6         # toleransi: max |x_baru - x_lama|
MAX_ITER = 100     # iterasi max

def build_system():
    return A.copy(), b.copy()
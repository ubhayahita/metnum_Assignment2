import numpy as np
from general import build_system, X0, TOL, MAX_ITER

def gauss_seidel(A, b, x0, tol, max_iter):
    n = len(b)
    x = x0.copy()
    hist = [] #history

    for k in range(1, max_iter + 1):
        x_old = x.copy()

        for i in range(n):
            s = sum(A[i, j] * x[j] for j in range(n) if j != i)   # x terbaru langsung dipakai
            x[i] = (b[i] - s) / A[i, i]

        err = np.max(np.abs(x - x_old))
        hist.append((k, x.copy(), err))

        if err < tol:
            return x, hist, True
    return x, hist, False

if __name__ == "__main__":
    A, b = build_system()

    print("GAUSS-SEIDEL METHOD")
    print("Initial guess:", X0)
    print("Tolerance    :", TOL)
    print("Iterasi max  :", MAX_ITER, "\n")
    print(f"{'k':>3} {'x1':>11} {'x2':>11} {'x3':>11} {'x4':>11} {'error':>12}")

    x, hist, ok = gauss_seidel(A, b, X0, TOL, MAX_ITER)

    for k, v, e in hist:
        print(f"{k:>3} {v[0]:>11.6f} {v[1]:>11.6f} {v[2]:>11.6f} {v[3]:>11.6f} {e:>12.3e}")
    print("\nKonvergen  :", "Ya" if ok else "Tidak", f"({len(hist)} iterasi)")
    print("Solusi akhir:", np.round(x, 6))
    print("Solusi eksak:", np.round(np.linalg.solve(A, b), 6))
import math

def f(L):
    return (
        100 * math.exp(-((L - 1) / 0.25)**2)
        + 10 * (math.exp(3 * (L - 1)) - 1)
        - 70
    )

L = 0.8
tolerance = 1e-6  # percent
max_iterations = 1000

print("Jacobi-style Fixed-Point Iteration")
print("k\tL\t\tf(L)\t\tRelative Error (%)")

for k in range(1, max_iterations + 1):
    old_L = L
    L = old_L - 0.005 * f(old_L)

    error = abs((L - old_L) / L) * 100

    print(f"{k}\t{L:.10f}\t{f(L):.6e}\t{error:.6e}")

    if error < tolerance:
        break

print("\nFinal solution:", L)
print("Iterations:", k)
print("Final residual:", f(L))

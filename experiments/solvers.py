"""Shared LASSO objective, data generator and NumPy solver implementations."""
import time

import numpy as np

TOL = 1e-4
MAX_ITER = 1200


def soft(v, t):
    return np.sign(v) * np.maximum(np.abs(v) - t, 0.0)


def objective(A, b, lam, x):
    r = A @ x - b
    return 0.5 * float(r @ r) + lam * float(np.abs(x).sum())


def pg_residual(A, b, lam, L, x, scale):
    grad = A.T @ (A @ x - b)
    prox = soft(x - grad / L, lam / L)
    return float(np.max(np.abs(L * (x - prox))) / scale)


def make_data(m, n, s, sigma, lam_factor, seed):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(m, n))
    A /= np.linalg.norm(A, axis=0)
    truth = np.zeros(n)
    support = rng.choice(n, s, replace=False)
    truth[support] = rng.choice([-1.0, 1.0], s) * rng.uniform(0.8, 1.2, s)
    b = A @ truth + rng.normal(scale=sigma, size=m)
    lam = lam_factor * float(np.max(np.abs(A.T @ b)))
    # Exact spectral norm for these deliberately modest dense instances.
    L = float(np.linalg.norm(A, ord=2) ** 2)
    scale = max(1.0, float(np.max(np.abs(A.T @ b))))
    return A, b, truth, lam, L, scale


def ista(A, b, lam, L, scale):
    x = np.zeros(A.shape[1]); At = A.T; Atb = At @ b
    start = time.perf_counter(); trace = []
    for k in range(1, MAX_ITER + 1):
        x = soft(x - (At @ (A @ x) - Atb) / L, lam / L)
        if k == 1 or k % 10 == 0 or k == MAX_ITER:
            res = pg_residual(A, b, lam, L, x, scale)
            trace.append((k, time.perf_counter() - start, objective(A, b, lam, x), res))
            if res <= TOL:
                break
    return x, trace, 0.0


def fista(A, b, lam, L, scale):
    x = np.zeros(A.shape[1]); y = x.copy(); t = 1.0
    At = A.T; Atb = At @ b
    start = time.perf_counter(); trace = []
    for k in range(1, MAX_ITER + 1):
        nxt = soft(y - (At @ (A @ y) - Atb) / L, lam / L)
        t_next = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        y = nxt + ((t - 1.0) / t_next) * (nxt - x)
        x, t = nxt, t_next
        if k == 1 or k % 10 == 0 or k == MAX_ITER:
            res = pg_residual(A, b, lam, L, x, scale)
            trace.append((k, time.perf_counter() - start, objective(A, b, lam, x), res))
            if res <= TOL:
                break
    return x, trace, 0.0


def admm(A, b, lam, L, scale):
    n = A.shape[1]; rho = 1.0
    start = time.perf_counter()
    Atb = A.T @ b
    # The inverse is reused for every iteration; its setup time is included.
    solve_matrix = np.linalg.inv(A.T @ A + rho * np.eye(n))
    setup = time.perf_counter() - start
    x = np.zeros(n); z = np.zeros(n); u = np.zeros(n); trace = []
    for k in range(1, MAX_ITER + 1):
        x = solve_matrix @ (Atb + rho * (z - u))
        old_z = z.copy()
        z = soft(x + u, lam / rho)
        u += x - z
        if k == 1 or k % 10 == 0 or k == MAX_ITER:
            res = pg_residual(A, b, lam, L, z, scale)
            primal = float(np.linalg.norm(x - z) / max(1.0, np.linalg.norm(z)))
            dual = float(rho * np.linalg.norm(z - old_z) / max(1.0, np.linalg.norm(Atb)))
            trace.append((k, time.perf_counter() - start, objective(A, b, lam, z), res, primal, dual))
            if res <= TOL and primal <= TOL and dual <= TOL:
                break
    return z, trace, setup



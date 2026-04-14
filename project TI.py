import numpy as np
import random
import matplotlib.pyplot as plt

# ======================================================
# ПАРАМЕТРЫ ПРОЕКТА (ТЗ)
# ======================================================
m = 7
n = 7
K = m * n

r = 70
s = 100
t = 70

seed0 = 123

# ======================================================
# ТЕСТОВЫЕ МАТРИЦЫ
# ======================================================
np.random.seed(42)
A = np.random.rand(r, s)
B = np.random.rand(s, t)
true_C = A @ B

# ======================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ======================================================
def frob_norm(X):
    return float(np.linalg.norm(X, ord="fro"))

def split_A_rows(A, m):
    assert A.shape[0] % m == 0
    return np.split(A, m, axis=0)

def split_B_cols(B, n):
    assert B.shape[1] % n == 0
    return np.split(B, n, axis=1)

def simulate_slow_nodes(N, S, rng):
    assert 0 <= S < N
    return set(rng.sample(range(N), S))

# ======================================================
# POLYNOMIAL ENCODING
# ======================================================
def encode_A(A_parts, x):
    out = np.zeros_like(A_parts[0], dtype=np.complex128)
    for i, Ai in enumerate(A_parts):
        out += Ai * (x ** i)
    return out

def encode_B(B_parts, x, m):
    out = np.zeros_like(B_parts[0], dtype=np.complex128)
    for j, Bj in enumerate(B_parts):
        out += Bj * (x ** (m * j))
    return out

# ======================================================
# ДЕКОДИРОВАНИЕ
# ======================================================
def build_vandermonde(xs, K):
    return np.vander(xs, N=K, increasing=True).astype(np.complex128)

def decode_coeffs(worker_values, worker_xs, K):
    # --- ПРОВЕРКИ МАСТЕРА ---
    assert len(worker_values) == K, f"Expected {K} values, got {len(worker_values)}"
    assert len(set(worker_xs.tolist())) == K, "Interpolation points are not distinct"

    V = build_vandermonde(worker_xs, K)
    block_r, block_c = worker_values[0].shape
    Y = np.stack(worker_values, axis=0).reshape(K, -1)

    Cflat, *_ = np.linalg.lstsq(V, Y, rcond=None)
    return Cflat.reshape(K, block_r, block_c)

def assemble_full_C_from_coeffs(coeffs, m, n):
    blocks = [[None]*n for _ in range(m)]
    for j in range(n):
        for i in range(m):
            blocks[i][j] = coeffs[i + m*j]

    rows = [np.concatenate(blocks[i], axis=1) for i in range(m)]
    return np.real(np.concatenate(rows, axis=0))

# ======================================================
# DISTRIBUTED ALGORITHM (MASTER + WORKERS)
# ======================================================
def distributed_matrix_multiplication_polynomial(A, B, m, n, N, S, seed=None):
    rng = random.Random(seed)

    A_parts = split_A_rows(A, m)
    B_parts = split_B_cols(B, n)

    x_values = np.exp(2j * np.pi * np.arange(N) / N)
    slow_nodes = simulate_slow_nodes(N, S, rng)

    worker_values = []
    worker_xs = []

    # --- ВОРКЕРЫ ---
    for w in range(N):
        if w in slow_nodes:
            continue
        x = x_values[w]
        Ax = encode_A(A_parts, x)
        Bx = encode_B(B_parts, x, m)
        worker_values.append(Ax @ Bx)
        worker_xs.append(x)

    worker_xs = np.array(worker_xs)

    # --- МАСТЕР ---
    coeffs = decode_coeffs(worker_values, worker_xs, K)
    return assemble_full_C_from_coeffs(coeffs, m, n)

# ======================================================
# ЭКСПЕРИМЕНТЫ + ГРАФИК
# ======================================================
def run_experiment_TZ(S_values, trials):
    err_mean, err_std = [], []

    print("\n=== Polynomial Codes: error summary ===")
    print(" S |   N |   mean error      |   std")
    print("-------------------------------------------")

    for S in S_values:
        N = K + S
        errs = []

        for tr in range(trials):
            seed = seed0 + 10000*S + tr
            C_rec = distributed_matrix_multiplication_polynomial(
                A, B, m, n, N, S, seed
            )
            errs.append(frob_norm(true_C - C_rec) / frob_norm(true_C))

        mean_err = np.mean(errs)
        std_err  = np.std(errs)

        err_mean.append(mean_err)
        err_std.append(std_err)

        # 🔹 ВЫВОД В ТЕРМИНАЛ
        print(f"{S:3d} | {N:4d} | {mean_err: .3e} | {std_err: .3e}")

    print("-------------------------------------------\n")

    return np.array(err_mean), np.array(err_std)

# ======================================================
# ЗАПУСК ЭКСПЕРИМЕНТОВ
# ======================================================
S_values = list(range(0, 200, 10))
err_m, err_s = run_experiment_TZ(S_values, trials=10)

plt.figure(figsize=(8,5))
plt.errorbar(S_values, err_m, yerr=err_s, fmt='o-', capsize=3)
plt.yscale("log")
plt.grid(True, alpha=0.3)
plt.xlabel("S (число неответивших узлов)")
plt.ylabel("Относительная ошибка (Фробениус)")
plt.title("Polynomial Codes — режим ТЗ (N = K + S)")
plt.show()

# ======================================================
# ВИЗУАЛЬНОЕ СРАВНЕНИЕ МАТРИЦ (ОТДЕЛЬНОЕ ОКНО)
# ======================================================
S_demo = 100
N_demo = K + S_demo
C_demo = distributed_matrix_multiplication_polynomial(
    A, B, m, n, N_demo, S_demo, seed=seed0
)

fig, axes = plt.subplots(1, 3, figsize=(15,4))
titles = ["True C", "Recovered C", "|Difference|"]
mats = [true_C, C_demo, np.abs(true_C - C_demo)]

for ax, mat, title in zip(axes, mats, titles):
    im = ax.imshow(mat, aspect="auto")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.suptitle(f"Matrix comparison (S={S_demo}, N={N_demo})")
plt.tight_layout()
plt.show()

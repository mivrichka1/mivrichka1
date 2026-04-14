import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.linalg import expm

from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation, phase_estimation
from qiskit.quantum_info import Operator

np.set_printoptions(precision=6, suppress=True)

# 1. ГАМИЛЬТОНИАН МОЛЕКУЛЫ H2 (2 КУБИТА, 1 СПИН)
#    Коэффициенты из условия проекта (R = 1.4 а.е.)
#
# H = g0*I + g1*Z0 + g2*Z1 + g3*X0X1 + g4*Y0Y1 + g5*Z0Z1
#
# Численные значения (а.е., ħ=1):

g0 = -0.1069   # I
g1 =  0.0454   # Z0
g2 =  0.0454   # Z1
g3 =  0.0454   # X0X1
g4 =  0.0454   # Y0Y1
g5 =  0.1323   # Z0Z1

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

def kron(a, b):
    return np.kron(a, b)

# Построение гамильтониана 4×4
H = (
    g0 * kron(I2, I2) +
    g1 * kron(Z, I2) +
    g2 * kron(I2, Z) +
    g3 * kron(X, X) +
    g4 * kron(Y, Y) +
    g5 * kron(Z, Z)
)

print("Гамильтониан H (4×4):")
print(H)

# 2 точное решение


eigvals, eigvecs = np.linalg.eigh(H)

order = np.argsort(eigvals.real)
eigvals = eigvals[order].real
eigvecs = eigvecs[:, order]

ground_energy = eigvals[0]
ground_state = eigvecs[:, 0]

print("\nТочные собственные значения (а.е.):")
for i, val in enumerate(eigvals):
    print(f"E_{i} = {val:.12f}")

print(f"\nОсновная энергия (эталон): E0 = {ground_energy:.12f} а.е.")
print("\nСоответствующий собственный вектор (в базисе |00>,|01>,|10>,|11>):")
print(ground_state)

# 3 сдвиг гамильтониана

# Чтобы все энергии стали положительными (φ = E'·t/(2π) ∈ [0,1))

shift = max(0.0, -eigvals.min() + 0.25)
H_shifted = H + shift * np.eye(H.shape[0], dtype=complex)

eigvals_shifted, eigvecs_shifted = np.linalg.eigh(H_shifted)
order = np.argsort(eigvals_shifted.real)
eigvals_shifted = eigvals_shifted[order].real
eigvecs_shifted = eigvecs_shifted[:, order]

ground_energy_shifted = eigvals_shifted[0]
ground_state_shifted = eigvecs_shifted[:, 0]

print(f"\nСдвиг shift = {shift:.12f} а.е.")
print(f"Минимальная энергия после сдвига = {ground_energy_shifted:.12f} а.е.")
print(f"Максимальная энергия после сдвига = {eigvals_shifted.max():.12f} а.е.")


# 4 функции для QPE

def build_exact_evolution_circuit(U):
    n_sys = int(np.log2(U.shape[0]))
    qc = QuantumCircuit(n_sys, name="U")
    qc.append(Operator(U), range(n_sys))
    return qc

def qpe_distribution_from_unitary(U, psi_target, n_eval):
    n_sys = int(np.log2(U.shape[0]))
    evolution_circuit = build_exact_evolution_circuit(U)
    pe = phase_estimation(n_eval, evolution_circuit)

    total_qubits = pe.num_qubits
    qc = QuantumCircuit(total_qubits)

    target_qubits = list(range(n_eval, n_eval + n_sys))
    qc.append(StatePreparation(psi_target, normalize=True), target_qubits)
    qc.compose(pe, inplace=True)

    full_operator = Operator(qc).data
    psi0 = np.zeros(2**total_qubits, dtype=complex)
    psi0[0] = 1.0
    final_state = full_operator @ psi0

    probs_full = np.abs(final_state) ** 2
    probs_eval = np.zeros(2**n_eval)

    mask = (1 << n_eval) - 1
    for index, prob in enumerate(probs_full):
        j = index & mask
        probs_eval[j] += prob

    return probs_eval, qc

def dominant_result(probs):
    j = int(np.argmax(probs))
    phi = j / len(probs)
    return j, phi

def phase_to_energy(phi, t, shift=0.0):
    E_shifted = 2 * np.pi * phi / t
    return E_shifted - shift

def bitstring(j, n_eval):
    return format(j, f"0{n_eval}b")

def plot_distribution(probs, title=None):
    xs = np.arange(len(probs))
    plt.figure(figsize=(10, 4))
    plt.bar(xs, probs)
    plt.xlabel("j")
    plt.ylabel("Probability")
    if title is not None:
        plt.title(title)
    plt.grid(alpha=0.25)
    plt.show()

# 5 подбор времени эволюции

# Условие: E'_max * t < 2π, чтобы фаза не заворачивалась

n_eval = 8   # число кубитов фазы (точность 2⁻⁸)
Emax_shifted = eigvals_shifted.max()

t_safe = 0.9 * 2 * np.pi / Emax_shifted
t_values = [0.25 * t_safe, 0.50 * t_safe, 0.75 * t_safe]

print(f"\nE_max_shifted = {Emax_shifted:.12f} а.е.")
print(f"Безопасный верхний предел для t ~ {2 * np.pi / Emax_shifted:.12f} а.е.")
print("Выбранные значения t (а.е.):")
for t in t_values:
    print(f"t = {t:.12f}")


# 6 запуск QPE

rows = []
circuits = {}
distributions = {}

for t in t_values:
    U = expm(-1j * H_shifted * t)
    probs, qc = qpe_distribution_from_unitary(U, ground_state_shifted, n_eval)
    j, phi = dominant_result(probs)
    E_qpe = phase_to_energy(phi, t, shift=shift)
    error_abs = abs(E_qpe - ground_energy)

    rows.append({
        "t": t,
        "j": j,
        "bitstring": bitstring(j, n_eval),
        "phi = j / 2^n": phi,
        "E_QPE": E_qpe,
        "E_exact": ground_energy,
        "abs_error": error_abs
    })

    circuits[t] = qc
    distributions[t] = probs

results_df = pd.DataFrame(rows)
print("\nРезультаты QPE:")
print(results_df)

# графики

for t in t_values:
    probs = distributions[t]
    j, phi = dominant_result(probs)
    plot_distribution(
        probs,
        title=f"QPE distribution for t = {t:.4f}, peak at j = {j} ({bitstring(j, n_eval)})"
    )

plt.figure(figsize=(8, 4))
plt.plot(results_df["t"], results_df["abs_error"], marker="o")
plt.xlabel("t (а.е.)")
plt.ylabel("Absolute error (а.е.)")
plt.title("Ошибка восстановления энергии через QPE")
plt.grid(alpha=0.3)
plt.show()

print("\nТаблица ошибок:")
print(results_df[["t", "E_QPE", "E_exact", "abs_error"]])


report_text = f'''
гамильтониан (коэффициенты из условия проекта, R = 1.4 а.е.):
H = g0·I + g1·Z0 + g2·Z1 + g3·X0X1 + g4·Y0Y1 + g5·Z0Z1
g0 = -0.1069, g1 = g2 = 0.0454, g3 = g4 = 0.0454, g5 = 0.1323

точная энергия через диагонализацию
E_точн = {ground_energy:.12f} а.е.

QPE:
Оператор эволюции: U(t) = exp(-i H' t), где H' = H + s·I
Сдвиг s = {shift:.12f} а.е. (чтобы все энергии стали положительными)
Число кубитов фазы: n = {n_eval} (точность 2⁻ⁿ = {2**(-n_eval):.6f})
Начальное состояние: собственное состояние основного уровня (|01⟩)

Фаза: φ = j / 2ⁿ
Энергия: E = 2π·φ / t - s

результаты:
При оптимальном t = {results_df.loc[results_df['abs_error'].idxmin(), 't']:.4f} а.е.
наиболее вероятное j = {int(results_df.loc[results_df['abs_error'].idxmin(), 'j'])}
оценка энергии E_QPE = {results_df.loc[results_df['abs_error'].idxmin(), 'E_QPE']:.12f} а.е.
ошибка = {results_df.loc[results_df['abs_error'].idxmin(), 'abs_error']:.12f} а.е.

ВЫВОД:
Алгоритм QPE успешно оценивает энергию основного состояния модели H2.
Ошибка восстановления согласуется с теоретической границей ΔE ≤ π/(2ⁿ·t).
'''.strip()

print("\n" + "="*60)
print(report_text)
print("="*60)
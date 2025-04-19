import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import spearmanr
import pandas as pd
import os


def compute_rivn_total_matrix(expert_opinions):
    linguistic_to_svnn = {
        0: (0.1, 0.8, 0.9),
        1: (0.3, 0.6, 0.7),
        2: (0.5, 0.5, 0.5),
        3: (0.7, 0.4, 0.3),
        4: (0.9, 0.2, 0.1)
    }

    expert_count, factor_count, _ = expert_opinions.shape
    svnn_opinions = np.empty((expert_count, factor_count, factor_count, 3))
    for e in range(expert_count):
        for i in range(factor_count):
            for j in range(factor_count):
                svnn_opinions[e, i, j] = linguistic_to_svnn[int(expert_opinions[e, i, j])]

    ivnn_opinions = np.zeros((expert_count, factor_count, factor_count, 3, 2))
    for e in range(expert_count):
        for i in range(factor_count):
            for j in range(factor_count):
                for k in range(3):
                    all_values = svnn_opinions[:, i, j, k]
                    current_value = svnn_opinions[e, i, j, k]
                    lower_vals = [val for val in all_values if val <= current_value]
                    upper_vals = [val for val in all_values if val >= current_value]
                    ivnn_opinions[e, i, j, k, 0] = np.mean(lower_vals) if lower_vals else current_value
                    ivnn_opinions[e, i, j, k, 1] = np.mean(upper_vals) if upper_vals else current_value

    return finalize_dematel(ivnn_opinions)

def compute_ivn_total_matrix(expert_opinions):
    linguistic_to_ivnn = {
        0: ([0.1, 0.2], [0.5, 0.6], [0.7, 0.8]),
        1: ([0.2, 0.4], [0.5, 0.6], [0.5, 0.6]),
        2: ([0.4, 0.6], [0.4, 0.5], [0.3, 0.4]),
        3: ([0.6, 0.8], [0.3, 0.4], [0.2, 0.4]),
        4: ([0.7, 0.9], [0.2, 0.3], [0.1, 0.2])
    }

    expert_count, factor_count, _ = expert_opinions.shape
    ivnn_opinions = np.zeros((expert_count, factor_count, factor_count, 3, 2))
    for e in range(expert_count):
        for i in range(factor_count):
            for j in range(factor_count):
                T, I, F = linguistic_to_ivnn[int(expert_opinions[e, i, j])]
                ivnn_opinions[e, i, j, 0] = T
                ivnn_opinions[e, i, j, 1] = I
                ivnn_opinions[e, i, j, 2] = F

    return finalize_dematel(ivnn_opinions)

def compute_fuzzy_total_matrix(expert_opinions):
    linguistic_to_tfn = {
        0: (0.0, 0.0, 0.25),
        1: (0.0, 0.25, 0.50),
        2: (0.25, 0.50, 0.75),
        3: (0.50, 0.75, 1.00),
        4: (0.75, 1.00, 1.00)
    }

    expert_count, factor_count, _ = expert_opinions.shape
    tfn_opinions = np.zeros((expert_count, factor_count, factor_count, 3))
    for e in range(expert_count):
        for i in range(factor_count):
            for j in range(factor_count):
                tfn_opinions[e, i, j] = linguistic_to_tfn[int(expert_opinions[e, i, j])]

    aggregated_tfn = np.zeros((factor_count, factor_count, 3))
    for i in range(factor_count):
        for j in range(factor_count):
            l_vals = tfn_opinions[:, i, j, 0]
            m_vals = tfn_opinions[:, i, j, 1]
            u_vals = tfn_opinions[:, i, j, 2]
            aggregated_tfn[i, j, 0] = np.prod(l_vals) ** (1 / expert_count)
            aggregated_tfn[i, j, 1] = np.prod(m_vals) ** (1 / expert_count)
            aggregated_tfn[i, j, 2] = np.prod(u_vals) ** (1 / expert_count)

    crisp_matrix = np.mean(aggregated_tfn, axis=2)
    row_sums = np.sum(crisp_matrix, axis=1)
    col_sums = np.sum(crisp_matrix, axis=0)
    max_row_sum = np.max(row_sums)
    max_col_sum = np.max(col_sums)
    k = min(1 / max_row_sum, 1 / max_col_sum) if max(max_row_sum, max_col_sum) != 0 else 0
    normalized_matrix = crisp_matrix * k
    identity_matrix = np.identity(factor_count)
    try:
        inverse_part = np.linalg.inv(identity_matrix - normalized_matrix)
        total_relation_matrix = normalized_matrix @ inverse_part
    except np.linalg.LinAlgError:
        total_relation_matrix = np.zeros_like(normalized_matrix)

    D = np.sum(total_relation_matrix, axis=1)
    R = np.sum(total_relation_matrix, axis=0)
    return D, R, total_relation_matrix

def finalize_dematel(ivnn_opinions):
    expert_count, factor_count, _, _, _ = ivnn_opinions.shape
    weights = np.full(expert_count, 1 / expert_count)
    aggregated_ivnn = np.zeros((factor_count, factor_count, 3, 2))
    for i in range(factor_count):
        for j in range(factor_count):
            for k in range(3):
                values_L = ivnn_opinions[:, i, j, k, 0]
                values_U = ivnn_opinions[:, i, j, k, 1]
                if k == 0:
                    aggregated_ivnn[i, j, k, 0] = 1 - np.prod([(1 - x) ** w for x, w in zip(values_L, weights)])
                    aggregated_ivnn[i, j, k, 1] = 1 - np.prod([(1 - x) ** w for x, w in zip(values_U, weights)])
                else:
                    aggregated_ivnn[i, j, k, 0] = np.prod([x ** w for x, w in zip(values_L, weights)])
                    aggregated_ivnn[i, j, k, 1] = np.prod([x ** w for x, w in zip(values_U, weights)])

    crisp_matrix = np.zeros((factor_count, factor_count))
    for i in range(factor_count):
        for j in range(factor_count):
            T_L, T_U = aggregated_ivnn[i, j, 0]
            I_L, I_U = aggregated_ivnn[i, j, 1]
            F_L, F_U = aggregated_ivnn[i, j, 2]
            numerator = (T_L + T_U + (1 - F_L) + (1 - F_U) + (T_L * T_U) + np.sqrt(abs((1 - F_L) * (1 - F_U)))) / 6
            denominator = ((1 - (I_L + I_U) / 2) * np.sqrt(abs((1 - I_L) * (1 - I_U)))) / 2
            crisp_matrix[i, j] = numerator * denominator if denominator != 0 else 0

    row_sums = np.sum(crisp_matrix, axis=1)
    col_sums = np.sum(crisp_matrix, axis=0)
    max_row_sum = np.max(row_sums)
    max_col_sum = np.max(col_sums)
    k = min(1 / max_row_sum, 1 / max_col_sum) if max(max_row_sum, max_col_sum) != 0 else 0
    normalized_matrix = crisp_matrix * k
    identity_matrix = np.identity(factor_count)
    try:
        inverse_part = np.linalg.inv(identity_matrix - normalized_matrix)
        total_relation_matrix = normalized_matrix @ inverse_part
    except np.linalg.LinAlgError:
        total_relation_matrix = np.zeros_like(normalized_matrix)
    D = np.sum(total_relation_matrix, axis=1)
    R = np.sum(total_relation_matrix, axis=0)
    return D, R, total_relation_matrix

def save_boxplot(data_lists, labels, title, filename, all_summaries):
    os.makedirs("outputs", exist_ok=True)
    colors = ['#66c2a5', '#8da0cb', '#fc8d62']
    plt.figure(figsize=(10, 6))
    bp = plt.boxplot(data_lists, patch_artist=True, tick_labels=labels)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    for i, data in enumerate(data_lists):
        mean_val = np.mean(data)
        plt.text(i + 1, mean_val + 0.01, f"μ = {mean_val:.3f}", ha='center', fontsize=10, fontweight='bold')
    plt.title(title)
    plt.ylabel("Correlation")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"outputs/{filename}.png")
    plt.close()

    print(f"\n📊 {title}")
    for lbl, data in zip(labels, data_lists):
        mean_val = np.mean(data)
        std_val = np.std(data)
        print(f"{lbl}: mean = {mean_val:.4f}, std = {std_val:.4f}")
        all_summaries.append({"Comparison": title, "Metric": lbl, "Mean": mean_val, "StdDev": std_val})


def run_simulation(num_runs):
    rivn_ivn_dpr, rivn_ivn_dmr, rivn_ivn_pearson = [], [], []
    rivn_fuzzy_dpr, rivn_fuzzy_dmr, rivn_fuzzy_pearson = [], [], []
    ivn_fuzzy_dpr, ivn_fuzzy_dmr, ivn_fuzzy_pearson = [], [], []
    all_summaries = []

    for _ in range(num_runs):
        factor_count = random.randint(15, 30)
        expert_count = random.randint(10, 20)
        expert_opinions = np.random.randint(0, 5, size=(expert_count, factor_count, factor_count))

        D_rivn, R_rivn, T_rivn = compute_rivn_total_matrix(expert_opinions)
        D_ivn, R_ivn, T_ivn = compute_ivn_total_matrix(expert_opinions)
        D_fuzzy, R_fuzzy, T_fuzzy = compute_fuzzy_total_matrix(expert_opinions)

        for lst, a1, a2 in zip(
            [rivn_ivn_dpr, rivn_fuzzy_dpr, ivn_fuzzy_dpr],
            [D_rivn + R_rivn, D_rivn + R_rivn, D_ivn + R_ivn],
            [D_ivn + R_ivn, D_fuzzy + R_fuzzy, D_fuzzy + R_fuzzy]
        ):
            lst.append(spearmanr(a1, a2)[0])

        for lst, a1, a2 in zip(
            [rivn_ivn_dmr, rivn_fuzzy_dmr, ivn_fuzzy_dmr],
            [D_rivn - R_rivn, D_rivn - R_rivn, D_ivn - R_ivn],
            [D_ivn - R_ivn, D_fuzzy - R_fuzzy, D_fuzzy - R_fuzzy]
        ):
            lst.append(spearmanr(a1, a2)[0])

        for lst, a1, a2 in zip(
            [rivn_ivn_pearson, rivn_fuzzy_pearson, ivn_fuzzy_pearson],
            [T_rivn.flatten(), T_rivn.flatten(), T_ivn.flatten()],
            [T_ivn.flatten(), T_fuzzy.flatten(), T_fuzzy.flatten()]
        ):
            lst.append(np.corrcoef(a1, a2)[0, 1])

    save_boxplot([rivn_ivn_dpr, rivn_ivn_dmr, rivn_ivn_pearson],
                 ["D+R", "D−R", "T Matrix"],
                 f"RIVN vs IVN (N={num_runs})", "rivn_vs_ivn", all_summaries)

    save_boxplot([rivn_fuzzy_dpr, rivn_fuzzy_dmr, rivn_fuzzy_pearson],
                 ["D+R", "D−R", "T Matrix"],
                 f"RIVN vs Fuzzy (N={num_runs})", "rivn_vs_fuzzy", all_summaries)

    save_boxplot([ivn_fuzzy_dpr, ivn_fuzzy_dmr, ivn_fuzzy_pearson],
                 ["D+R", "D−R", "T Matrix"],
                 f"IVN vs Fuzzy (N={num_runs})", "ivn_vs_fuzzy", all_summaries)

    df_all = pd.DataFrame(all_summaries)
    df_all.to_excel("outputs/all_comparisons_summary.xlsx", index=False)


if __name__ == "__main__":
    num_runs = int(input("Enter number of simulation runs: "))
    run_simulation(num_runs)

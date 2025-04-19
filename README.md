# RIVN-DEMATEL vs IVN & Fuzzy DEMATEL – Simulation Comparison

This repository contains a Python implementation of a simulation-based comparative analysis between three decision-making methodologies:

- **Rough Interval-Valued Neutrosophic DEMATEL (RIVN-DEMATEL)** – the proposed method,
- **Interval-Valued Neutrosophic DEMATEL (IVN-DEMATEL)**,
- **Triangular Fuzzy DEMATEL (F-DEMATEL)**.

The simulation aims to evaluate the **reliability** and **consistency** of the proposed RIVN-DEMATEL method by comparing its results to the other two established techniques using correlation-based statistical metrics.

## Features

- Generates random expert evaluations using a discrete linguistic scale (0 to 4)
- Applies all three DEMATEL methods under the same simulated conditions
- Computes:
  - Spearman correlation for **D+R (Prominence)** and **D−R (Relation)**
  - Pearson correlation for **T matrices**
- Outputs results as:
  - **Boxplot visualizations**
  - **Excel summary files** with mean and standard deviation values for each metric

## Requirements

- Python 3.8+
- Libraries: `numpy`, `pandas`, `matplotlib`, `scipy`, `tqdm`

Install all dependencies using:

```bash
pip install -r requirements.txt
How to Use
Run the main script and enter the number of simulation iterations:

bash
Kopyala
Düzenle
python F_RIVN_DEMATEL.py
All plots and results will be saved automatically in the outputs/ folder.

Output Structure
rivn_vs_ivn.png, rivn_vs_fuzzy.png, ivn_vs_fuzzy.png: Boxplot visualizations

Corresponding .xlsx files with descriptive statistics per comparison

A consolidated file: all_comparisons_summary.xlsx

Academic Context
This simulation was conducted as part of a research study aiming to validate the proposed RIVN-DEMATEL method in comparison with conventional IVN and fuzzy-based DEMATEL techniques. The methodology and analysis were designed for academic use and reproducibility.

Author
Ahmet Öztel
Bartın University
aoztel@bartin.edu.tr

📘 For questions, collaborations, or citations, feel free to reach out or fork the repository.

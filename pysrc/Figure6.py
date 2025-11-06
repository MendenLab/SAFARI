import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import tight_layout

from Figure3 import load_qpcr_data

sns.set_style('white')

def probability_plot(sample_data):
    plt.figure(figsize=(1.,1.5))
    if sample_data["HOMER1_TBP_LCK_TBP_HOMER1_SDHAF_LCK_SDHAF_baseline_probabilities_calibrated"].values[0] > 0.6:
        color = "#ff5b57"
    elif sample_data["HOMER1_TBP_LCK_TBP_HOMER1_SDHAF_LCK_SDHAF_baseline_probabilities_calibrated"].values[0] < 0.4:
        color = "#8F8071"
    else: color="grey"

    sns.scatterplot(data=sample_data,x="sampleID",
                    y="HOMER1_TBP_LCK_TBP_HOMER1_SDHAF_LCK_SDHAF_baseline_probabilities_calibrated",
                    s=5, color=color)
    sns.despine()
    plt.ylim((0,1.05))
    plt.ylabel("Probability for MF", fontsize=6)
    plt.xlabel("")
    plt.tick_params(axis='both', labelsize=6)
    plt.grid(False)
    plt.tight_layout()
    sample = sample_data["sampleID"].values[0]
    plt.savefig(os.path.join(f"/Figures/Figure6/Verlauf",
                             f"{sample}.pdf"),  dpi=300, bbox_inches='tight')
    plt.close()


def plot_patient_history(patient_data, patient):
    timesteps = [f"T{i+1}" for i in range(patient_data.shape[0])]
    patient_data["time"] = timesteps
    if len(timesteps) > 10:
        figsize = (3,2)
    else:
        figsize = (2,1.5)
    plt.figure(figsize=figsize)
    sns.lineplot(data=patient_data, y="HOMER1_TBP_LCK_TBP_HOMER1_SDHAF_LCK_SDHAF_baseline_probabilities_calibrated",
                    x="time",linewidth=0.8)
    plt.ylim((0,1.05))
    plt.ylabel("Probability for MF", fontsize=6)
    plt.xlabel("Time", fontsize=6)
    plt.tick_params(axis='both', labelsize=6)
    plt.yticks(np.arange(0, 1.05, 0.2))
    plt.grid(False)
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(f"/Figures/Figure6/Verlauf2",
                             f"patient_{patient+1}.pdf"),  dpi=300, bbox_inches='tight')
    plt.close()


def main():
    subplot = "b"
    if subplot == "a":
        sample_list = ["NL23-003", "NL23-004", "NL23-005", "NL23-006", "NL23-007", "NL23-008",
                       "NL23-009", "NL23-010", "NL23-011",
                       "AKF3-015", "AKF3-007", "AKF3-008",
                       "AKF3-019", "AKF3-012",
                       "AKF3-011", "AKF3-003",
                       "AKF9-022", "AKF9-023", "AKF9-024"]
        data = load_qpcr_data()
        for sample in sample_list:
            sample_data = data.loc[data["sampleID"] == sample, :]
            if len(sample_data) == 0:
                raise ValueError(f"Sample {sample} not found")
            probability_plot(sample_data)
    else:
        pat1 = ["AKF9-008", "AKF9-009", "AKF9-010", "AKF9-011", "AKF9-012", "AKF9-013",
                "AKF9-014", "AKF9-015", "AKF9-016", "AKF9-017", "AKF9-018", "AKF9-019",
                "AKF9-020", "AKF9-021"]
        pat2 = ["NL35-113", "NL35-076 Wdh2", "NL35-084"]
        pat3 = ["AKF10-001", "AKF10-002", "AKF10-003"]
        pat4 = ["NL35-091", "NL35-089", "NL35-072"]
        patset = [pat1, pat2, pat3, pat4]
        data = load_qpcr_data()
        data.index = data["sampleID"]
        for i, pat in enumerate(patset):
            patient_data = data.loc[pat, :]
            plot_patient_history(patient_data, patient=i)

if __name__ == '__main__':
    main()
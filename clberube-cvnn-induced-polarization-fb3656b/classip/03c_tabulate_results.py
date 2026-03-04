#
# Author: Charles L. Bérubé
# Created on: Tue Sep 10 2024
#
# Copyright (c) 2024 CL Bérubé JL Gagnon & S Gagnon
#

import os
import pickle

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from utilities import str_with_err

plt.style.use("../seg.mplstyle")


res_dir = "./results"
fig_dir = "./figures"
os.makedirs(fig_dir, exist_ok=True)


with open(f"{res_dir}/complex/results.pkl", "rb") as f:
    r = pickle.load(f)
    results_complex = np.array(r["auc"])
    time_complex = np.array(r["time"])
    df_complex = pd.concat([pd.DataFrame().from_dict(d) for d in r["report"]])
    n_epochs = len(r["train"][0])
    results_complex_train = np.array(r["train"])
    results_complex_valid = np.array(r["valid"])


with open(f"{res_dir}/real/results.pkl", "rb") as f:
    r = pickle.load(f)
    results_real = np.array(r["auc"])
    time_real = np.array(r["time"])
    df_real = pd.concat([pd.DataFrame().from_dict(d) for d in r["report"]])
    results_real_train = np.array(r["train"])
    results_real_valid = np.array(r["valid"])


print(results_complex.mean(), "+/-", results_complex.std())
print(results_real.mean(), "+/-", results_real.std())

fig, ax = plt.subplots()
ax.bar(0, results_real.mean(), yerr=results_real.std())
ax.bar(1, results_complex.mean(), yerr=results_complex.std())
ax.set_ylabel("ROC-AUC")
ax.set_xticks([0, 1], ["Real", "Complex"])
plt.tight_layout()
plt.savefig(f"{fig_dir}/ROC-AUC.pdf")
plt.savefig(f"{fig_dir}/ROC-AUC.png")

fig, ax = plt.subplots()
ax.bar(0, time_real.mean(), yerr=time_real.std())
ax.bar(1, time_complex.mean(), yerr=time_complex.std())
ax.set_ylabel("Training time (s)")
ax.set_xticks([0, 1], ["Real", "Complex"])
plt.tight_layout()


rows = ["f1-score", "precision", "recall"]
cols = ["macro avg"]

df_real_avg = df_real.groupby(df_real.index).mean().loc[rows, cols]
df_complex_avg = df_complex.groupby(df_complex.index).mean().loc[rows, cols]

df_real_std = df_real.groupby(df_real.index).std().loc[rows, cols]
df_complex_std = df_complex.groupby(df_complex.index).std().loc[rows, cols]


min_real_valid = results_real_valid.mean(0).min()
min_complex_valid = results_complex_valid.mean(0).min()

mean_complex_valid = results_complex_valid.mean(0)
comp_epoch = np.where(mean_complex_valid < min_real_valid)[0][0]

print("Epoch fraction for complex to pass real:", comp_epoch / n_epochs)


df_avg = pd.concat([df_real_avg, df_complex_avg], axis=1)
df_avg.loc["ROC-AUC"] = [results_real.mean(0), results_complex.mean(0)]
df_avg.loc["Time per epoch"] = (
    np.array([time_real.mean(), time_complex.mean()]) / n_epochs
)
df_avg.loc["Equal epoch"] = [n_epochs, comp_epoch]
df_avg.loc["Training time"] = df_avg.loc["Equal epoch"] * df_avg.loc["Time per epoch"]

df_std = pd.concat([df_real_std, df_complex_std], axis=1)
df_std.loc["ROC-AUC"] = [results_real.std(0), results_complex.std(0)]
df_std.loc["Time per epoch"] = (
    np.array([time_real.std(), time_complex.std()]) / n_epochs
)
df_std.loc["Equal epoch"] = [0, 0]
df_std.loc["Training time"] = df_std.loc["Time per epoch"]


df = pd.DataFrame().reindex_like(df_avg).astype(str)
for i in range(df_avg.shape[0]):
    for j in range(df_avg.shape[1]):
        df.iloc[i, j] = str_with_err(df_avg.iloc[i, j], df_std.iloc[i, j])

header = ["Real", "Complex"]
df.columns = header


print(df.to_latex())

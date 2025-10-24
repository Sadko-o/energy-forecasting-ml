# 1_data_prep.py
import numpy as np, pandas as pd, json, os
from pathlib import Path

HIST=60; PRED_OFS=1  # history length (minutes), prediction offset (minutes)
DATA = "household_power_consumption.txt"  
OUT  = Path("artifacts"); 
OUT.mkdir(exist_ok=True, parents=True)





df = pd.read_csv(DATA, sep=";", low_memory=False, na_values="?")
df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst=True)
df = df.drop(columns=["Date","Time"]).set_index("datetime").sort_index()
df = df.apply(pd.to_numeric, errors="coerce")

# --- Unify to strict 1-minute grid, fill short gaps ---
idx = pd.date_range(df.index.min().floor("min"), df.index.max().ceil("min"), freq="1min")
df = df.reindex(idx)
df = df.interpolate(limit=5).ffill().bfill()  # short gaps only

# --- Select features & add time encodings ---
FEATS = ["Global_active_power","Voltage","Global_reactive_power","Global_intensity","Sub_metering_1","Sub_metering_2","Sub_metering_3"]
df = df[FEATS]
h = df.index.hour; dow = df.index.dayofweek
df["hour_sin"] = np.sin(2*np.pi*h/24); df["hour_cos"] = np.cos(2*np.pi*h/24)
df["dow_sin"]  = np.sin(2*np.pi*dow/7); df["dow_cos"]  = np.cos(2*np.pi*dow/7)

ALLCOLS = df.columns.tolist()
TARGET_COL = "Global_active_power"

# --- Chronological split (example: train 2007–2009, val 2010-01..09, test 2010-10..11)
train = df.loc[:"2009-12-31"]
val   = df.loc["2010-01-01":"2010-09-30"]
test  = df.loc["2010-10-01":]

# --- Standardize by train (save stats for deployment) ---
mu, sigma = train.mean(), train.std().replace(0, 1.0)
train_n = (train - mu)/sigma
val_n   = (val   - mu)/sigma
test_n  = (test  - mu)/sigma

WITH_TIME = ALLCOLS  # keep engineered time encodings

def build_windows(frame, hist=60, pred_ofs=1, cols=WITH_TIME, target=TARGET_COL, stride=1):
    X, y = [], []
    arr = frame[cols].to_numpy(dtype=np.float32)
    t   = frame[target].to_numpy(dtype=np.float32)
    for i in range(0, len(frame) - hist - pred_ofs + 1, stride):
        X.append(arr[i:i+hist])
        y.append(t[i+hist+pred_ofs-1])
    return np.stack(X), np.array(y)[:, None]


Xtr, ytr = build_windows(train_n, HIST, PRED_OFS)
Xva, yva = build_windows(val_n,   HIST, PRED_OFS)
Xte, yte = build_windows(test_n,  HIST, PRED_OFS)


np.savez_compressed(
    OUT/"reg_data.npz",
    Xtr=Xtr, ytr=ytr,
    Xva=Xva, yva=yva,
    Xte=Xte, yte=yte,
    cols=np.array(WITH_TIME)
)

# with open(OUT/"norm_stats.json", "w") as f:
#     json.dump({"mu":mu.to_dict(), "sigma":sigma.to_dict(), "cols":WITH_TIME, "hist":HIST}, f, indent=2)

stats = {
    "hist": HIST,
    "cols": WITH_TIME, 
    "mu":   [round(float(mu[c]),   6) for c in WITH_TIME],
    "sigma":[round(float(sigma[c]),6) for c in WITH_TIME],
    "target_col": TARGET_COL
}

with open(OUT/"norm_stats.json", "w") as f:
    json.dump(stats, f, indent=2)

print("train/val/test windows:", Xtr.shape, Xva.shape, Xte.shape)

import numpy as np
import yaml
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- config ----------
DOG_ID = 26
NPZ_PATH = Path("processed_data") / f"dog_{DOG_ID}_intelligent_new.npz"
CFG_PATH = Path("dog_behaviors_config.yaml")
BEHAV_KEY = "main_behaviors"  # backbone+cil order matches this in your YAML
SENSOR_NAME_FOR_TRACE = "ABack_x"  # change if you want a different channel
# ----------------------------

# Load config to get canonical behavior order -> indices
with open(CFG_PATH, "r") as f:
    cfg = yaml.safe_load(f)
behaviors = cfg[BEHAV_KEY]  # e.g. ["Standing","Walking","Sitting","Lying chest","Sniffing","Trotting","Galloping"]
behavior_to_idx = {b:i for i,b in enumerate(behaviors)}

# Load dog npz
data = np.load(NPZ_PATH, allow_pickle=True)
X = data["X"]            # (N, L, C)
y = data["y"]            # (N,) global indices per cfg order above
seg_ids = data["segment_ids"]       # (N,)
sess_ids = data["session_ids"]      # (N, 2) [DogID, TestNum]
sensor_cols = list(data["sensor_cols"])
L = int(data["window_len"])
stride = int(data["stride"])
fs = float(data["sample_rate"])     # Hz
sessions = data["sessions"]         # unique session TestNums

# Choose a sensor channel index for the diagnostic trace
if SENSOR_NAME_FOR_TRACE not in sensor_cols:
    print(f"Warning: {SENSOR_NAME_FOR_TRACE} not found; using first channel.")
    chan_idx = 0
else:
    chan_idx = sensor_cols.index(SENSOR_NAME_FOR_TRACE)

# Helper: make a per-session timeline (time, y, seg) and a per-window mean trace
def build_session_view(testnum):
    mask = (sess_ids[:, 1] == testnum)
    y_s = y[mask]
    seg_s = seg_ids[mask]
    X_s = X[mask]  # (Nsess, L, C)

    # Relative time per window (start time), seconds
    dt = stride / fs
    t = np.arange(len(y_s)) * dt

    # Per-window mean of chosen channel
    trace = X_s[:, :, chan_idx].mean(axis=1)
    return t, y_s, seg_s, trace

# Plotting per session
num_sessions = len(sessions)
fig, axes = plt.subplots(num_sessions, 2, figsize=(14, 3.8 * num_sessions), gridspec_kw={"width_ratios":[3,3]})
if num_sessions == 1:
    axes = np.array([axes])  # shape (1,2)

for row, testnum in enumerate(sorted(sessions)):
    t, y_s, seg_s, trace = build_session_view(testnum)

    # --- (A) Behaviour timeline ribbon ---
    ax = axes[row, 0]
    # draw as an image of shape (1, N) so each window is a vertical strip
    img = y_s[np.newaxis, :]  # shape (1, N)
    # extent maps window index to time axis
    extent = (0, len(y_s) * (stride / fs), 0, 1)
    im = ax.imshow(img, aspect='auto', extent=extent, interpolation='nearest')
    ax.set_yticks([])
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Dog {DOG_ID} — Session {int(testnum)} — Behaviour timeline")

    # Behaviour legend (ticks at discrete ints)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_ticks(list(range(len(behaviors))))
    cbar.set_ticklabels(behaviors)

    # Segment boundaries + window counts per segment (annotate near segment mid)
    unique_segs, first_idx = np.unique(seg_s, return_index=True)
    # sort segments by first occurrence to get chronological order
    order = np.argsort(first_idx)
    unique_segs = unique_segs[order]

    for s in unique_segs:
        idxs = np.where(seg_s == s)[0]
        start_t = idxs[0] * (stride / fs)
        end_t = (idxs[-1] + 1) * (stride / fs)
        # vertical lines at boundaries
        ax.axvline(start_t, linewidth=1, linestyle='--', alpha=0.6)
        # annotate number of windows in this segment
        mid_t = 0.5 * (start_t + end_t)
        n_windows = len(idxs)
        ax.text(mid_t, 0.5, f"{n_windows}", ha='center', va='center', fontsize=8, alpha=0.9)

    # --- (B) Per-window mean of chosen channel ---
    ax2 = axes[row, 1]
    ax2.plot(t, trace)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel(f"{SENSOR_NAME_FOR_TRACE} (mean per window)")
    ax2.set_title(f"Dog {DOG_ID} — Session {int(testnum)} — {SENSOR_NAME_FOR_TRACE} (per-window mean)")
    # also draw segment boundaries for alignment
    for s in unique_segs:
        idxs = np.where(seg_s == s)[0]
        start_t = idxs[0] * (stride / fs)
        ax2.axvline(start_t, linewidth=1, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig(f"plots_and_metrics/visualise_dog_{DOG_ID}")
# plt.show()
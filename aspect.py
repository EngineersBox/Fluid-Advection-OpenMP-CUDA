import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy

# ==== 1D STUFF ====

p = [1,6,12,16,24,36,48]
p2 = [2,4,8,10,14,18,20,22,26,28,30,32,34,38,40,42,44,46]
at_2 = [
    1.37e-01,
    7.25e-02,
    4.31e-02,
    3.84e-02,
    1.06e-01,
    1.02e-01,
    1.03e-01,
    1.14e-01,
    2.36e-02,
    2.25e-02,
    2.16e-02,
    2.13e-02,
    2.11e-02,
    2.08e-02,
    1.43e-01,
    1.99e-02,
    2.03e-02,
    2.02e-02,
]
advection_time = [
    2.27e-01,
    5.40e-02,
    3.70e-02,
    1.06e-01,
    1.14e-01,
    2.13e-02,
    2.51e-02,
]
load_misses = [
13.99,
12.31,
11.94,
10.28,
11.21,
11.40,
11.42,
13.35,
12.13,
14.22,
13.51,
12.21,
13.37,
15.44,
13.13,
18.42,
16.12,
10.89,
15.27,
14.53,
11.85,
11.82,
15.64,
12.52,
14.94
] + [
13.77,
11.65,
4.99,
3.62,
2.48,
2.08,
1.96,
2.30,
3.36,
3.53,
4.00,
4.54,
4.14,
50.49,
49.12,
43.85,
46.53,
55.52,
51.10,
51.46,
57.56,
53.73,
45.32,
56.77,
52.11
]

raw = dict(zip(p + p2, advection_time + at_2))
ordered_p = sorted(p + p2)
ordered_at = [raw[i] for i in ordered_p]

# ==== 2D STUFF ====

ordered_p = [
    "1x1",
    "2x2",
    "3x3",
    "4x4",
    "5x5",
    "6x6",

    "1x48",
    "2x24",
    "3x16",
    "4x12",
    "6x8",

    "1x24",
    "2x12",
    "3x8",
    "4x6"
]
ordered_at = [
    7.22e-02,
    6.39e-02,
    3.78e-02,
    8.39e-02,
    8.39e-02,
    9.42e-02,

    1.48e-01,
    1.23e-01,
    1.30e-01,
    1.14e-01,
    1.06e-01,

    8.13e-02,
    8.07e-02,
    8.05e-02,
    7.99e-02
]

load_misses = [
    4.35,
    4.19,
    4.31,
    4.37,
    5.68,
    6.26,

    7.04,
    7.00,
    6.42,
    5.59,
    5.73,

    6.10,
    6.04,
    5.22,
    4.98
] + [
    5.44,
    3.22,
    1.43,
    1.41,
    1.71,
    1.46,

    1.31,
    1.24,
    1.24,
    1.72,
    1.92,

    1.25,
    1.24,
    1.32,
    1.46
]

# ==== GPU ====

ordered_p [
    "1x1,1x1",
    "1x1,1x32",
    "1x1,32x1",
    "128x128,32x32",
    "128x128,32x16",
    "256x128,16x32",
    "256x256,16x16",

    "512x512,32x32",
    "512x512,32x16",
    "1024x512,16x32",
    "1024x1024,16x16"
]

ordered_at = [
    0,
    0,
    1.52e+02,
    6.21e-01,
    4.25e-01,
    5.37e-01,
    4.97e-01,
]

types = (["L1 D-Cache"] * len(ordered_p)) + (["LLC"] * len(ordered_p))

data = pd.DataFrame({"Aspect Ratios": ordered_p, "Advection Time (s)": ordered_at})
miss_data = pd.DataFrame({"Aspect Ratios": ordered_p + ordered_p, "Load Misses (%)": load_misses, "type": types})

fig, ax1 = plt.subplots(figsize=(12,6))

sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
sns.barplot(y = "Load Misses (%)", x = "Aspect Ratios", data = miss_data, hue = "type", ax=ax1, palette=sns.color_palette("muted"))
ax2 = ax1.twinx()
g = sns.lineplot(y = "Advection Time (s)", x = "index", data = data.reset_index(), marker="o", ax=ax2, color="black")
plt.title("Varying Aspect Ratios 2D Advection OpenMP")
plt.ticklabel_format(style="scientific", axis="y", scilimits=(0,0))
# plt.xlim(0,50)
# plt.xticks(range(0,50,4))
ax1.legend(loc='upper right')
plt.axvline(5.5, color="red")
plt.axvline(10.5, color="red")

# ymin, ymax = g.get_ylim()
# offset = (ymax - ymin) / 50
# for x, y in zip(ordered_p, ordered_at):
    # g.text(x, y + offset, f"{x}")

plt.show()


import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy

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

print(len(ordered_p))

types = (["L1 D-Cache"] * len(ordered_p)) + (["LLC"] * len(ordered_p))

data = pd.DataFrame({"P": dict(zip(range(0,len(ordered_p)), ordered_p)), "Advection Time (s)": dict(zip(range(0,len(ordered_p)),ordered_at))})
miss_data = pd.DataFrame({"P": dict(zip(range(0,len(ordered_p)*2), ordered_p + ordered_p)), "Load Misses (%)": dict(zip(range(0,len(ordered_p)*2),load_misses)), "type": dict(zip(range(0,len(ordered_p)*2),types))})

fig, ax1 = plt.subplots(figsize=(12,6))

sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
sns.barplot(y = "Load Misses (%)", x = "P", data = miss_data, hue = "type", ax=ax1, palette=sns.color_palette("muted"))
ax2 = ax1.twinx()
g = sns.lineplot(y = "Advection Time (s)", x = "index", data = data.reset_index(), marker="o", ax=ax2, color="black")
plt.title("Varying Thread Count 1D Advection OpenMP")
plt.ticklabel_format(style="scientific", axis="y", scilimits=(0,0))
plt.xlim(0,50)
plt.xticks(range(0,50,4))
ax1.legend(loc='upper center')

# ymin, ymax = g.get_ylim()
# offset = (ymax - ymin) / 50
# for x, y in zip(ordered_p, ordered_at):
    # g.text(x, y + offset, f"{x}")

plt.show()

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.size'] = 16
mpl.rcParams['lines.linewidth'] = 2
mpl.rc('text', usetex=False)


# %%

custom_lines = [Line2D([0], [0], color='indigo', lw=2),
                Line2D([0], [0], color='tab:blue', lw=2),
                Line2D([0], [0], color='orange', lw=2)]

fig, ax = plt.subplots(figsize=(1, 1), dpi=100)
ax.axis('off')
ax.legend(custom_lines, ['1.0',
                         '0.6',
                         '0.3'], title='$\eta$',
          frameon=False)

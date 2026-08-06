# Plotting Style (`SA_RC`)

All figures use a shared `SA_RC` rcParams dict, 
applied in analysis scripts via `plt.rcParams.update(SA_RC)`.

```python
SA_RC = {
    # Figure
    'figure.dpi': 150,
    'figure.facecolor': 'white',
    'figure.autolayout': True,
    # Axes
    'axes.facecolor': 'white',
    'axes.edgecolor': '#444444',
    'axes.linewidth': 0.8,
    'axes.spines.top': True,
    'axes.spines.right': True,
    'axes.labelsize': 12,
    'axes.labelcolor': 'black',
    'axes.labelweight': '500',
    'axes.titlelocation': 'center',
    'axes.grid': False,
    # Ticks
    'xtick.labelsize': 12,
    'xtick.color': 'black',
    'xtick.direction': 'out',
    'ytick.labelsize': 10,
    'ytick.color': 'black',
    'ytick.direction': 'out',
    # Font
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica Neue LT Pro', 'Helvetica Neue', 'Arial', 'DejaVu Sans'],
    'font.size': 10,
    'font.weight': '300',
    'text.color': 'black',
    # Patches (bars, legend swatches)
    'patch.edgecolor': 'black',
    'patch.linewidth': 0.7,
    'patch.force_edgecolor': True,
    # Legend
    'legend.fontsize': 8,
    'legend.title_fontsize': 9,
    'legend.framealpha': 0.0,
    'legend.edgecolor': '#444444',
    # Lines
    'lines.linewidth': 0.7,
    'lines.markersize': 3,
    # Saving
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
}
```

## Key conventions

- **No grid** — `axes.grid: False`; never call `ax.grid(...)`
- **Transparent legend** — `legend.framealpha: 0.0`; no explicit `fontsize=` overrides
- **Y headroom** — `ax.set_ylim(0, ax.get_ylim()[1] * 1.2)` after plotting
- **No explicit fontsize** on axis labels or titles — let `SA_RC` handle all sizes
- **Save formats** — save **`.png` only** (150 dpi). Regenerate a vector copy on demand only
  when a specific figure needs one for publication.
- **No titles**

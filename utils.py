import pandas as pd
import numpy as np
import os

class CustomEvaluator(object):
    def __init__(self, obj_fun, excel_path, var_names):
        self.iteration_count = 0  # 반복 횟수 카운터 초기화
        self.obj_fun = obj_fun

        self.excel_path = excel_path
        self.row_names = ["Iter"] + var_names
        self.data = pd.DataFrame(columns=self.row_names)

    def evaluate(self, x):
        print(f"현재 BO Iteration: {self.iteration_count}")
        val, optional_res = self.obj_fun(x)

        data_row = np.concatenate(([[self.iteration_count]], x, [[val]], optional_res), axis=1)
        self.data = self.data.append(pd.DataFrame(data_row, columns=self.row_names), ignore_index=True)

        writer_options = self._get_writer_options()
        with pd.ExcelWriter(self.excel_path, **writer_options) as writer:
            self.data.to_excel(writer, index=False, sheet_name="BO_Result")

        self.iteration_count += 1
        return val

    def save_best_result(self):
        best_row = self.data.nsmallest(1, "OBJ")
        writer_options = self._get_writer_options()
        with pd.ExcelWriter(self.excel_path, **writer_options) as writer:
            best_row.to_excel(writer, index=False, sheet_name="Best_Result")

    def read_past_results(self, past_path):
        try:
            with pd.ExcelFile(past_path) as xls:
                df = pd.read_excel(xls, sheet_name="BO_Result")
            return df
        except Exception:
            return pd.DataFrame(columns=self.row_names)

    def _get_writer_options(self):
        if os.path.exists(self.excel_path):
            writer_options = dict(mode='a', engine='openpyxl', if_sheet_exists='replace')
        else:
            writer_options = {}
        return writer_options

#%%
import matplotlib.pyplot as plt

def plot_heatmap(matrix, title=None, xlabel=None, ylabel=None,
                 xticklabels=None, yticklabels=None,
                 cmap='viridis', colorbar_label='Value',
                 show_values=True, figsize=None):
    """
    Creates a high-quality, annotated heatmap from a 2D NumPy matrix.

    Args:
        matrix (np.ndarray): The 2D matrix to plot.
        title (str, optional): Title for the plot.
        xlabel (str, optional): Label for the x-axis.
        ylabel (str, optional): Label for the y-axis.
        xticklabels (list of str, optional): Labels for the x-axis ticks.
        yticklabels (list of str, optional): Labels for the y-axis ticks.
        cmap (str, optional): The Matplotlib colormap to use. Defaults to 'viridis'.
        colorbar_label (str, optional): Label for the colorbar. Defaults to 'Value'.
        show_values (bool, optional): If True, shows the numeric value in each cell. Defaults to True.
        figsize (tuple, optional): (width, height) of the figure. If None, it auto-sizes.
    """

    # --- Input Validation ---
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
        raise ValueError("Input 'matrix' must be a 2D NumPy array.")

    num_rows, num_cols = matrix.shape

    # --- Auto-sizing Figure ---
    if figsize is None:
        # Auto-size based on matrix dimensions
        width = max(8, num_cols * 0.8)  # 0.8 inches per column, min 8
        height = max(5, num_rows * 0.6)  # 0.6 inches per row, min 5
        figsize = (width, height)

    # --- Plot Setup ---
    fig, ax = plt.subplots(figsize=figsize)

    # Get min/max for color scaling and text color logic
    vmin = matrix.min()
    vmax = matrix.max()
    v_mid = (vmin + vmax) / 2

    # Plot the heatmap using imshow()
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)

    # Add a colorbar
    fig.colorbar(im, ax=ax, label=colorbar_label)

    # --- Set Ticks and Labels ---

    # X-axis
    ax.set_xticks(np.arange(num_cols))
    if xticklabels:
        ax.set_xticklabels(xticklabels, rotation=45, ha='right')
    # (If no xticklabels, it will just show the indices 0, 1, 2...)

    # Y-axis
    ax.set_yticks(np.arange(num_rows))
    if yticklabels:
        ax.set_yticklabels(yticklabels)
    # (If no yticklabels, it will just show the indices 0, 1, 2...)

    # Set axis labels and title
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if title: ax.set_title(title)

    # --- Add Text Annotations (if enabled) ---
    if show_values:
        for i in range(num_rows):
            for j in range(num_cols):
                val = matrix[i, j]
                # Set text color to white for dark cells, black for light cells
                text_color = "white" if val < v_mid else "black"
                # Use 'g' for general format (avoids ".0" for integers)
                ax.text(j, i, f'{val:g}', ha='center', va='center', color=text_color)

    # --- Add Gridlines ---
    # Add minor ticks to create a grid *between* cells
    ax.set_xticks(np.arange(num_cols + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(num_rows + 1) - 0.5, minor=True)
    # Add a white grid on top of the minor ticks
    ax.grid(which="minor", color="white", linestyle='-', linewidth=2)
    # Hide the minor tick marks themselves
    ax.tick_params(which="minor", bottom=False, left=False)

    # --- Final Layout ---
    fig.tight_layout()
    plt.show()

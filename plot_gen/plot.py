import numpy as np
import matplotlib.pyplot as plt
import json

def main():
    # Parse the data
    with open('data.json', 'r') as file:
        data = json.load(file)

    # Generate for every available thread data
    thread_counts = [1, 2, 4, 8, 16, 32, 48]

    for thread_count in thread_counts:
        generate_plot(data, thread_count=thread_count, save_path=f"graphs/results_graph_{thread_count}", show=False)

def generate_plot(data, thread_count, save_path=None, show=True):
    """
    Generate a grouped bar chart for a specific thread count.

    Parameters:
    - data: dict loaded from JSON
    - thread_count: str or int, number of threads (e.g., "1", "2", etc.)
    - save_path: optional path to save the figure
    - show: whether to show the plot
    """

    # Ensure thread_count is a string
    thread_count = str(thread_count)

    # Set categories
    inputs = ['input_a', 'input_b', 'input_c', 'input_d']
    methods = ['eigen', 'matrixmultiply', 'cblas']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Optional color scheme

    # Collect the data
    y_values = []
    for input_key in inputs:
        method_values = [data[thread_count][input_key][method] for method in methods]
        y_values.append(method_values)

    # Convert to numpy array for plotting
    y_values = np.array(y_values)  # shape: (num_inputs, num_methods)

    # Bar chart settings
    x = np.arange(len(inputs))  # input positions
    width = 0.2  # width of each bar
    offsets = [-width, 0, width]  # for 3 bars per group

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, method in enumerate(methods):
        ax.bar(x + offsets[i], y_values[:, i], width, label=method, color=colors[i])

    # Labeling and formatting
    ax.set_ylabel('Time (seconds)')
    ax.set_xlabel('Input')
    ax.set_title(f'Performance by Method for Thread Count: {thread_count}')
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels([i.replace("input_", "").upper() for i in inputs])
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    # Save or Show
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

if __name__ == "__main__":
    main()
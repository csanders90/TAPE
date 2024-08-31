import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors

sns.set_theme(style="whitegrid", palette="pastel")
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Amiri"
})


def load_data(file_path):
    """
    Load and process the data from a CSV file.

    Args:
    - file_path (str): Path to the CSV file.

    Returns:
    - pd.DataFrame: Processed DataFrame with numerical AUC and MRR columns.
    """
    # Load the CSV file
    data = pd.read_csv(file_path)
    
    # Convert the 'AUC' and 'MRR' columns to numerical values by removing the "±" symbol and converting to float
    data['AUC'] = data['AUC'].str.split(' ± ').str[0].astype(float)
    data['MRR'] = data['MRR'].str.split(' ± ').str[0].astype(float)
    
    return data

# Load the data using the function
file_path = '/hkfs/work/workspace/scratch/cc7738-benchmark_tag/TAPE_chen/results/plots/plot1/complexity_cora.csv'
data = load_data(file_path)

# Extract the relevant columns
plot_data = data[['method', 'parameters', 'inference time', 'AUC', 'MRR']]

# Convert the 'AUC' and 'MRR' columns to numerical values by removing the "±" symbol and converting to float
plot_data['AUC'] = plot_data['AUC'].astype(float)
plot_data['MRR'] = plot_data['MRR'].astype(float)

# Data preparation
methods = plot_data['method'].tolist()
inference_time = plot_data['inference time'].tolist()
AUC_val = plot_data['AUC'].tolist()
MRR_val = plot_data['MRR'].tolist()
params = [i/100 for i in plot_data['parameters'].tolist()]
colors = np.random.rand(len(methods), 3)  # Generate random colors


fig, ax = plt.subplots(facecolor='white')
cmap = cm.get_cmap('Blues')
colors = plt.cm.Blues(np.linspace(0.3, 1, len(methods)))  # Adjust the range for desired darkness of blue

# Plot each method on a linear x-axis with a center marker and blue color scheme
for i, (method, time, metric, color, param) in enumerate(zip(methods, inference_time, AUC_val, colors, params)):
    # Plot the marker in the center of the bubble
    ax.scatter(i, metric, s=30, color='black', marker='x', label=method)

    offset_x = 0.1  # Adjust this value as needed
    offset_y = 0.1  # Adjust this value as needed

    ax.text(i + offset_x, metric + offset_y, f'{method}', fontsize=6, ha='right', va='top')
    ax.scatter(i, metric, s=param, color=1-color, alpha=0.5, edgecolors='w', linewidth=2)

# Set the x-ticks to correspond to each model
ax.set_xticks(range(len(methods)))
ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=12)

# Customize the plot
ax.set_xlabel('Total Time (min)', fontsize=14)
ax.set_ylabel('AUC (\%)', fontsize=14)
ax.grid(True, which="both", ls="--")

# Show the plot
plt.tight_layout()
plt.savefig('plot_benchmark.pdf')

# Plot each method
# for method, time, metric, color, param in zip(methods, inference_time, AUC_val, colors, params):
#     ax.text(time, metric, f'{method}\n{time}min, {metric}%', fontsize=12, ha='center', va='center')
#     ax.scatter(time, metric, s=param * 10, color=color, label=method, alpha=0.5, edgecolors='w', linewidth=2)

# for i, (method, time, metric, color, param) in enumerate(zip(methods, inference_time, AUC_val, colors, params)):
#     # ax.text(time, metric, f'{method}\n{time}min, {metric}%', fontsize=12, ha='center', va='center')
#     print(i)
#     ax.scatter(i, metric, s=30, color='black', marker='x', label=method)
#     ax.scatter(i, metric, s=param, color=color, alpha=0.5, edgecolors='w', linewidth=2)
    
    
# # Customize the plot
# ax.set_xlabel('Total Time (min)', fontsize=14)
# ax.set_ylabel('Accuracy (%)', fontsize=14)
# ax.grid(True, which="both", ls="--")

# # Show the plot
# plt.savefig('plot_benchmark.pdf')

# Now you can proceed with plotting using the `plot_data` DataFrame
# Data
# methods = ['GNN', 'LM', 'GLEM', 'TAPE (Ours)']
# inference_time = [4, 104, 551, 192]
# AUC_val = [70.83, 73.61, 76.57, 77.50]
# MRR_val = [70.83, 73.61, 76.57, 77.50]
# categories = ['Heuristic', 'Embeddings', 'GCNs', 'GCN2LP', 'LLM', 'Fine-Tuning']
# colors = np.random.rand(len(methods), 3)  # Generate random colors
# params = [10, 106, 1008, 10]

# # Create the plot
# fig, ax = plt.subplots(facecolor='white')
# # ax.set_facecolor('white')


# # Plot each method
# for method, time, metric, category, color, param in zip(methods, times, metrics, categories[:len(methods)], colors, params):
#     ax.text(time, metric, f'{method}\n{time}min, {metric}%', fontsize=12, ha='center', va='center')
#     ax.scatter(time, metric, s=param, color=color, label=category, alpha=0.5, edgecolors='w', linewidth=2)
    
# # Annotation for computation time reduction
# # ax.annotate('2.88× lower computation time',
# #             xy=(551, 76.57), xycoords='data',
# #             xytext=(192, 77.5), textcoords='data',
# #             arrowprops=dict(arrowstyle="->", lw=2, color='red'),
# #             fontsize=15, color='red')
# ax.xaxis.set_minor_formatter(plt.ScalarFormatter())
# ax.yaxis.set_minor_formatter(plt.ScalarFormatter())

# # Customize the plot
# ax.set_xlabel('Total Time (min)', fontsize=14)
# ax.set_ylabel('Accuracy (\%)', fontsize=14)
# ax.grid(True, which="both", ls="--")


# # Adjust y-axis limits to prevent cutoff

# # Adjust axis limits
# ax.set_xlim([1, 600])  # Extend x-axis limits
# ax.set_ylim([70, 80])  # Set y-axis limits

# #ax.xaxis.set_minor_formatter(plt.ScalarFormatter())
# ax.yaxis.set_minor_formatter(plt.ScalarFormatter())

# # Ensure the ticks and labels are visible
# ax.tick_params(axis='both', which='both', direction='in', length=6)
# ax.tick_params(axis='both', which='major', labelsize=12)
# ax.tick_params(axis='both', which='minor', labelsize=8)

# # Title and legend
# ax.set_title('Comparison of Different Methods', fontsize=16)
# ax.legend(title='Methods', fontsize=12, loc='lower right', bbox_to_anchor=(1.0, 0.0))

# # Save and show the plot
# plt.savefig('comparison.png')

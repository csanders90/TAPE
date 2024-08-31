import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Enable LaTeX formatting in Matplotlib
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'  # Use serif fonts
plt.rcParams['text.latex.preamble'] = r'\usepackage{helvet}'  # Use Helvetica font

# Function to extract the mean value from the string "mean ± std"
def extract_mean(value):
    try:
        mean_value = float(value.split(' ± ')[0])
    except:
        mean_value = None
    return mean_value

# Calculate group start indices dynamically
def calculate_group_start_indices(group_labels, bar_width, group_spacing_factor=7):
    current_position = 0
    group_start_indices = {}
    
    for group, models in group_labels.items():
        group_start_indices[group] = np.arange(len(models)) * (bar_width * 3) + current_position
        current_position = group_start_indices[group][-1] + bar_width * group_spacing_factor
    
    return group_start_indices

# Define the groups (including all available models)
group_labels = {
    'NCNC': [r"\textbf{bert-ncnc}", r"\textbf{minilm-ncnc}", r"\textbf{e5-ncnc}", r"\textbf{llama-ncnc}"],
    'Buddy': [r"\textbf{bert-buddy}", r"\textbf{minilm-buddy}", r"\textbf{e5-buddy}", r"\textbf{llama-buddy}", ],
    'NCN': [ r"\textbf{bert-ncn}", r"\textbf{minilm-ncn}", r"\textbf{e5-ncn}", r"\textbf{llama-ncn}"],
    'HLGNN': [ r"\textbf{bert-hlgnn}", r"\textbf{minilm-hlgnn}",  r"\textbf{e5-hlgnn}", r"\textbf{llama-hlgnn}",],
    'NeoGNN': [  r"\textbf{bert-neognn}", r"\textbf{minilm-neognn}", r"\textbf{e5-neognn}",  r"\textbf{llama-neognn}",]
}


for data_name in ['cora', 'arxiv_2023', 'pubmed']:
    
    file_path = f'/hkfs/work/workspace/scratch/cc7738-benchmark_tag/TAPE_chen/results/plots/plot2/{data_name}.csv'
    data_name = f'-{data_name}'
    data = pd.read_csv(file_path)
    
    # Apply the function to relevant columns
    for col in data.columns[1:]:  # Skip the 'Metric' column
        data[col] = data[col].apply(extract_mean)

    bar_width = 3.0  
    group_start_indices = calculate_group_start_indices(group_labels, bar_width)

    mrr_data = []
    hits50_data = []
    auc_data = []
    labels = []
    x_positions = []

    # Populate data for plotting
    for group, models in group_labels.items():
        for i, model in enumerate(models):
            mrr_value = data.loc[data['Metric'] == model.split('{')[1].split('}')[0] + data_name, 'MRR'].values[0] if not data.loc[data['Metric'] == model.split('{')[1].split('}')[0] + data_name, 'MRR'].empty else 0
            hits50_value = data.loc[data['Metric'] == model.split('{')[1].split('}')[0] + data_name, 'Hits@50'].values[0] if not data.loc[data['Metric'] == model.split('{')[1].split('}')[0] + data_name, 'Hits@50'].empty else 0
            auc_value = data.loc[data['Metric'] == model.split('{')[1].split('}')[0] + data_name, 'AUC'].values[0] if not data.loc[data['Metric'] == model.split('{')[1].split('}')[0] + data_name, 'AUC'].empty else 0
            
            mrr_data.append(mrr_value)
            hits50_data.append(hits50_value)
            auc_data.append(auc_value)

            x_positions.append(group_start_indices[group][i])

    labels = [r"\textbf{Bert}",  r"\textbf{Minilm}",  r"\textbf{e5}", r"\textbf{Llama3}", ] * 5
    cmap = cm.get_cmap('Blues')
    norm = mcolors.Normalize(vmin=min(mrr_data + hits50_data + auc_data), vmax=max(mrr_data + hits50_data + auc_data))

    # Plotting
    plt.figure(figsize=(55, 50))  

    # Color the bars using the colormap
    plt.bar(x_positions, mrr_data, bar_width, label=r'\textbf{MRR}', color=cmap(norm(mrr_data)))
    plt.bar([x + bar_width for x in x_positions], hits50_data, bar_width, label=r'\textbf{Hits@50}', color=cmap(norm(hits50_data)))
    plt.bar([x + 2 * bar_width for x in x_positions], auc_data, bar_width, label=r'\textbf{AUC}', color=cmap(norm(auc_data)))

    fs = 115
    for group_name, indices in group_start_indices.items():
        plt.text(np.mean(indices) + bar_width, np.mean(mrr_data + hits50_data + auc_data) * 0.75, f"{group_name}", 
                fontsize=fs, ha='center', va='bottom', fontweight='bold')

    plt.title(r'\textbf{GCN2LPs initialized by Different LLMs}', fontsize=164)
    plt.xticks([x + bar_width for x in x_positions], labels, rotation=45, ha="right", fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.legend(fontsize=85, framealpha=0.4, loc='upper right')  # Added transparency to the legend
    plt.grid(True)

    plt.savefig(f'{data_name}_plot.pdf')
    plt.show()
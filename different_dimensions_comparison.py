import argparse
import os
import sqlite3

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import re

def extract_numeric_part(dataset_name):
    """
    Extracts the first integer found in the dataset name.
    For example, 'glove-10-angular' -> 10.
    If no integer is found, returns a large number as a fallback
    (or you could return 0, depending on your needs).
    """
    match = re.search(r'(\d+)', dataset_name)
    if match:
        return int(match.group(1))
    return 999999999  # fallback if no digits

def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error(f"The file {arg} does not exist!")
    else:
        return arg

def fetch_distance_computations(db_path, prefix, git_commit_hash=None, num_clusters=0.5, kb_per_point=1):
    """
    Fetches distance computation data for datasets whose names start with the given prefix.
    It unions the CLANN and Puffinn queries.
    """
    conn = sqlite3.connect(db_path)
    
    clann_query = f"""
    SELECT 
        'clustered' AS method, 
        cr.dataset, 
        cr.k, 
        cr.delta, 
        cr.num_clusters AS num_cluster_factor,
        cr.kb_per_point,
        crq.distance_computations AS value
    FROM clann_results cr
    JOIN clann_results_query crq ON (
        cr.num_clusters = crq.num_clusters AND 
        cr.kb_per_point = crq.kb_per_point AND 
        cr.k = crq.k AND 
        cr.delta = crq.delta AND 
        cr.dataset = crq.dataset AND 
        cr.git_commit_hash = crq.git_commit_hash
    )
    WHERE cr.dataset LIKE '{prefix}%'
      AND cr.num_clusters BETWEEN {num_clusters} - 1e-6 AND {num_clusters} + 1e-6
      AND cr.kb_per_point = {kb_per_point}
    """
    if git_commit_hash:
        clann_query += f" AND cr.git_commit_hash = '{git_commit_hash}'"
    
    puffinn_query = f"""
    SELECT 
        'puffinn' AS method, 
        pr.dataset, 
        pr.k, 
        pr.delta, 
        NULL AS num_cluster_factor,
        pr.kb_per_point,
        prq.distance_computations AS value
    FROM puffinn_results pr
    JOIN puffinn_results_query prq ON (
        pr.kb_per_point = prq.kb_per_point AND 
        pr.k = prq.k AND 
        pr.delta = prq.delta AND 
        pr.dataset = prq.dataset
    )
    WHERE pr.dataset LIKE '{prefix}%'
    """
    
    union_query = f"{clann_query} UNION ALL {puffinn_query}"
    data = pd.read_sql_query(union_query, conn)
    conn.close()
    return data

def plot_distance_computations(data, output_folder, prefix):
    """
    Plots side-by-side box plots for CLANN and Puffinn distance computations per dataset.
    Also overlays a line plot showing the mean CLANN distance computations.
    """
    sns.set_theme(style="whitegrid")
    
    datasets_sorted = sorted(data['dataset'].unique(), key=extract_numeric_part)
    
    plt.figure(figsize=(16, 8))
    ax = sns.boxplot(
        x='dataset', 
        y='value', 
        hue='method', 
        data=data,
        order=datasets_sorted,
        palette='Set3'
    )

    plt.yscale('log')
    
    # Compute the mean values for CLANN ("clustered") results per dataset
    clann_data = data[data['method'] == 'clustered']
    clann_means = clann_data.groupby('dataset')['value'].mean().reindex(datasets_sorted)
    
    # Since the x-axis is categorical, use integer positions for the line plot
    x_positions = range(len(datasets_sorted))
    plt.plot(x_positions, clann_means.values, color='black', marker='o',
             linestyle='-', linewidth=2, label='CLANN mean')
    
    plt.title(f"Distance Computations Comparison for datasets starting with '{prefix}'", fontsize=16)
    plt.xlabel("Dataset", fontsize=12)
    plt.ylabel("Distance Computations", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Merge the legend with a custom entry for the CLANN mean line
    handles, labels = ax.get_legend_handles_labels()
    custom_line = Line2D([0], [0], color='black', marker='o', linestyle='-', linewidth=2, label='CLANN mean')
    handles.append(custom_line)
    labels.append('CLANN mean')
    ax.legend(handles=handles, labels=labels)
    
    # Save the plot to the specified output folder
    output_file = os.path.join(output_folder, f"distance_comparisons_{prefix}.png")
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Plot saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Visualize distance computations for datasets with a common prefix."
    )
    parser.add_argument(
        "-db",
        dest="database_path",
        required=True,
        help="Path to SQLite database",
        type=lambda x: is_valid_file(parser, x)
    )
    parser.add_argument(
        "-output",
        dest="output_folder",
        default=".",
        help="Output folder for the plot"
    )
    parser.add_argument(
        "-prefix",
        dest="dataset_prefix",
        required=True,
        help="Common prefix for dataset names (e.g., 'glove-')"
    )
    parser.add_argument(
        "-commit",
        dest="git_commit_hash",
        help="Git commit hash for CLANN results (optional)"
    )
    args = parser.parse_args()
    
    data = fetch_distance_computations(args.database_path, args.dataset_prefix, args.git_commit_hash)
    if data.empty:
        print("No data found for the given dataset prefix.")
        return
    
    plot_distance_computations(data, args.output_folder, args.dataset_prefix)

if __name__ == "__main__":
    main()

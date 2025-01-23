import argparse
import sqlite3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils.utils import is_valid_file

def fetch_distance_computations(db_path, git_commit_hash=None):
    conn = sqlite3.connect(db_path)
    
    # Query to fetch distance computations from both methods
    clann_query = """
    SELECT 
        'clustered' as method, 
        cr.dataset, 
        cr.k, 
        cr.delta, 
        cr.num_clusters as num_cluster_factor,
        crq.distance_computations as value
    FROM clann_results cr
    JOIN clann_results_query crq ON (
        cr.num_clusters = crq.num_clusters AND 
        cr.kb_per_point = crq.kb_per_point AND 
        cr.k = crq.k AND 
        cr.delta = crq.delta AND 
        cr.dataset = crq.dataset AND 
        cr.git_commit_hash = crq.git_commit_hash
    )
    """
    
    puffinn_query = """
    SELECT 
        'puffinn' as method, 
        pr.dataset, 
        pr.k, 
        pr.delta, 
        NULL as num_cluster_factor,
        prq.distance_computations as value
    FROM puffinn_results pr
    JOIN puffinn_results_query prq ON (
        pr.kb_per_point = prq.kb_per_point AND 
        pr.k = prq.k AND 
        pr.delta = prq.delta AND 
        pr.dataset = prq.dataset
    )
    """
    
    # Add git commit hash filter for CLANN if provided
    if git_commit_hash:
        clann_query += f" AND cr.git_commit_hash = '{git_commit_hash}'"
    
    data = pd.read_sql_query(f"{clann_query} UNION ALL {puffinn_query}", conn)
    conn.close()
    return data

def plot_distance_computations(data, output_folder):
    sns.set_theme(style="whitegrid")
    
    # First, get all unique configurations for CLANN
    clann_configs = data[data['method'] == 'clustered'].drop_duplicates(subset=['dataset', 'k', 'delta', 'num_cluster_factor'])
    
    for _, clann_config in clann_configs.iterrows():
        # Find matching data for CLANN and PUFFINN
        matching_data = data[
            (data['dataset'] == clann_config['dataset']) & 
            (data['k'] == clann_config['k']) & 
            (data['delta'] == clann_config['delta']) &
            ((data['num_cluster_factor'] == clann_config['num_cluster_factor']) | (data['num_cluster_factor'].isna()))
        ]
        
        # Ensure we have both CLANN and PUFFINN data
        if len(matching_data['method'].unique()) == 2:
            # Now we separate data by method
            clustered_data = matching_data[matching_data['method'] == 'clustered']
            puffinn_data = matching_data[matching_data['method'] == 'puffinn']
            
            # Create plots only if both methods are present
            if len(clustered_data) > 0 and len(puffinn_data) > 0:
                fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

                # Plot CLANN data
                sns.boxplot(
                    y='value', 
                    data=clustered_data, 
                    ax=axes[0], 
                    color='lightblue'
                )
                axes[0].set_title(
                    f"Clustered (Cluster Factor: {clann_config['num_cluster_factor']:.2})",
                    fontsize=14
                )

                # Plot PUFFINN data
                sns.boxplot(
                    y='value', 
                    data=puffinn_data, 
                    ax=axes[1], 
                    color='lightgreen'
                )
                axes[1].set_title("PUFFINN", fontsize=14)

                axes[0].set_yscale('log')
                axes[1].set_yscale('log')
                axes[0].set_ylabel('Distance Computations (log scale)', fontsize=12)
                
                plot_title = f"Distance Computations Comparison\nDataset: {clann_config['dataset']}, k: {clann_config['k']}, delta: {clann_config['delta']:.2}"
                plt.suptitle(plot_title, fontsize=16)
                plt.tight_layout()
                plt.subplots_adjust(top=0.85)
                
                # Create filename-safe version of plot title
                safe_filename = f"distance_computations_{clann_config['dataset']}_k{clann_config['k']}_delta{clann_config['delta']:.2}_cluster{clann_config['num_cluster_factor']:.2}.png"
                safe_filename = "".join(x for x in safe_filename if x.isalnum() or x in "._-").lower()
                
                output_file = f"{output_folder}/{safe_filename}"
                plt.savefig(output_file, bbox_inches='tight', dpi=300)
                plt.close()
        else:
            print("FUCK")

def main():
    parser = argparse.ArgumentParser(description="Visualize distance computations from SQLite database")
    parser.add_argument("-db", dest="database_path", required=True, help="Path to SQLite database", type=lambda x: is_valid_file(parser, x))
    parser.add_argument("-output", dest="output_folder", default="", help="Output folder for plots")
    parser.add_argument("-commit", dest="git_commit_hash", help="Git commit hash for CLANN results")
    args = parser.parse_args()
    
    distance_data = fetch_distance_computations(args.database_path, args.git_commit_hash)
    plot_distance_computations(distance_data, args.output_folder)

if __name__ == "__main__":
    main()
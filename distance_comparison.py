import argparse
import sqlite3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils.utils import is_valid_file

def fetch_distance_computations(db_path, git_commit_hash=None):
    conn = sqlite3.connect(db_path)
    
    clann_query = """
    SELECT 
        'clustered' as method, 
        cr.dataset, 
        cr.k, 
        cr.delta, 
        cr.num_clusters as num_cluster_factor,
        cr.kb_per_point,
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
        pr.kb_per_point,
        prq.distance_computations as value
    FROM puffinn_results pr
    JOIN puffinn_results_query prq ON (
        pr.kb_per_point = prq.kb_per_point AND 
        pr.k = prq.k AND 
        pr.delta = prq.delta AND 
        pr.dataset = prq.dataset
    )
    """
    
    if git_commit_hash:
        clann_query += f" AND cr.git_commit_hash = '{git_commit_hash}'"
    
    data = pd.read_sql_query(f"{clann_query} UNION ALL {puffinn_query}", conn)
    conn.close()
    return data

def plot_distance_computations(data, output_folder):
    sns.set_theme(style="whitegrid")
    
    grouping_cols = ['dataset', 'k', 'delta']
    
    unique_configs = data.groupby(grouping_cols).apply(lambda x: x['method'].nunique() > 1).reset_index()
    unique_configs = unique_configs[unique_configs[0]]['dataset k delta'.split()]
    
    for _, config in unique_configs.iterrows():
        config_data = data[
            (data['dataset'] == config['dataset']) & 
            (data['k'] == config['k']) & 
            (data['delta'] == config['delta'])
        ]
        
        # Round num_cluster_factor to 2 decimals for legend
        if 'num_cluster_factor' in config_data.columns:
            config_data['cluster_factor_rounded'] = config_data['num_cluster_factor'].apply(lambda x: f'{x:.2f}' if pd.notnull(x) else 'PUFFINN')
        
        plt.figure(figsize=(16, 8))
        
        ax = sns.boxplot(
            x='cluster_factor_rounded', 
            y='value', 
            hue='method',
            data=config_data,
            palette='Set3'
        )
        
        plt.yscale('log')
        plt.title(f"Distance Computations: {config['dataset']}, k={config['k']}, delta={config['delta']:.2}", fontsize=16)
        plt.ylabel('Distance Computations (log scale)', fontsize=12)
        plt.xlabel('Cluster Factor', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        safe_filename = f"distance_computations_{config['dataset']}_k{config['k']}_delta{config['delta']:.2}.png"
        safe_filename = "".join(x for x in safe_filename if x.isalnum() or x in "._-").lower()
        
        output_file = f"{output_folder}/{safe_filename}"
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()

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
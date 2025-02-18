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
        cr.num_tables,
        crq.distance_computations as value
    FROM clann_results cr
    JOIN clann_results_query crq ON (
        cr.num_clusters = crq.num_clusters AND 
        cr.num_tables = crq.num_tables AND 
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
        pr.num_tables,
        prq.distance_computations as value
    FROM puffinn_results pr
    JOIN puffinn_results_query prq ON (
        pr.num_tables = prq.num_tables AND 
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
    
    grouping_cols = ['dataset', 'k', 'delta', 'num_tables']
    
    method_counts = data.groupby(grouping_cols)['method'].nunique()
    unique_configs = method_counts[method_counts > 1].reset_index()
    
    for _, config in unique_configs.iterrows():
        # Create boolean mask for filtering
        mask = (
            (data['dataset'] == config['dataset']) & 
            (data['k'] == config['k']) & 
            (data['delta'] == config['delta']) &
            (data['num_tables'] == config['num_tables'])
        )
        
        # Create a copy of the filtered data to avoid SettingWithCopyWarning
        config_data = data[mask].copy()
        config_data.loc[:, 'cluster_factor_rounded'] = config_data['num_cluster_factor'].apply(
            lambda x: f'{x:.2f}' if pd.notnull(x) else 'PUFFINN'
        )
        
        plt.figure(figsize=(16, 8))
        
        ax = sns.boxplot(
            x='cluster_factor_rounded', 
            y='value', 
            hue='method',
            data=config_data,
            palette='Set3'
        )
        
        #plt.yscale('log')
        plt.title(
            f"Distance Computations: {config['dataset']}, k={config['k']}, "
            f"delta={config['delta']:.2}, num_tables={config['num_tables']}", 
            fontsize=16
        )
        plt.ylabel('Distance Computations (log scale)', fontsize=12)
        plt.xlabel('Cluster Factor', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        safe_filename = (
            f"distance_computations_{config['dataset']}_k{config['k']}_"
            f"delta{config['delta']:.2}_kb{config['num_tables']}.png"
        )
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
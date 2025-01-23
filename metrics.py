import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sqlite3
from pathlib import Path
import pandas as pd
import math

from utils.utils import is_valid_file
from argparse import ArgumentParser

class MetricsVisualizer:
    def __init__(self):
        self.results_data = None
        self.input_filename = ""
        self.git_commit_hash = ""
        self.query_data = None

    def load_data(self, db_path, git_commit_hash):
        """Load metrics data from SQLite database for a specific git commit."""
        self.input_filename = Path(db_path).stem
        self.git_commit_hash = git_commit_hash

        # Connect to the database
        conn = sqlite3.connect(db_path)
        
        try:
            # Fetch overall results with all relevant parameters
            results_query = """
            SELECT num_clusters, k, delta, dataset, dataset_len,
                   memory_used_bytes, recall_mean, recall_std 
            FROM clann_results 
            WHERE git_commit_hash = ?
            """
            self.results_data = pd.read_sql_query(results_query, conn, params=(git_commit_hash,))
            
            # Fetch per-query cluster data for each configuration
            query_data_list = []
            for _, config_row in self.results_data.iterrows():
                query = """
                SELECT ? as num_clusters, ? as k, ? as delta, ? as dataset,
                       query_idx, cluster_idx, n_candidates, 
                       (SELECT COUNT(*) FROM clann_results_query_cluster qc2 
                        WHERE qc2.num_clusters = qc.num_clusters 
                        AND qc2.kb_per_point = qc.kb_per_point 
                        AND qc2.k = qc.k 
                        AND qc2.delta = qc.delta 
                        AND qc2.dataset = qc.dataset 
                        AND qc2.git_commit_hash = qc.git_commit_hash 
                        AND qc2.query_idx = qc.query_idx) as total_candidates
                FROM clann_results_query_cluster qc
                WHERE qc.git_commit_hash = ? 
                  AND qc.num_clusters = ?
                  AND qc.k = ?
                  AND qc.delta = ?
                  AND qc.dataset = ?
                """
                query_data = pd.read_sql_query(query, conn, 
                    params=(
                        config_row['num_clusters'], config_row['k'], config_row['delta'], config_row['dataset'],
                        git_commit_hash,
                        config_row['num_clusters'], config_row['k'], config_row['delta'], config_row['dataset']
                    )
                )
                query_data_list.append(query_data)
            
            # Combine all query data
            self.query_data = pd.concat(query_data_list, ignore_index=True)
        
        finally:
            conn.close()

    def calculate_cluster_statistics(self, config_query_data):
        """Calculate statistics about cluster visits and point distribution."""
        if config_query_data is None:
            return {}

        # Total clusters visited across all queries (counting each query separately)
        total_clusters_visited = config_query_data.groupby(['query_idx', 'cluster_idx']).ngroups

        # Clusters with 0 candidates (visited but no candidates) across all queries
        clusters_with_few_points = config_query_data.groupby(['query_idx', 'cluster_idx'])['n_candidates'].sum().eq(0).sum()

        return {
            'total_visited_clusters': total_clusters_visited,
            'clusters_with_few_points': clusters_with_few_points,
            'few_points_ratio': f"{clusters_with_few_points}/{total_clusters_visited}",
            'few_points_percentage': (clusters_with_few_points / total_clusters_visited) * 100 if total_clusters_visited > 0 else 0
        }

    def add_stats_text(self, plt, config_row, config_query_data, recall_mean, recall_std, recall_stats=None):
        """Add parameter and recall statistics to the plot."""
        # Calculate cluster statistics
        if recall_stats is None:
            recall_stats = self.calculate_cluster_statistics(config_query_data)
        
        num_clusters = math.floor(config_row['num_clusters'] * math.sqrt(config_row['dataset_len']))
        params_text = '\n'.join([
            f'K={config_row["k"]}',
            f'num_centers_factor={config_row["num_clusters"]:.2f}',
            f'num_centers={num_clusters}',
            f'δ={config_row["delta"]:.2f}',
            f'total memory (GB)={config_row["memory_used_bytes"] / (1024**3):.2f}',
            '\n',
            f'Recall: {recall_mean:.2f}',
            f'std: {recall_std:.2f}',
            '\n',
            f'Clusters with 0 candidates: {recall_stats["few_points_ratio"]}',
            f'({recall_stats["few_points_percentage"]:.1f}%)'
        ])

        plt.text(0.95, 0.95, params_text,
                transform=plt.gca().transAxes,
                verticalalignment='top',
                horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    def plot_metrics(self, output_folder):
        """Generate violin and cluster distribution plots for each configuration."""
        if self.query_data is None or self.results_data is None:
            print("No data to visualize")
            return

        # Create plots for each configuration
        for _, config_row in self.results_data.iterrows():
            # Filter data for this specific configuration
            config_query_data = self.query_data[
                (self.query_data['num_clusters'] == config_row['num_clusters']) &
                (self.query_data['k'] == config_row['k']) &
                (self.query_data['delta'] == config_row['delta']) &
                (self.query_data['dataset'] == config_row['dataset'])
            ]

            # Prepare plot data
            plot_data = config_query_data[['cluster_idx', 'n_candidates']]
            clusters_per_query = config_query_data.groupby('query_idx')['cluster_idx'].nunique()
            
            # Violin Plot
            plt.figure(figsize=(12, 6))
            max_candidates = plot_data['n_candidates'].max()
            
            sns.violinplot(data=plot_data, 
                           x="cluster_idx", 
                           y="n_candidates", 
                           inner="quart", 
                           palette="vlag",
                           cut=0)
            
            plt.ylim(0, max_candidates * 1.1)
            mean_visited = clusters_per_query.mean()
            
            plt.title(f"Candidates per Cluster\n"
                      f"Dataset: {config_row['dataset']}, "
                      f"K={config_row['k']}, δ={config_row['delta']:.2f}, "
                      f"Mean Visited Clusters: {mean_visited:.2f}", 
                      fontsize=12)
            plt.xlabel("Cluster Index")
            plt.ylabel("Number of Candidates")
            plt.grid(True)
            
            # Add stats text
            self.add_stats_text(plt, config_row, config_query_data, 
                                config_row['recall_mean'], config_row['recall_std'])
            
            output_dir = Path(output_folder)
            output_dir.mkdir(exist_ok=True)
            plt.savefig(output_dir / f"{self.input_filename}_{config_row['dataset']}_n{config_row['num_clusters']:.2f}_violin.png", 
                        bbox_inches='tight', dpi=300)
            plt.close()

            # Visited Clusters Distribution
            plt.figure(figsize=(12, 6))

            num_clusters = math.floor(config_row['num_clusters'] * math.sqrt(config_row['dataset_len']))
            sns.histplot(data=clusters_per_query, 
                         bins=range(num_clusters + 2), 
                         stat='probability',
                         discrete=True)
            
            plt.axvline(x=mean_visited, color='r', linestyle='--', label='Mean')
            
            stats_text = (
                f'Mean: {mean_visited:.2f}\n'
                f'Median: {clusters_per_query.median():.2f}\n'
                f'Std: {clusters_per_query.std():.2f}\n'
                f'Min: {clusters_per_query.min()}\n'
                f'Max: {clusters_per_query.max()}\n'
            )
            plt.text(0.05, 0.9, stats_text,
                     transform=plt.gca().transAxes,
                     verticalalignment='top',
                     horizontalalignment='left',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.title(f"Visited Clusters Distribution\n"
                      f"Dataset: {config_row['dataset']}, "
                      f"K={config_row['k']}, δ={config_row['delta']:.2f}", 
                      fontsize=12)
            plt.xlabel("Number of Clusters Visited")
            plt.ylabel("Percentage")
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Add stats text
            cluster_stats = self.calculate_cluster_statistics(config_query_data)
            self.add_stats_text(plt, config_row, config_query_data, 
                                config_row['recall_mean'], config_row['recall_std'], 
                                cluster_stats)
            
            plt.savefig(output_dir / f"{self.input_filename}_{config_row['dataset']}_n{config_row['num_clusters']:.2f}_visited_clusters.png", 
                        bbox_inches='tight', dpi=300)
            plt.close()

    def plot_all(self, output_folder):
        """Generate all visualizations."""
        self.plot_metrics(output_folder)


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="visualizer.py",
        description="Visualizes metrics from a SQLite database for multiple configurations.",
    )
    parser.add_argument("-input", dest="db_filename", required=True,
                        help="Path to the SQLite database file", metavar="FILE",
                        type=lambda x: is_valid_file(parser, x))
    parser.add_argument("-commit", dest="git_commit_hash", required=True,
                        help="Git commit hash to analyze", type=str)
    parser.add_argument("-output", dest="output_folder", required=False,
                        help="Output folder", metavar="FOLDER", 
                        default="", type=str)
    args = parser.parse_args()

    metrics_aggr = MetricsVisualizer()
    metrics_aggr.load_data(args.db_filename, args.git_commit_hash)

    metrics_aggr.plot_all(args.output_folder)
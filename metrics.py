"""
Input a csv file with this header:

param,total_clusters,<int>
param,K,<int>
param,delta,<float>
param,total_memory,<int>
res,recall_mean,<float>
res,recall_std,<float>
query,cluster_idx,n_candidates,total_candidates
<data>

Outputs two plots, one that displays the number of candidates per cluster the other how many clusters are visited
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import csv
from pathlib import Path
import pandas as pd

from utils.utils import is_valid_file
from argparse import ArgumentParser

class MetricsVisualizer:
    def __init__(self):
        self.params = {
            'total_clusters': 0,
            'K': 0,
            'delta': 0.0,
            'total_memory': 0
        }
        self.res = {
            'recall_mean': 0.0,
            'recall_std': 0.0
        }
        self.data = None

        self.input_filename = ""

    def load_data(self, file_path):
        """Load metrics data from CSV file."""
        self.input_filename = Path(file_path).stem

        # Initialize lists for parameters and results
        param_rows = []
        res_rows = []
        data_rows = []
        
        with open(file_path) as file:
            reader = csv.reader(file)
            
            # Parse file
            for row in reader:
                if row[0] == "param":
                    param_rows.append(row)
                elif row[0] == "res":
                    res_rows.append(row)
                elif row[0] == "query":
                    continue
                else:
                    data_rows.append(row)
        
        # Process parameters
        for row in param_rows:
            param_name = row[1]
            if param_name in self.params:
                self.params[param_name] = int(row[2]) if param_name != 'delta' else float(row[2])
        
        # Process results
        for row in res_rows:
            param_name = row[1]
            if param_name in self.res:
                self.res[param_name] = float(row[2])
        
        # Convert data rows to DataFrame
        self.data = pd.DataFrame(data_rows, columns=['query', 'cluster_idx', 'n_candidates', 'total_candidates'])
        self.data = self.data.astype({
            'query': int,
            'cluster_idx': int,
            'n_candidates': int,
            'total_candidates': int
        })

    def calculate_cluster_statistics(self):
        """Calculate statistics about cluster visits and point distribution."""
        if self.data is None:
            return {}

        # Total clusters visited across all queries (counting each query separately)
        total_clusters_visited = self.data.groupby(['query', 'cluster_idx']).ngroups

        # Clusters with 0 candidates (visited but no candidates) across all queries
        clusters_with_few_points = self.data.groupby(['query', 'cluster_idx'])['n_candidates'].sum().eq(0).sum()

        return {
            'total_visited_clusters': total_clusters_visited,
            'clusters_with_few_points': clusters_with_few_points,
            'few_points_ratio': f"{clusters_with_few_points}/{total_clusters_visited}",
            'few_points_percentage': (clusters_with_few_points / total_clusters_visited) * 100
        }


    def calculate_mean_visited_clusters(self):
        """Calculate mean number of clusters visited per query."""
        if self.data is None:
            return 0
        
        clusters_per_query = self.data.groupby('query')['cluster_idx'].nunique()
        return clusters_per_query.mean()

    def prepare_plot_data(self):
        """Prepare data for plotting."""
        if self.data is None:
            return pd.DataFrame()
        
        return self.data[['cluster_idx', 'n_candidates']]

    def add_stats_text(self, plt):
        """Add parameter and recall statistics to the plot."""
        # Calculate cluster statistics
        cluster_stats = self.calculate_cluster_statistics()
        
        params_text = '\n'.join([
            f'K={self.params["K"]}',
            f'num_centers={self.params["total_clusters"]}',
            f'δ={self.params["delta"]:.2f}',
            f'total memory (GB)={self.params["total_memory"] / (1024**3):.2f}',
            '\n',
            f'Recall: {self.res["recall_mean"]:.2f}',
            f'std: {self.res["recall_std"]:.2f}',
            '\n',
            f'Clusters with 0 candidates: {cluster_stats["few_points_ratio"]}',
            f'({cluster_stats["few_points_percentage"]:.1f}%)'
        ])

        plt.text(0.95, 0.95, params_text,
                transform=plt.gca().transAxes,
                verticalalignment='top',
                horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    def plot_violin(self, output_folder):
        """Generate a violin plot visualization with clipped distributions."""
        if self.data is None:
            print("No data to visualize")
            return

        plot_data = self.prepare_plot_data()
        mean_visited = self.calculate_mean_visited_clusters()
        
        plt.figure(figsize=(12, 6))
        
        # Get the max value for setting ylim
        max_candidates = plot_data['n_candidates'].max()
        
        # Create violin plot with clipped distributions
        sns.violinplot(data=plot_data, 
                      x="cluster_idx", 
                      y="n_candidates", 
                      hue="cluster_idx",
                      legend=False,
                      inner="quart", 
                      palette="vlag",
                      cut=0) 
        
        plt.ylim(0, max_candidates * 1.1) 
        
        plt.title(f"Candidates per Cluster Distribution (10,000 Queries)\nMean Visited Clusters: {mean_visited:.2f}", 
                 fontsize=16)
        plt.xlabel("Cluster Index", fontsize=14)
        plt.ylabel("Number of Candidates", fontsize=14)
        plt.grid(True)
        
        self.add_stats_text(plt)
        
        output_dir = Path(output_folder)
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / f"{self.input_filename}_violin.png", 
                   bbox_inches='tight', dpi=300)
        plt.close()

    def plot_visited_clusters_distribution(self, output_folder):
        """Generate a visualization of the visited clusters distribution."""
        if self.data is None:
            print("No data to visualize")
            return

        # Count unique clusters visited per query
        clusters_per_query = self.data.groupby('query')['cluster_idx'].nunique()
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Create distribution plot
        sns.histplot(data=clusters_per_query, 
                    bins=range(self.params['total_clusters'] + 2), 
                    stat='probability',
                    discrete=True)
        
        # Add mean line
        mean_visited = clusters_per_query.mean()
        plt.axvline(x=mean_visited, color='r', linestyle='--', 
                   label=f'Mean')
        
        # Calculate cluster statistics
        cluster_stats = self.calculate_cluster_statistics()
        
        # Add statistics text
        stats_text = (
            f'Mean: {mean_visited:.2f}\n'
            f'Median: {clusters_per_query.median():.2f}\n'
            f'Std: {clusters_per_query.std():.2f}\n'
            f'Min: {clusters_per_query.min()}\n'
            f'Max: {clusters_per_query.max()}\n'
        )
        plt.text(0.13, 0.9, stats_text,
                transform=plt.gca().transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        self.add_stats_text(plt)
        
        # Customize plot
        plt.title(f"Distribution of Visited Clusters per Query\n(Total Queries: {len(clusters_per_query)})", 
                 fontsize=16)
        plt.xlabel("Number of Clusters Visited", fontsize=14)
        plt.ylabel("Percentage", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Ensure x-axis shows all possible values
        plt.xlim(-0.5, self.params['total_clusters'] + 0.5)
        
        # Save plot
        output_dir = Path(output_folder)
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / f"{self.input_filename}_visited_clusters.png", 
                   bbox_inches='tight', dpi=300)
        plt.close()

    def plot_all(self, output_folder):
        """Generate all visualizations."""
        self.plot_violin(output_folder)
        self.plot_visited_clusters_distribution(output_folder)


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="visualizer.py",
        description="Visualizes the relationship between hyperparameters and collision statistics from a CSV file, generating corresponding plots.",
    )
    parser.add_argument("-input", dest="csv_filename", required=True,
                        help="Path to the CSV file containing run data", metavar="FILE",
                        type=lambda x: is_valid_file(parser, x))
    parser.add_argument("-output", dest="output_folder", required=False,
                        help="Output folder", metavar="FOLDER", 
                        default="", type=str)
    args = parser.parse_args()

    metrics_aggr = MetricsVisualizer()
    metrics_aggr.load_data(args.csv_filename)

    metrics_aggr.plot_all(args.output_folder)
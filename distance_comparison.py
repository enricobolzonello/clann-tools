from argparse import ArgumentParser
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils.utils import is_valid_file

def main(filepath, output_folder):
    data = pd.read_csv(filepath)

    methods_of_interest = ['clustered', 'puffinn']
    filtered_data = data[data['method'].isin(methods_of_interest)]
    sns.set_theme(style="whitegrid")

    for config_id, config_data in filtered_data.groupby('config_id'):
        shared_params = config_data.iloc[0][['dataset', 'k', 'delta']].to_dict()

        clustered_data = config_data[config_data['method'] == 'clustered']
        puffinn_data = config_data[config_data['method'] == 'puffinn']

        fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

        sns.boxplot(y='value', data=clustered_data, ax=axes[0], color='lightblue')
        axes[0].set_title('Clustered', fontsize=14)
        axes[0].set_ylabel('Distance Computations (log scale)', fontsize=12)
        axes[0].set_yscale('log')

        sns.boxplot(y='value', data=puffinn_data, ax=axes[1], color='lightgreen')
        axes[1].set_title('Puffinn', fontsize=14)
        axes[1].set_yscale('log')

        plt.suptitle(
            f"Comparison of Distance Computations by Method\n"
            f"Config ID: {config_id}, Dataset: {shared_params['dataset']}, "
            f"k: {shared_params['k']}, delta: {shared_params['delta']}",
            fontsize=16
        )
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)

        output_file = f"{output_folder}distance_computations_config_{config_id}.png"
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()

if __name__ == "__main__":
    parser = ArgumentParser(
        prog="distance_comparison.py",
        description="Visualize distance computations by method for each config_id.",
    )
    parser.add_argument("-input", dest="csv_filepath", required=True,
                        help="Path to the CSV file", metavar="FILE",
                        type=lambda x: is_valid_file(parser, x))
    parser.add_argument("-output", dest="output_folder", required=False,
                        help="Output folder", metavar="FOLDER", 
                        default="", type=str)
    args = parser.parse_args()
    
    main(args.csv_filepath, args.output_folder)

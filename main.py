import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
import warnings

from data_utils import load_and_prepare_data
from models import ImprovedGCN, GAT, GraphSAGE
from trainer import AdvancedTrainer
from evaluation import detailed_evaluation, explain_predictions
from network_analysis import analyze_network_properties
from visualization import create_interactive_plots, create_enhanced_animations
from optimization import optimize_hyperparameters

warnings.filterwarnings('ignore')

def main():
    print("=== Enhanced Graph Neural Network Analysis ===\n")
    
    # Load and prepare data
    dataset, data, num_countries = load_and_prepare_data()

    # Visualize initial graph
    G = to_networkx(data, to_undirected=True)
    plt.figure(figsize=(12, 10))
    plt.axis('off')
    nx.draw_networkx(G,
                    pos=nx.spring_layout(G, seed=0),
                    with_labels=True,
                    node_size=800,
                    node_color=data.y.numpy(),
                    cmap="hsv",
                    vmin=-2, vmax=3,
                    width=0.8,
                    edge_color="grey",
                    font_size=14)
    plt.title("Initial Karate Club Graph", fontsize=16)
    plt.savefig('initial_graph_enhanced.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Hyperparameter optimization (optional - uncomment to run)
    # print("Optimizing hyperparameters...")
    # best_params = optimize_hyperparameters(data, dataset, num_countries, n_trials=10)

    # Initialize models
    models = {
        'GCN': ImprovedGCN(dataset.num_features, 64, num_countries, 0.5),
        'GAT': GAT(dataset.num_features, num_countries, 0.6),
        'GraphSAGE': GraphSAGE(dataset.num_features, num_countries)
    }

    # Training results storage
    all_results = {}
    final_embeddings = {}

    # Train each model
    for model_name, model in models.items():
        print(f"\n=== Training {model_name} ===")
        trainer = AdvancedTrainer(model, data, num_countries, model_name)
        train_losses, val_losses, train_accs, val_accs, embeddings, outputs = trainer.train_with_validation(epochs=100)

        # Store results
        all_results[model_name] = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs,
            'embeddings': embeddings,
            'outputs': outputs
        }

        # Detailed evaluation
        report, cm, final_embed = detailed_evaluation(model, data, model_name)
        final_embeddings[model_name] = final_embed.numpy()

        # Feature importance for first node
        if model_name == 'GCN':  # Only for GCN to avoid repetition
            feature_imp = explain_predictions(model, data, node_idx=0)

        # Create enhanced animations
        create_enhanced_animations(embeddings[::10], outputs[::10],
                                 train_losses[::10], train_accs[::10], model_name, data)

    # Network analysis
    print("\n=== Network Analysis ===")
    metrics, centrality_df = analyze_network_properties(G, final_embeddings['GCN'], data.y.numpy())
    centrality_df.to_csv('network_centrality_analysis.csv', index=True)

    # Create interactive comparison dashboard
    print("\n=== Creating Interactive Dashboard ===")
    losses_dict = {name: results['train_losses'] for name, results in all_results.items()}
    accs_dict = {name: results['train_accs'] for name, results in all_results.items()}
    create_interactive_plots(final_embeddings, losses_dict, accs_dict, data)

    # Model comparison table
    print("\n=== Model Comparison ===")
    comparison_data = []
    for model_name, results in all_results.items():
        final_train_acc = results['train_accs'][-1]
        final_val_acc = results['val_accs'][-1]
        final_train_loss = results['train_losses'][-1]
        final_val_loss = results['val_losses'][-1]
        comparison_data.append({
            'Model': model_name,
            'Final Train Acc': f"{final_train_acc*100:.2f}%",
            'Final Val Acc': f"{final_val_acc*100:.2f}%",
            'Final Train Loss': f"{final_train_loss:.4f}",
            'Final Val Loss': f"{final_val_loss:.4f}"
        })

    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    comparison_df.to_csv('model_comparison.csv', index=False)

    print("\n=== Analysis Complete ===")
    print("Generated files:")
    print("- initial_graph_enhanced.png")
    print("- *_confusion_matrix.png (for each model)")
    print("- *_enhanced.mp4/.gif (animations for each model)")
    print("- interactive_results.html (interactive dashboard)")
    print("- network_centrality_analysis.csv")
    print("- model_comparison.csv")
    print("- feature_importance_node_0.png")
    print("- TensorBoard logs in 'runs/' directory")
    print("\nRun 'tensorboard --logdir=runs' to view training logs!")

if __name__ == "__main__":
    main()

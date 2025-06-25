import torch
import numpy as np
import pandas as pd
from torch_geometric.datasets import KarateClub

def load_and_prepare_data():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Load the Karate Club dataset
    dataset = KarateClub()
    data = dataset[0]

    print(f"Dataset features: {dataset.num_features}")

    # Convert node features to DataFrame
    node_features_df = pd.DataFrame(data.x.numpy(), columns=[f'feature_{i}' for i in range(data.x.shape[1])])
    node_features_df.index.name = 'node'
    print("\nNode features:")
    print(node_features_df.head())

    # Convert edge indices to DataFrame
    edge_index_df = pd.DataFrame(data.edge_index.numpy().T, columns=['source', 'target'])
    print(f"\nEdge indices (count: {edge_index_df.shape[0]}):")
    print(edge_index_df.head())

    # Print dataset statistics
    print(f"\nNumber of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.edge_index.shape[1]}")

    # Assign random country labels
    num_countries = 4
    np.random.seed(42)
    data.y = torch.tensor(np.random.choice(num_countries, data.num_nodes))

    return dataset, data, num_countries

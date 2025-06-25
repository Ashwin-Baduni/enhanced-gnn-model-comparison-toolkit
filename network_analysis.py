import networkx as nx
import pandas as pd

def analyze_network_properties(G, embeddings, labels):
    # Calculate network metrics
    metrics = {
        'clustering_coefficient': nx.average_clustering(G),
        'average_path_length': nx.average_shortest_path_length(G),
        'density': nx.density(G),
        'assortativity': nx.degree_assortativity_coefficient(G)
    }

    print("\nNetwork Properties:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    # Node centrality measures
    centrality_measures = {
        'degree': nx.degree_centrality(G),
        'betweenness': nx.betweenness_centrality(G),
        'closeness': nx.closeness_centrality(G),
        'eigenvector': nx.eigenvector_centrality(G)
    }

    # Create centrality DataFrame
    centrality_df = pd.DataFrame(centrality_measures)
    centrality_df['predicted_label'] = labels
    if embeddings.shape[1] >= 3:
        centrality_df['embedding_x'] = embeddings[:, 0]
        centrality_df['embedding_y'] = embeddings[:, 1]
        centrality_df['embedding_z'] = embeddings[:, 2]

    return metrics, centrality_df

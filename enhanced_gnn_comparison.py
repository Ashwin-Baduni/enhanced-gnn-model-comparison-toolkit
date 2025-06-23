import torch
from torch_geometric.datasets import KarateClub
from torch_geometric.utils import to_networkx
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import Linear
import pandas as pd
import matplotlib.animation as animation
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import optuna
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings('ignore')

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

# Enhanced Model Architectures
class ImprovedGCN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim=64, num_classes=4, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, 32)
        self.classifier = Linear(32, num_classes)
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, x, edge_index):
        h1 = self.conv1(x, edge_index).relu()
        h1 = self.dropout(h1)
        h2 = self.conv2(h1, edge_index).relu()
        h2 = self.dropout(h2)
        h3 = self.conv3(h2, edge_index).relu()
        z = self.classifier(h3)
        return h3, z

class GAT(torch.nn.Module):
    def __init__(self, num_features, num_classes=4, dropout=0.6):
        super().__init__()
        self.conv1 = GATConv(num_features, 8, heads=8, dropout=dropout)
        self.conv2 = GATConv(8*8, num_classes, heads=1, dropout=dropout)
        
    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index).relu()
        z = self.conv2(h, edge_index)
        return h, z

class GraphSAGE(torch.nn.Module):
    def __init__(self, num_features, num_classes=4):
        super().__init__()
        self.conv1 = SAGEConv(num_features, 64)
        self.conv2 = SAGEConv(64, 32)
        self.classifier = Linear(32, num_classes)
        
    def forward(self, x, edge_index):
        h1 = self.conv1(x, edge_index).relu()
        h2 = self.conv2(h1, edge_index).relu()
        z = self.classifier(h2)
        return h2, z

# Advanced Trainer Class
class AdvancedTrainer:
    def __init__(self, model, data, num_classes, model_name="GCN"):
        self.model = model
        self.data = data
        self.model_name = model_name
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        self.scheduler = ReduceLROnPlateau(self.optimizer, patience=20, factor=0.5)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.writer = SummaryWriter(f'runs/{model_name}')
        
        # Create train/val split
        num_nodes = self.data.num_nodes
        self.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        self.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        self.train_mask[:int(0.8 * num_nodes)] = True
        self.val_mask[int(0.8 * num_nodes):] = True
        
    def accuracy(self, pred_y, y):
        return (pred_y == y).sum() / len(y)
    
    def train_with_validation(self, epochs=200, patience=50):
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        embeddings, outputs = [], []
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            self.optimizer.zero_grad()
            h, z = self.model(self.data.x, self.data.edge_index)
            train_loss = self.criterion(z[self.train_mask], self.data.y[self.train_mask])
            train_acc = self.accuracy(z[self.train_mask].argmax(dim=1), self.data.y[self.train_mask])
            train_loss.backward()
            self.optimizer.step()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                h, z = self.model(self.data.x, self.data.edge_index)
                val_loss = self.criterion(z[self.val_mask], self.data.y[self.val_mask])
                val_acc = self.accuracy(z[self.val_mask].argmax(dim=1), self.data.y[self.val_mask])
                
            train_losses.append(train_loss.item())
            val_losses.append(val_loss.item())
            train_accs.append(train_acc.item())
            val_accs.append(val_acc.item())
            embeddings.append(h.detach())
            outputs.append(z.argmax(dim=1).detach())
            
            # Log to tensorboard
            self.writer.add_scalar('Loss/Train', train_loss.item(), epoch)
            self.writer.add_scalar('Loss/Val', val_loss.item(), epoch)
            self.writer.add_scalar('Accuracy/Train', train_acc.item(), epoch)
            self.writer.add_scalar('Accuracy/Val', val_acc.item(), epoch)
            
            self.scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.patience_counter = 0
                torch.save(self.model.state_dict(), f'best_{self.model_name.lower()}_model.pth')
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
                    
            if epoch % 10 == 0:
                print(f'{self.model_name} Epoch {epoch:>3} | Train Loss: {train_loss.item():.3f} | Val Loss: {val_loss.item():.3f} | Train Acc: {train_acc.item()*100:.1f}% | Val Acc: {val_acc.item()*100:.1f}%')
        
        self.writer.close()
        return train_losses, val_losses, train_accs, val_accs, embeddings, outputs

# Evaluation Functions
def detailed_evaluation(model, data, model_name="Model"):
    model.eval()
    with torch.no_grad():
        h, z = model(data.x, data.edge_index)
        pred = z.argmax(dim=1)
        
    # Classification report
    report = classification_report(data.y.numpy(), pred.numpy(), 
                                 target_names=[f'Country_{i}' for i in range(4)])
    print(f"\n{model_name} Classification Report:")
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(data.y.numpy(), pred.numpy())
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{model_name.lower()}_confusion_matrix.png')
    plt.close()
    
    return report, cm, h

# Network Analysis
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

# Interactive Visualization
def create_interactive_plots(embeddings_dict, losses_dict, accuracies_dict):
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}],
               [{"type": "scatter"}, {"type": "scatter"}]],
        subplot_titles=("GCN 3D Embeddings", "GAT 3D Embeddings", "Training Losses", "Training Accuracies")
    )
    
    colors = ['red', 'blue', 'green', 'orange']
    
    # 3D scatter plots for different models
    for i, (model_name, embeddings) in enumerate(embeddings_dict.items()):
        if embeddings.shape[1] >= 3:
            row, col = (1, 1) if i == 0 else (1, 2)
            fig.add_trace(
                go.Scatter3d(
                    x=embeddings[:, 0], y=embeddings[:, 1], z=embeddings[:, 2],
                    mode='markers',
                    marker=dict(size=8, color=data.y.numpy(), colorscale='Viridis'),
                    text=[f'Node {j}' for j in range(len(embeddings))],
                    name=model_name,
                    hovertemplate='Node: %{text}<br>X: %{x}<br>Y: %{y}<br>Z: %{z}<extra></extra>'
                ),
                row=row, col=col
            )
    
    # Loss and accuracy plots
    for i, (model_name, losses) in enumerate(losses_dict.items()):
        fig.add_trace(
            go.Scatter(x=list(range(len(losses))), y=losses, name=f'{model_name} Loss', 
                      line=dict(color=colors[i])),
            row=2, col=1
        )
    
    for i, (model_name, accs) in enumerate(accuracies_dict.items()):
        fig.add_trace(
            go.Scatter(x=list(range(len(accs))), y=accs, name=f'{model_name} Accuracy',
                      line=dict(color=colors[i], dash='dash')),
            row=2, col=2
        )
    
    fig.update_layout(height=800, title="Multi-Model Comparison Dashboard")
    fig.write_html('interactive_results.html')
    print("Interactive dashboard saved as 'interactive_results.html'")

# Model Interpretability
def explain_predictions(model, data, node_idx=0):
    """Simple gradient-based explanation"""
    model.eval()
    data.x.requires_grad_(True)
    
    h, z = model(data.x, data.edge_index)
    target_logit = z[node_idx, data.y[node_idx]]
    target_logit.backward()
    
    # Feature importance
    feature_importance = data.x.grad[node_idx].abs()
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(feature_importance)), feature_importance.detach().numpy())
    plt.title(f'Feature Importance for Node {node_idx}')
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.savefig(f'feature_importance_node_{node_idx}.png')
    plt.close()
    
    return feature_importance

# Hyperparameter Optimization
def optimize_hyperparameters(data, n_trials=20):
    def objective(trial):
        # Suggest hyperparameters
        lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
        hidden_dim = trial.suggest_int('hidden_dim', 16, 128)
        dropout = trial.suggest_float('dropout', 0.1, 0.8)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
        
        # Create and train model
        model = ImprovedGCN(dataset.num_features, hidden_dim, num_countries, dropout)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Quick training loop
        best_acc = 0
        for epoch in range(50):
            model.train()
            optimizer.zero_grad()
            h, z = model(data.x, data.edge_index)
            loss = torch.nn.functional.cross_entropy(z, data.y)
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    _, z = model(data.x, data.edge_index)
                    acc = (z.argmax(dim=1) == data.y).sum() / len(data.y)
                    best_acc = max(best_acc, acc.item())
        
        return best_acc
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    print(f"\nBest hyperparameters: {study.best_params}")
    print(f"Best accuracy: {study.best_value:.4f}")
    return study.best_params

# Enhanced Animation Function
def create_enhanced_animations(embeddings_list, outputs_list, losses, accuracies, model_name="GCN"):
    # 2D Animation
    G = to_networkx(data, to_undirected=True)
    pos = nx.spring_layout(G, seed=0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    def animate_enhanced(i):
        ax1.clear()
        ax2.clear()
        
        # Graph visualization
        nx.draw_networkx(G, pos=pos, ax=ax1,
                        with_labels=True,
                        node_size=800,
                        node_color=outputs_list[i].numpy(),
                        cmap="hsv",
                        vmin=-2, vmax=3,
                        width=0.8,
                        edge_color="grey",
                        font_size=12)
        ax1.set_title(f'{model_name} - Epoch {i*10} | Loss: {losses[i]:.3f} | Acc: {accuracies[i]*100:.1f}%', 
                     fontsize=16)
        ax1.axis('off')
        
        # Metrics plot
        epochs_so_far = list(range(0, (i+1)*10, 10))
        ax2.plot(epochs_so_far, losses[:i+1], 'r-', label='Loss', linewidth=2)
        ax2_twin = ax2.twinx()
        ax2_twin.plot(epochs_so_far, accuracies[:i+1], 'b-', label='Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss', color='r')
        ax2_twin.set_ylabel('Accuracy', color='b')
        ax2.set_title('Training Progress')
        ax2.grid(True, alpha=0.3)
    
    try:
        anim = animation.FuncAnimation(fig, animate_enhanced, 
                                     frames=min(len(embeddings_list), 20), 
                                     interval=800)
        anim.save(f'{model_name.lower()}_enhanced.mp4', writer='ffmpeg')
        print(f"Enhanced animation saved as '{model_name.lower()}_enhanced.mp4'")
    except Exception as e:
        print(f"FFmpeg not available: {e}. Saving as GIF instead.")
        anim.save(f'{model_name.lower()}_enhanced.gif', writer='pillow')
    
    plt.close()

# Main Execution
if __name__ == "__main__":
    print("=== Enhanced Graph Neural Network Analysis ===\n")
    
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
    # best_params = optimize_hyperparameters(data, n_trials=10)
    
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
                                 train_losses[::10], train_accs[::10], model_name)
    
    # Network analysis
    print("\n=== Network Analysis ===")
    metrics, centrality_df = analyze_network_properties(G, final_embeddings['GCN'], data.y.numpy())
    centrality_df.to_csv('network_centrality_analysis.csv', index=True)
    
    # Create interactive comparison dashboard
    print("\n=== Creating Interactive Dashboard ===")
    losses_dict = {name: results['train_losses'] for name, results in all_results.items()}
    accs_dict = {name: results['train_accs'] for name, results in all_results.items()}
    create_interactive_plots(final_embeddings, losses_dict, accs_dict)
    
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

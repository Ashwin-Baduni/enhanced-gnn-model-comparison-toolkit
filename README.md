# Graph Neural Networks on Karate Club Dataset: A Comprehensive Comparison

A comprehensive implementation and comparison of Graph Neural Network (GNN) architectures including GCN, GAT, and GraphSAGE on the classic Karate Club social network dataset. This project features advanced training techniques, interactive visualizations, network analysis, and model interpretability tools.

## 🎯 Features

- **Multiple GNN Architectures**: Implementation of GCN, GAT, and GraphSAGE models
- **Advanced Training**: Early stopping, learning rate scheduling, and validation splits
- **Interactive Visualizations**: 3D embeddings, training progress, and comparison dashboards
- **Network Analysis**: Centrality measures, clustering coefficients, and graph properties
- **Model Interpretability**: Feature importance analysis and gradient-based explanations
- **Hyperparameter Optimization**: Optuna-based automated hyperparameter tuning
- **TensorBoard Integration**: Real-time training monitoring and logging
- **Animated Training Progress**: MP4/GIF animations showing model learning evolution

## 📊 Dataset

The project uses the **Karate Club dataset** - a famous social network representing friendships in a university karate club:
- **34 nodes** (club members)
- **78 edges** (friendship connections)
- **Synthetic labels**: 4 randomly assigned country categories for node classification

## 🏗️ Model Architectures

### 1. Improved Graph Convolutional Network (GCN)
- 3-layer architecture with dropout regularization
- Hidden dimensions: 64 → 64 → 32 → 4 classes
- ReLU activations and dropout for regularization

### 2. Graph Attention Network (GAT)
- Multi-head attention mechanism (8 heads in first layer)
- Attention-based neighbor aggregation
- Dropout for attention weights

### 3. GraphSAGE
- Sampling and aggregating approach
- 2-layer architecture with mean aggregation
- Scalable to larger graphs

## 🚀 Installation

```
# Install required packages
pip install -r requirements.txt
```

### Requirements
```
torch>=1.9.0
torch-geometric>=2.0.0
networkx>=2.6
matplotlib>=3.4.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
seaborn>=0.11.0
plotly>=5.0.0
optuna>=2.10.0
tensorboard>=2.7.0
```

## 📖 Usage

### Basic Execution
```
python gnn_karate_club_analysis.py
```

### With Hyperparameter Optimization
Uncomment the hyperparameter optimization section in the main execution block:
```
# Uncomment these lines in the main execution
print("Optimizing hyperparameters...")
best_params = optimize_hyperparameters(data, n_trials=10)
```

### TensorBoard Monitoring
```
tensorboard --logdir=runs
```

## 📁 Output Files

The script generates comprehensive analysis outputs:

### Visualizations
- `initial_graph_enhanced.png` - Initial graph visualization
- `*_confusion_matrix.png` - Confusion matrices for each model
- `*_enhanced.mp4/.gif` - Training progress animations
- `interactive_results.html` - Interactive 3D dashboard
- `feature_importance_node_0.png` - Feature importance analysis

### Data Files
- `network_centrality_analysis.csv` - Network centrality measures
- `model_comparison.csv` - Final model performance comparison
- `best_*_model.pth` - Saved model checkpoints

### Logs
- `runs/` directory - TensorBoard training logs

## 🔍 Key Components

### Advanced Trainer Class
- **Early Stopping**: Prevents overfitting with patience-based stopping
- **Learning Rate Scheduling**: ReduceLROnPlateau for adaptive learning rates
- **Train/Validation Split**: 80/20 split for proper evaluation
- **TensorBoard Logging**: Real-time training metrics

### Network Analysis
- **Centrality Measures**: Degree, betweenness, closeness, eigenvector centrality
- **Graph Properties**: Clustering coefficient, average path length, density
- **Embedding Analysis**: 3D visualization of learned node representations

### Model Interpretability
- **Feature Importance**: Gradient-based feature attribution
- **Attention Visualization**: GAT attention weight analysis
- **Embedding Evolution**: Track how representations change during training

## 📈 Results

The project provides comprehensive model comparison across multiple metrics:

| Model | Architecture | Key Features |
|-------|-------------|--------------|
| **GCN** | 3-layer convolution | Fast, simple, effective for homophilic graphs |
| **GAT** | Multi-head attention | Learns importance of different neighbors |
| **GraphSAGE** | Sampling-based | Scalable to large graphs, inductive learning |

### Performance Metrics
- Training and validation accuracy/loss curves
- Classification reports with precision, recall, F1-score
- Confusion matrices for detailed error analysis
- Network-level performance correlations

## 🎨 Visualizations

### Interactive Dashboard
- **3D Embeddings**: Explore learned node representations in 3D space
- **Training Progress**: Real-time loss and accuracy curves
- **Model Comparison**: Side-by-side performance analysis

### Animated Training
- **Node Classification Evolution**: Watch predictions improve over epochs
- **Embedding Dynamics**: See how node representations evolve
- **Metrics Tracking**: Real-time training progress visualization

## 🔧 Customization

### Adding New Models
```
class YourGNN(torch.nn.Module):
    def __init__(self, num_features, num_classes=4):
        super().__init__()
        # Your architecture here
        
    def forward(self, x, edge_index):
        # Your forward pass
        return embeddings, logits

# Add to models dictionary
models['YourGNN'] = YourGNN(dataset.num_features, num_countries)
```

### Hyperparameter Tuning
Modify the `optimize_hyperparameters` function to tune different parameters:
```
def objective(trial):
    lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
    hidden_dim = trial.suggest_int('hidden_dim', 16, 128)
    # Add more parameters as needed
```

## 📚 Educational Value

This project serves as an excellent learning resource for:
- **Graph Neural Network fundamentals**
- **PyTorch Geometric framework**
- **Advanced training techniques**
- **Model comparison methodologies**
- **Interactive visualization creation**
- **Scientific computing best practices**

## 🙏 Acknowledgments

- **PyTorch Geometric** team for the excellent graph learning framework
- **Karate Club dataset** from Zachary (1977) for the classic social network data
- **NetworkX** for graph analysis utilities
- **Plotly** for interactive visualizations

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from torch_geometric.utils import to_networkx

def create_interactive_plots(embeddings_dict, losses_dict, accuracies_dict, data):
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
                    hovertemplate='Node: %{text}<br>X: %{x}<br>Y: %{y}<br>Z: %{z}'
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

def create_enhanced_animations(embeddings_list, outputs_list, losses, accuracies, model_name, data):
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

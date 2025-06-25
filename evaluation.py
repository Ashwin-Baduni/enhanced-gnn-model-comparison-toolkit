import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

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

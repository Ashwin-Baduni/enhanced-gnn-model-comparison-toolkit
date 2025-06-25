import torch
import optuna
from models import ImprovedGCN

def optimize_hyperparameters(data, dataset, num_countries, n_trials=20):
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

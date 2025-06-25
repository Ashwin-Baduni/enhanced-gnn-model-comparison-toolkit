import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

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

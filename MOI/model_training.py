import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
import numpy as np
from scipy import sparse
import os
import joblib



config = {
    'data_dir': './preprocessed',
    'batch_size':16,
    'shuffle_train': True,

    'hidden_dim': 128,
    'dropout_rate': 0.15,

    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'lr': 0.003,
    'epochs': 100,
    'patience': 30,
    'scheduler_step': 100,
    'scheduler_gamma': 0.5,

    'model_save_path': './models/best_model_o.pth'
}


# ============== Atten Layer & Model ==============
class AttentionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, return_sequence=False):
        super().__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.return_sequence = return_sequence
        self.scale = np.sqrt(hidden_dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attended = torch.matmul(attn_weights, V)
        return attended if self.return_sequence else attended.sum(dim=1)


class AttentionMOI(nn.Module):
    def __init__(self, dna_features, rna_features, num_classes, hidden_dim=128, dropout_rate=0.3):
        super().__init__()
        self.dna_attention = AttentionLayer(dna_features, hidden_dim)
        self.dna_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )

        self.rna_attention = AttentionLayer(rna_features, hidden_dim)
        self.rna_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )

        self.cross_attention = AttentionLayer(hidden_dim // 2, hidden_dim // 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate + 0.2),
            nn.Linear(hidden_dim // 4, num_classes)
        )

    def forward(self, dna_data, rna_data):
        dna_vec = self.dna_attention(dna_data)
        dna_rd = self.dna_mlp(dna_vec)
        rna_vec = self.rna_attention(rna_data)
        rna_rd = self.rna_mlp(rna_vec)

        seq = torch.stack([dna_rd, rna_rd], dim=1)
        fused = self.cross_attention(seq)
        return self.classifier(fused)


# ============== 数据集 ==============
class MultiOmicsDataset(Dataset):
    def __init__(self, dna_data, rna_data, labels):
        self.dna_data = torch.FloatTensor(dna_data)
        self.rna_data = torch.FloatTensor(rna_data)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.dna_data[idx], self.rna_data[idx], self.labels[idx]


# ============== 训练器 ==============
class AttentionMOITrainer:
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        self.best_model_state = None

    def train_epoch(self, dataloader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        for dna_data, rna_data, labels in dataloader:
            dna_data, rna_data, labels = dna_data.to(self.device), rna_data.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            outputs = self.model(dna_data, rna_data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)

    def evaluate(self, dataloader, criterion):
        self.model.eval()
        total_loss = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for dna_data, rna_data, labels in dataloader:
                dna_data, rna_data, labels = dna_data.to(self.device), rna_data.to(self.device), labels.to(self.device)
                outputs = self.model(dna_data, rna_data)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_preds)
        return avg_loss, accuracy, all_preds, all_labels

    def train(self, train_loader, val_loader, epochs, lr, patience, scheduler_step, scheduler_gamma, model_save_path):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
        best_val_acc, patience_counter = 0.0, 0

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            val_loss, val_acc, _, _ = self.evaluate(val_loader, criterion)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                }, model_save_path)
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
            scheduler.step()

            if (epoch + 1) % 50 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}] Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}')

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        print(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")
        return best_val_acc


# ============== 主函数 ==============
def main(cfg):
    print("=" * 50)
    print("Model Training")
    print("=" * 50)
    device = cfg['device']
    print(f"Using device: {device}")

    if not os.path.exists(cfg['data_dir']):
        print(f"Error: Preprocessed data directory {cfg['data_dir']} does not exist")
        return

    # 数据加载
    print("Loading preprocessed data...")
    X_dna_train = sparse.load_npz(os.path.join(cfg['data_dir'], 'dna_train.npz')).toarray()
    X_dna_val = sparse.load_npz(os.path.join(cfg['data_dir'], 'dna_val.npz')).toarray()
    X_rna_train = sparse.load_npz(os.path.join(cfg['data_dir'], 'rna_train.npz')).toarray()
    X_rna_val = sparse.load_npz(os.path.join(cfg['data_dir'], 'rna_val.npz')).toarray()
    y_train = np.load(os.path.join(cfg['data_dir'], 'y_train.npy'))
    y_val = np.load(os.path.join(cfg['data_dir'], 'y_val.npy'))
    label_encoder = joblib.load(os.path.join(cfg['data_dir'], 'label_encoder.pkl'))

    train_dataset = MultiOmicsDataset(X_dna_train, X_rna_train, y_train)
    val_dataset = MultiOmicsDataset(X_dna_val, X_rna_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=cfg['shuffle_train'])
    val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], shuffle=False)

    num_classes = len(np.unique(y_train))
    model = AttentionMOI(
        dna_features=X_dna_train.shape[1],
        rna_features=X_rna_train.shape[1],
        num_classes=num_classes,
        hidden_dim=cfg['hidden_dim'],
        dropout_rate=cfg['dropout_rate']
    )

    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Number of classes: {num_classes}, Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    trainer = AttentionMOITrainer(model, device)
    best_val_acc = trainer.train(
        train_loader, val_loader,
        epochs=cfg['epochs'], lr=cfg['lr'], patience=cfg['patience'],
        scheduler_step=cfg['scheduler_step'], scheduler_gamma=cfg['scheduler_gamma'],
        model_save_path=cfg['model_save_path']
    )

    print(f"Training completed. Best model saved to: {cfg['model_save_path']}")


if __name__ == "__main__":
    main(config)
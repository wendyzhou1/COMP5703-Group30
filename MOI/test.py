import os
import logging
from datetime import datetime
import numpy as np
from scipy import sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, \
    recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler

# =========================
# Config (Same as training)
# =========================
config = {
    'data_dir': './preprocessed',
    'batch_size': 16,
    'hidden_dim': 128,
    'dropout_rate': 0.2,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'activation': 'gelu',
    'class_names': ['Classical', 'Mesenchymal', 'Neural', 'Proneural'],
    'model_save_path': './best_model/best_model.pth',
}


# =========================
# Logging
# =========================
def setup_test_logging():
    os.makedirs("./results", exist_ok=True)
    log_path = f"./results/test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__), log_path


# =========================
# Dataset (Same as training)
# =========================
class MultiOmicsDataset(Dataset):
    def __init__(self, dna: np.ndarray, rna: np.ndarray, labels: np.ndarray):
        assert dna.shape[0] == rna.shape[0] == labels.shape[0]
        self.dna = torch.tensor(dna, dtype=torch.float32)
        self.rna = torch.tensor(rna, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return self.dna[idx], self.rna[idx], self.labels[idx]


# =========================
# Model (Same as training)
# =========================
class AttentionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.scale = float(np.sqrt(hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)
        out = out.sum(dim=1)
        return self.norm(out)


class AttentionMOI(nn.Module):
    def __init__(self, dna_dim, rna_dim, num_classes, hidden_dim=128, dropout_rate=0.5, activation='gelu'):
        super().__init__()
        act = nn.GELU() if activation == 'gelu' else (nn.ReLU() if activation == 'relu' else nn.Tanh())

        # DNA branch
        self.dna_attn = AttentionLayer(dna_dim, hidden_dim)
        self.dna_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            act,
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            act,
            nn.Dropout(dropout_rate),
        )

        # RNA branch
        self.rna_attn = AttentionLayer(rna_dim, hidden_dim)
        self.rna_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            act,
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            act,
            nn.Dropout(dropout_rate),
        )

        # Fusion
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            act,
            nn.Dropout(min(0.9, dropout_rate + 0.1)),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            act,
            nn.Dropout(min(0.9, dropout_rate + 0.1)),
        )

        self.fuse_norm = nn.LayerNorm(hidden_dim // 4)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 4, num_classes)
        )

    def forward(self, dna_x, rna_x):
        dna_vec = self.dna_mlp(self.dna_attn(dna_x))
        rna_vec = self.rna_mlp(self.rna_attn(rna_x))
        fused = torch.stack([dna_vec, rna_vec], dim=1).sum(dim=1)
        fused = self.fuse_norm(self.fuse(fused))
        logits = self.classifier(fused)
        return logits


# =========================
# Data Loading
# =========================
def load_split(data_dir, split: str):
    """Load data split"""
    try:
        X_dna = sparse.load_npz(os.path.join(data_dir, f'dna_{split}.npz')).toarray()
        X_rna = sparse.load_npz(os.path.join(data_dir, f'rna_{split}.npz')).toarray()
        y = np.load(os.path.join(data_dir, f'y_{split}.npy'))
        return X_dna, X_rna, y
    except Exception as e:
        logging.error(f"Failed to load {split} data: {e}")
        return None, None, None


# =========================
# Model Loading
# =========================
def load_trained_model(model_path, dna_dim, rna_dim, num_classes, config):
    """Load trained model"""
    try:
        model = AttentionMOI(
            dna_dim=dna_dim,
            rna_dim=rna_dim,
            num_classes=num_classes,
            hidden_dim=config['hidden_dim'],
            dropout_rate=config['dropout_rate'],
            activation=config['activation']
        )

        # Load model weights
        checkpoint = torch.load(model_path, map_location=config['device'])
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.to(config['device'])
        model.eval()

        logging.info(f"Model loaded successfully: {model_path}")
        return model

    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return None


# =========================
# Evaluation Functions
# =========================
def evaluate_model_with_loss(model, test_loader, config, criterion):
    """Evaluate model performance with loss calculation"""
    device = config['device']
    class_names = config['class_names']

    all_predictions = []
    all_targets = []
    all_probabilities = []
    all_dna_features = []
    all_rna_features = []
    total_loss = 0.0

    model.eval()
    with torch.no_grad():
        for dna, rna, targets in test_loader:
            dna, rna, targets = dna.to(device), rna.to(device), targets.to(device)
            outputs = model(dna, rna)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_dna_features.extend(dna.cpu().numpy())
            all_rna_features.extend(rna.cpu().numpy())

    avg_loss = total_loss / len(test_loader)

    return (np.array(all_predictions), np.array(all_targets),
            np.array(all_probabilities), np.array(all_dna_features),
            np.array(all_rna_features), avg_loss)


def calculate_comprehensive_metrics(y_true, y_pred, y_prob, class_names):
    """Calculate comprehensive evaluation metrics"""
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    macro_precision = precision_score(y_true, y_pred, average='macro')
    macro_recall = recall_score(y_true, y_pred, average='macro')

    # Per-class metrics
    class_f1 = f1_score(y_true, y_pred, average=None)
    class_precision = precision_score(y_true, y_pred, average=None)
    class_recall = recall_score(y_true, y_pred, average=None)

    # Detailed classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    metrics = {
        'overall': {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
        },
        'per_class': {}
    }

    for i, class_name in enumerate(class_names):
        metrics['per_class'][class_name] = {
            'f1_score': class_f1[i],
            'precision': class_precision[i],
            'recall': class_recall[i],
            'support': report[class_name]['support']
        }

    return metrics, report


def generate_detailed_report(y_true, y_pred, class_names, test_loss):
    """Generate detailed classification report"""
    metrics, report = calculate_comprehensive_metrics(y_true, y_pred, None, class_names)

    print("\n" + "=" * 80)
    print("📊 DETAILED CLASSIFICATION REPORT")
    print("=" * 80)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Overall Accuracy: {metrics['overall']['accuracy']:.4f}")
    print(f"Macro F1-Score: {metrics['overall']['macro_f1']:.4f}")
    print(f"Weighted F1-Score: {metrics['overall']['weighted_f1']:.4f}")
    print(f"Macro Precision: {metrics['overall']['macro_precision']:.4f}")
    print(f"Macro Recall: {metrics['overall']['macro_recall']:.4f}")

    print("\n📈 PER-CLASS METRICS:")
    print("-" * 60)
    print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print("-" * 60)

    for class_name in class_names:
        class_metrics = metrics['per_class'][class_name]
        print(f"{class_name:<15} {class_metrics['precision']:<10.4f} {class_metrics['recall']:<10.4f} "
              f"{class_metrics['f1_score']:<10.4f} {class_metrics['support']:<10}")

    return metrics, report


# =========================
# Visualization Functions
# =========================
def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})

    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    return cm


def plot_class_performance(y_true, y_pred, class_names, save_path):
    """Plot class performance visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # True label distribution
    true_counts = [np.sum(y_true == i) for i in range(len(class_names))]
    ax1.bar(class_names, true_counts, color='skyblue', alpha=0.7)
    ax1.set_title('True Label Distribution', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Sample Count')
    ax1.tick_params(axis='x', rotation=45)

    # Add count annotations
    for i, count in enumerate(true_counts):
        ax1.text(i, count + 0.1, str(count), ha='center', va='bottom')

    # Class accuracy
    accuracies = []
    for i in range(len(class_names)):
        mask = y_true == i
        if np.sum(mask) > 0:
            accuracy = np.sum(y_pred[mask] == i) / np.sum(mask)
        else:
            accuracy = 0
        accuracies.append(accuracy)

    bars = ax2.bar(class_names, accuracies, color='lightcoral', alpha=0.7)
    ax2.set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 1.1)
    ax2.tick_params(axis='x', rotation=45)

    # Add accuracy annotations
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                 f'{acc:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    return true_counts, accuracies


def plot_metrics_comparison(metrics, class_names, save_path):
    """Plot comparison of different metrics across classes"""
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(class_names))
    width = 0.25

    # Extract metrics
    precisions = [metrics['per_class'][name]['precision'] for name in class_names]
    recalls = [metrics['per_class'][name]['recall'] for name in class_names]
    f1_scores = [metrics['per_class'][name]['f1_score'] for name in class_names]

    rects1 = ax.bar(x - width, precisions, width, label='Precision', alpha=0.7)
    rects2 = ax.bar(x, recalls, width, label='Recall', alpha=0.7)
    rects3 = ax.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.7)

    ax.set_xlabel('Classes')
    ax.set_ylabel('Scores')
    ax.set_title('Per-Class Performance Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45)
    ax.legend()
    ax.set_ylim(0, 1.1)

    # Add value labels on bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# =========================
# Result Saving
# =========================
def save_detailed_results(y_true, y_pred, probabilities, dna_features, rna_features,
                          class_names, metrics, test_loss, save_path):
    """Save detailed results to CSV file"""
    results_df = pd.DataFrame({
        'true_label': y_true,
        'predicted_label': y_pred,
        'true_label_name': [class_names[i] for i in y_true],
        'predicted_label_name': [class_names[i] for i in y_pred],
        'is_correct': y_true == y_pred,
        'confidence': np.max(probabilities, axis=1),
        'predicted_class_prob': [probabilities[i, y_pred[i]] for i in range(len(y_pred))]
    })

    # Add prediction probabilities for each class
    for i, class_name in enumerate(class_names):
        results_df[f'prob_{class_name}'] = probabilities[:, i]

    # Add a few DNA and RNA features as examples
    n_dna_features = min(5, dna_features.shape[1])
    n_rna_features = min(5, rna_features.shape[1])

    for i in range(n_dna_features):
        results_df[f'dna_feature_{i}'] = dna_features[:, i]

    for i in range(n_rna_features):
        results_df[f'rna_feature_{i}'] = rna_features[:, i]

    # Save to CSV
    results_df.to_csv(save_path, index=False)
    logging.info(f"Detailed results saved to: {save_path}")

    return results_df


def save_comprehensive_summary(y_true, y_pred, probabilities, class_names, metrics, test_loss, cm, save_path):
    """Save comprehensive summary statistics"""
    summary = {
        'test_loss': test_loss,
        'overall_metrics': metrics['overall'],
        'per_class_metrics': metrics['per_class'],
        'confusion_matrix': cm.tolist(),
        'class_distribution': {},
        'misclassification_analysis': {}
    }

    # Class distribution
    for i, class_name in enumerate(class_names):
        class_mask = y_true == i
        class_count = np.sum(class_mask)
        class_correct = np.sum(y_pred[class_mask] == i)
        class_accuracy = class_correct / class_count if class_count > 0 else 0

        summary['class_distribution'][class_name] = {
            'sample_count': int(class_count),
            'correct_predictions': int(class_correct),
            'accuracy': float(class_accuracy)
        }

        # Misclassification analysis
        wrong_mask = class_mask & (y_pred != i)
        wrong_predictions = y_pred[wrong_mask]
        wrong_distribution = {}
        if len(wrong_predictions) > 0:
            wrong_counts = np.bincount(wrong_predictions, minlength=len(class_names))
            for j, wrong_class in enumerate(class_names):
                if wrong_counts[j] > 0 and j != i:
                    wrong_distribution[wrong_class] = int(wrong_counts[j])

        summary['misclassification_analysis'][class_name] = wrong_distribution

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logging.info(f"Comprehensive summary saved to: {save_path}")
    return summary


# =========================
# Main Test Function
# =========================
def main():
    """Main test function"""
    logger, log_path = setup_test_logging()
    logger.info("Starting model testing...")

    # Create results directory
    os.makedirs("./results", exist_ok=True)

    # Load test data
    logger.info("Loading test data...")
    X_dna_test, X_rna_test, y_test = load_split(config['data_dir'], 'test')

    if X_dna_test is None:
        logger.error("Test data loading failed, exiting test")
        return

    logger.info(f"Test data shapes - DNA: {X_dna_test.shape}, RNA: {X_rna_test.shape}, Labels: {y_test.shape}")
    logger.info(f"Test set label distribution: {np.bincount(y_test)}")

    # Create test dataset and dataloader
    test_dataset = MultiOmicsDataset(X_dna_test, X_rna_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    # Define loss function for evaluation
    criterion = nn.CrossEntropyLoss()

    # Load model
    model_path = config.get('model_save_path', './models/best_model.pth')
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return

    model = load_trained_model(
        model_path=model_path,
        dna_dim=X_dna_test.shape[1],
        rna_dim=X_rna_test.shape[1],
        num_classes=len(config['class_names']),
        config=config
    )

    if model is None:
        return

    # Evaluate model with loss calculation
    logger.info("Starting model evaluation...")
    predictions, targets, probabilities, dna_features, rna_features, test_loss = evaluate_model_with_loss(
        model, test_loader, config, criterion
    )

    # Calculate comprehensive metrics
    metrics, report = calculate_comprehensive_metrics(targets, predictions, probabilities, config['class_names'])

    # Generate detailed report
    generate_detailed_report(targets, predictions, config['class_names'], test_loss)

    # Create visualizations
    logger.info("Generating visualizations...")
    cm = plot_confusion_matrix(
        targets, predictions, config['class_names'],
        './results/confusion_matrix.png'
    )

    true_counts, class_accuracies = plot_class_performance(
        targets, predictions, config['class_names'],
        './results/class_performance.png'
    )

    plot_metrics_comparison(
        metrics, config['class_names'],
        './results/metrics_comparison.png'
    )

    # Save detailed results
    logger.info("Saving detailed results...")
    results_df = save_detailed_results(
        targets, predictions, probabilities, dna_features, rna_features,
        config['class_names'], metrics, test_loss, './results/detailed_results.csv'
    )

    summary = save_comprehensive_summary(
        targets, predictions, probabilities, config['class_names'],
        metrics, test_loss, cm, './results/comprehensive_summary.json'
    )

    # Print detailed per-class analysis
    logger.info("\n" + "=" * 80)
    logger.info("DETAILED PER-CLASS ANALYSIS")
    logger.info("=" * 80)

    for i, class_name in enumerate(config['class_names']):
        class_mask = targets == i
        class_count = np.sum(class_mask)
        class_correct = np.sum(predictions[class_mask] == i)
        class_accuracy = class_correct / class_count if class_count > 0 else 0

        logger.info(f"\n{class_name}:")
        logger.info(f"  Sample count: {class_count}")
        logger.info(f"  Correct predictions: {class_correct}")
        logger.info(f"  Accuracy: {class_accuracy:.4f}")
        logger.info(f"  F1-Score: {metrics['per_class'][class_name]['f1_score']:.4f}")
        logger.info(f"  Precision: {metrics['per_class'][class_name]['precision']:.4f}")
        logger.info(f"  Recall: {metrics['per_class'][class_name]['recall']:.4f}")

        # Misclassification analysis
        wrong_mask = class_mask & (predictions != i)
        wrong_predictions = predictions[wrong_mask]
        if len(wrong_predictions) > 0:
            wrong_counts = np.bincount(wrong_predictions, minlength=len(config['class_names']))
            logger.info(f"  Misclassification distribution:")
            for j, wrong_class in enumerate(config['class_names']):
                if wrong_counts[j] > 0 and j != i:
                    logger.info(f"    → {wrong_class}: {wrong_counts[j]} samples")

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST COMPLETION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Overall Accuracy: {metrics['overall']['accuracy']:.4f}")
    logger.info(f"Macro F1-Score: {metrics['overall']['macro_f1']:.4f}")
    logger.info(f"Weighted F1-Score: {metrics['overall']['weighted_f1']:.4f}")
    logger.info(f"Macro Precision: {metrics['overall']['macro_precision']:.4f}")
    logger.info(f"Macro Recall: {metrics['overall']['macro_recall']:.4f}")

    for i, class_name in enumerate(config['class_names']):
        logger.info(f"{class_name} F1-Score: {metrics['per_class'][class_name]['f1_score']:.4f}")

    logger.info(f"All results saved to ./results/ directory")
    logger.info(f"Detailed log: {log_path}")


if __name__ == "__main__":
    main()
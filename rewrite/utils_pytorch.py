"""
PyTorch utilities for CNN sentence classification
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader


class SentenceDataset(Dataset):
    """
    PyTorch Dataset for sentence classification
    """

    def __init__(self, sentences, labels, max_length=None):
        self.sentences = sentences
        self.labels = labels
        self.max_length = max_length

        if max_length is None:
            self.max_length = max(len(sent) for sent in sentences)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]

        # Pad or truncate sentence
        if len(sentence) < self.max_length:
            sentence = sentence + [0] * (self.max_length - len(sentence))
        else:
            sentence = sentence[: self.max_length]

        return torch.LongTensor(sentence), torch.LongTensor([label])


def create_data_loaders(
    train_data, val_data, test_data, batch_size=50, max_length=None
):
    """
    Create PyTorch DataLoaders from data

    Args:
        train_data: tuple of (sentences, labels)
        val_data: tuple of (sentences, labels)
        test_data: tuple of (sentences, labels)
        batch_size: batch size for training
        max_length: maximum sentence length (None for automatic)

    Returns:
        train_loader, val_loader, test_loader
    """

    # Determine max length if not provided
    if max_length is None:
        all_sentences = train_data[0] + val_data[0] + test_data[0]
        max_length = max(len(sent) for sent in all_sentences)

    # Create datasets
    train_dataset = SentenceDataset(train_data[0], train_data[1], max_length)
    val_dataset = SentenceDataset(val_data[0], val_data[1], max_length)
    test_dataset = SentenceDataset(test_data[0], test_data[1], max_length)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )

    return train_loader, val_loader, test_loader


def create_optimizer(
    model, optimizer_type="adadelta", learning_rate=1.0, weight_decay=0
):
    """
    Create optimizer for training

    Args:
        model: PyTorch model
        optimizer_type: 'adadelta', 'adam', 'sgd'
        learning_rate: learning rate
        weight_decay: L2 regularization

    Returns:
        optimizer
    """

    if optimizer_type.lower() == "adadelta":
        optimizer = torch.optim.Adadelta(
            model.parameters(),
            lr=learning_rate,
            rho=0.95,
            eps=1e-6,
            weight_decay=weight_decay,
        )
    elif optimizer_type.lower() == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer_type.lower() == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    return optimizer


def get_device():
    """
    Get the best available device (GPU if available, else CPU)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print(
            f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )
    else:
        device = torch.device("cpu")
        print("Using CPU")

    return device


def set_seed(seed=42):
    """
    Set random seeds for reproducibility
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_model(model, filepath, additional_info=None):
    """
    Save model and additional information
    """
    save_dict = {
        "model_state_dict": model.state_dict(),
        "model_class": model.__class__.__name__,
    }

    if additional_info:
        save_dict.update(additional_info)

    torch.save(save_dict, filepath)
    print(f"Model saved to {filepath}")


def load_model(model, filepath, device=None):
    """
    Load model from file
    """
    if device is None:
        device = get_device()

    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    print(f"Model loaded from {filepath}")
    return model


def count_parameters(model):
    """
    Count total and trainable parameters in model
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    return total_params, trainable_params


def get_model_size(model):
    """
    Get model size in MB
    """
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    model_size = (param_size + buffer_size) / 1024 / 1024
    print(f"Model size: {model_size:.2f} MB")

    return model_size


def warmup_gpu():
    """
    Warm up GPU for more accurate timing
    """
    if torch.cuda.is_available():
        # Warm up
        dummy_input = torch.randn(1, 100, 300).cuda()
        for _ in range(10):
            _ = torch.sum(dummy_input)
        torch.cuda.synchronize()


class AverageMeter:
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy_score(predictions, targets):
    """
    Calculate accuracy score
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    return np.mean(predictions == targets)


def confusion_matrix(predictions, targets, num_classes=2):
    """
    Create confusion matrix
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(len(predictions)):
        matrix[targets[i], predictions[i]] += 1

    return matrix


def print_confusion_matrix(cm, class_names=None):
    """
    Print confusion matrix in a nice format
    """
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(cm))]

    print("\nConfusion Matrix:")
    print("Predicted")
    header = "Actual".ljust(10) + " ".join(f"{name:>8}" for name in class_names)
    print(header)

    for i, name in enumerate(class_names):
        row = name.ljust(10) + " ".join(f"{cm[i, j]:8d}" for j in range(len(cm)))
        print(row)


def calculate_metrics(predictions, targets, average="binary"):
    """
    Calculate precision, recall, and F1 score
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    # For binary classification
    if average == "binary":
        tp = np.sum((predictions == 1) & (targets == 1))
        fp = np.sum((predictions == 1) & (targets == 0))
        fn = np.sum((predictions == 0) & (targets == 1))
        tn = np.sum((predictions == 0) & (targets == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": (tp + tn) / (tp + tn + fp + fn),
        }

    return None

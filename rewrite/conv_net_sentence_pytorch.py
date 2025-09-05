"""
PyTorch implementation of CNN for Sentence Classification
With GPU support and complete training pipeline
"""

import pickle
import numpy as np
import torch
import torch.nn as nn
import sys
import time
import argparse
from collections import defaultdict

# Import our PyTorch modules
from conv_net_classes_pytorch import CNNSentenceClassifier, CNNTrainer, create_model
from utils_pytorch import (
    create_data_loaders,
    create_optimizer,
    get_device,
    set_seed,
    save_model,
    load_model,
    count_parameters,
    get_model_size,
    calculate_metrics,
    print_confusion_matrix,
    confusion_matrix,
)


def prepare_data(datasets, batch_size=50):
    """
    Prepare data for PyTorch training

    Args:
        datasets: [train_data, test_data] from original format
        batch_size: batch size for training

    Returns:
        train_loader, val_loader, test_loader, max_length
    """
    train_data, test_data = datasets

    # Convert to lists of sentences and labels
    train_sentences = train_data[:, :-1].tolist()
    train_labels = train_data[:, -1].tolist()

    test_sentences = test_data[:, :-1].tolist()
    test_labels = test_data[:, -1].tolist()

    # Split training data into train/validation (90/10)
    split_idx = int(0.9 * len(train_sentences))

    val_sentences = train_sentences[split_idx:]
    val_labels = train_labels[split_idx:]

    train_sentences = train_sentences[:split_idx]
    train_labels = train_labels[:split_idx]

    # Find max length
    all_sentences = train_sentences + val_sentences + test_sentences
    max_length = max(len(sent) for sent in all_sentences)

    print(f"Dataset statistics:")
    print(f"  Train samples: {len(train_sentences)}")
    print(f"  Validation samples: {len(val_sentences)}")
    print(f"  Test samples: {len(test_sentences)}")
    print(f"  Max sentence length: {max_length}")

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        (train_sentences, train_labels),
        (val_sentences, val_labels),
        (test_sentences, test_labels),
        batch_size=batch_size,
        max_length=max_length,
    )

    return train_loader, val_loader, test_loader, max_length


def train_conv_net_pytorch(
    datasets,
    embeddings,
    vocab_size,
    batch_size=50,
    n_epochs=25,
    static_embeddings=False,
    optimizer_type="adadelta",
    learning_rate=1.0,
    clip_grad_norm=9.0,
    early_stopping_patience=5,
    device=None,
):
    """
    Train CNN using PyTorch

    Args:
        datasets: [train_data, test_data]
        embeddings: pretrained word embeddings
        vocab_size: vocabulary size
        batch_size: batch size
        n_epochs: number of epochs
        static_embeddings: whether to freeze embeddings
        optimizer_type: 'adadelta', 'adam', 'sgd'
        learning_rate: learning rate
        clip_grad_norm: gradient clipping norm
        early_stopping_patience: patience for early stopping
        device: device to use

    Returns:
        test_accuracy
    """

    if device is None:
        device = get_device()

    # Prepare data
    train_loader, val_loader, test_loader, max_length = prepare_data(
        datasets, batch_size
    )

    # Create model
    model = CNNSentenceClassifier(
        vocab_size=vocab_size,
        embed_dim=300,
        filter_sizes=[3, 4, 5],
        num_filters=100,
        num_classes=2,
        dropout_rate=0.5,
        static_embeddings=static_embeddings,
        pretrained_embeddings=embeddings,
    )

    model = model.to(device)

    # Print model info
    print(f"\nModel architecture:")
    print(model)
    print()
    count_parameters(model)
    get_model_size(model)
    print()

    # Create trainer
    trainer = CNNTrainer(model, device)

    # Create optimizer
    optimizer = create_optimizer(model, optimizer_type, learning_rate)

    print(f"Training with {optimizer.__class__.__name__} optimizer")
    print(f"Device: {device}")
    print(f"Static embeddings: {static_embeddings}")
    print()

    # Train model
    start_time = time.time()

    training_results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=n_epochs,
        clip_grad_norm=clip_grad_norm,
        early_stopping_patience=early_stopping_patience,
    )

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    print(f"Best validation accuracy: {training_results['best_val_acc']:.4f}")

    # Test evaluation
    print("\nEvaluating on test set...")
    test_loss, test_accuracy = trainer.evaluate(test_loader)
    print(f"Test accuracy: {test_accuracy:.4f}")

    # Detailed test evaluation
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).squeeze()

            logits = model(batch_x)
            predictions = torch.argmax(logits, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())

    # Calculate detailed metrics
    metrics = calculate_metrics(np.array(all_predictions), np.array(all_targets))
    print(f"\nDetailed test metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1-score: {metrics['f1']:.4f}")

    # Confusion matrix
    cm = confusion_matrix(all_predictions, all_targets)
    print_confusion_matrix(cm, ["Negative", "Positive"])

    return test_accuracy


def get_idx_from_sent(sent, word_idx_map, max_l=51, k=300, filter_h=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = filter_h - 1
    for i in range(pad):
        x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l + 2 * pad:
        x.append(0)
    return x


def make_idx_data_splits(revs, word_idx_map, max_l=51, k=300, filter_h=5):
    """
    Transforms sentences into train/validation splits based on existing split field.
    Returns [train_data, validation_data] where split=0 is train, split=1 is validation
    """
    train, validation = [], []
    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l, k, filter_h)
        sent.append(rev["y"])
        if rev["split"] == 0:  # training data
            train.append(sent)
        elif rev["split"] == 1:  # validation data
            validation.append(sent)

    train = np.array(train, dtype="int")
    validation = np.array(validation, dtype="int")
    return [train, validation]


def load_test_data(test_file_path, word_idx_map, max_l=51, k=300, filter_h=5):
    """
    Load test data from separate CSV file
    """
    try:
        # Simple CSV parsing without pandas
        test_data = []

        with open(test_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

            # Skip header if exists
            start_idx = (
                1
                if lines[0].strip().lower().startswith(("text", "sentence", "review"))
                else 0
            )

            for line in lines[start_idx:]:
                line = line.strip()
                if not line:
                    continue

                # Assuming format: "text,label" or "label,text"
                parts = line.split(",")
                if len(parts) >= 2:
                    # Try both orders: text,label and label,text
                    try:
                        # Try text,label format first
                        text = ",".join(parts[:-1]).strip().strip("\"'")
                        label = int(parts[-1].strip())
                    except ValueError:
                        # Try label,text format
                        try:
                            label = int(parts[0].strip())
                            text = ",".join(parts[1:]).strip().strip("\"'")
                        except ValueError:
                            continue

                    sent = get_idx_from_sent(text, word_idx_map, max_l, k, filter_h)
                    sent.append(label)
                    test_data.append(sent)

        if test_data:
            test_data = np.array(test_data, dtype="int")
            print(f"Loaded {len(test_data)} test samples from {test_file_path}")
            return test_data
        else:
            print(f"No valid data found in {test_file_path}")
            return None

    except FileNotFoundError:
        print(f"Test file {test_file_path} not found. Using validation data as test.")
        return None
    except Exception as e:
        print(f"Error loading test file: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="CNN for Sentence Classification (PyTorch)"
    )
    parser.add_argument("data_file", help="Pickled data file (mr.p)")
    parser.add_argument("mode", choices=["static", "nonstatic"], help="Training mode")
    parser.add_argument(
        "embeddings", choices=["rand", "word2vec"], help="Word embeddings to use"
    )
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size")
    parser.add_argument("--lr", type=float, default=1.0, help="Learning rate")
    parser.add_argument(
        "--optimizer",
        choices=["adadelta", "adam", "sgd"],
        default="adadelta",
        help="Optimizer",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--test-file", type=str, default="test.csv", help="Test data CSV file"
    )

    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)

    print("Loading data...", end="")
    with open(args.data_file, "rb") as f:
        x = pickle.load(f)
    revs, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]
    print("data loaded!")

    print(f"Number of sentences: {len(revs)}")
    print(f"Vocabulary size: {len(vocab)}")

    # Set model parameters
    if args.mode == "nonstatic":
        print("Model architecture: CNN-non-static")
        static_embeddings = False
    else:
        print("Model architecture: CNN-static")
        static_embeddings = True

    if args.embeddings == "rand":
        print("Using: random vectors")
        embeddings = W2
    else:
        print("Using: word2vec vectors")
        embeddings = W

    vocab_size = embeddings.shape[0]

    # Prepare train/validation data from pickle
    print("\nPreparing train/validation data from pickle...")
    train_val_datasets = make_idx_data_splits(
        revs, word_idx_map, max_l=56, k=300, filter_h=5
    )

    # Load test data from separate CSV file
    print(f"Loading test data from {args.test_file}...")
    test_data = load_test_data(
        args.test_file, word_idx_map, max_l=56, k=300, filter_h=5
    )

    if test_data is not None:
        # Use separate test file
        datasets = [train_val_datasets[0], test_data]  # [train, test]
        print(f"Using separate test file with {len(test_data)} samples")
    else:
        # Fallback to using validation as test
        datasets = train_val_datasets  # [train, validation_as_test]
        print(
            f"Using validation data as test with {len(train_val_datasets[1])} samples"
        )

    # Train model
    perf = train_conv_net_pytorch(
        datasets=datasets,
        embeddings=embeddings,
        vocab_size=vocab_size,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        static_embeddings=static_embeddings,
        optimizer_type=args.optimizer,
        learning_rate=args.lr,
    )

    print(f"Final test performance: {perf:.4f}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Default arguments for testing
        sys.argv = [
            "conv_net_sentence_pytorch.py",
            "mr.p",
            "nonstatic",
            "rand",
            "--epochs",
            "3",
        ]

    main()

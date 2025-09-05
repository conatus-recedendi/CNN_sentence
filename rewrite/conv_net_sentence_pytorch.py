"""
PyTorch implementation of CNN for Sentence Classification
With GPU support andef train_conv_net_pytorch(
    datasets,
    embeddings,
    vocab_size,
    num_classes,
    batch_size=50,
    n_epochs=25,
    static_embeddings=False,
    optimizer_type="adadelta",
    learning_rate=1.0,
    clip_grad_norm=9.0,
    early_stopping_patience=5,
    device=None,
):aining pipeline
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
    print(train_sentences[100])
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
    num_classes,
    batch_size=50,
    n_epochs=25,
    static_embeddings=False,
    optimizer_type="adadelta",
    learning_rate=1.0,
    clip_grad_norm=3.0,
    early_stopping_patience=10,
    device=None,
):
    """
    Train CNN using PyTorch

    Args:
        datasets: [train_data, test_data]
        embeddings: pretrained word embeddings
        vocab_size: vocabulary size
        num_classes: number of classes
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
        num_classes=num_classes,
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
    print(f"\nTraining completed in {training_time:.2f} seconds", file=sys.stderr)
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
    cm = confusion_matrix(all_predictions, all_targets, num_classes=num_classes)
    print_confusion_matrix(cm, ["Negative", "Positive"])

    return test_accuracy


def get_idx_from_sent(sent, word_idx_map, max_l=51, k=300, filter_h=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = filter_h - 1

    # Add padding at the beginning
    for i in range(pad):
        x.append(0)

    # Convert words to indices
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
        else:
            x.append(0)  # Unknown word -> 0

    # Truncate if too long, pad if too short
    target_length = max_l + 2 * pad
    if len(x) > target_length:
        x = x[:target_length]  # Truncate
    else:
        # Pad to target length
        while len(x) < target_length:
            x.append(0)

    return x


def make_idx_data_splits(revs, word_idx_map, max_l=51, k=300, filter_h=5):
    """
    Transforms sentences into train/validation splits based on existing split field.
    Returns [train_data, validation_data] where split=0 is train, split=1 is validation
    """
    train, validation = [], []

    # Debug: 실제 데이터 분포 확인
    split_counts = {}
    sentence_lengths = []

    for rev in revs:
        split_val = rev["split"]
        split_counts[split_val] = split_counts.get(split_val, 0) + 1
        sentence_lengths.append(len(rev["text"].split()))

    print(f"Split distribution: {split_counts}")
    print(f"Actual max sentence length: {max(sentence_lengths)}")
    print(f"Using max_l: {max_l}")

    # 실제 최대 길이가 max_l보다 크면 조정
    actual_max_l = max(max_l, max(sentence_lengths))
    if actual_max_l > max_l:
        print(f"Adjusting max_l from {max_l} to {actual_max_l}")
        max_l = actual_max_l

    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l, k, filter_h)
        sent.append(rev["y"])

        # 원래 논문에서는 split=0이 첫 번째 fold이므로 train으로 사용
        if rev["split"] == 0:
            train.append(sent)
        elif rev["split"] == 1:
            validation.append(sent)
        # 다른 split 값들은 무시하거나 train에 추가
        else:
            train.append(sent)

    print(f"Train samples: {len(train)}, Validation samples: {len(validation)}")

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

    # Calculate number of classes from the data
    unique_labels = set(rev["y"] for rev in revs)
    num_classes = len(unique_labels)
    print(f"Number of classes: {num_classes} (labels: {sorted(unique_labels)})")

    # Set model parameters
    if args.mode == "nonstatic":
        print("Model architecture: CNN-non-static")
        print("Model architecture: CNN-non-static", file=sys.stderr)
        static_embeddings = False
    else:
        print("Model architecture: CNN-static")
        print("Model architecture: CNN-static", file=sys.stderr)
        static_embeddings = True

    # print(W2[3])
    # print(W[3])
    if args.embeddings == "rand":
        print("Using: random vectors")
        print("Using: random vectors", file=sys.stderr)
        embeddings = W2
    else:
        print("Using: word2vec vectors")
        print("Using: word2vec vectors", file=sys.stderr)
        embeddings = W

    vocab_size = embeddings.shape[0]

    # Prepare train/validation data from pickle
    print("\nPreparing train/validation data from pickle...")
    train_val_datasets = make_idx_data_splits(
        revs, word_idx_map, max_l=100, k=300, filter_h=5
    )

    # Load test data from separate CSV file
    print(f"Loading test data from {args.test_file}...")
    test_data = load_test_data(
        args.test_file, word_idx_map, max_l=100, k=300, filter_h=5
    )
    print(train_val_datasets[0].shape, train_val_datasets[1].shape)
    if test_data is not None:
        # Use separate test file
        datasets = [
            # #nucpy concatate train and validation for training
            # train_val_datasets[0] + train_val_datasets[1], # ValueError: operands could not be broadcast together with shapes (8663,65) (931,65) !
            np.concatenate([train_val_datasets[0], train_val_datasets[1]], axis=0),
            test_data,
        ]  # [train, test]
        print(f"Using separate test file with {len(test_data)} samples")
    else:
        # Fallback to using validation as test
        datasets = train_val_datasets  # [train, validation_as_test]
        print(
            f"Using validation data as test with {len(train_val_datasets[1])} samples"
        )
    #   #
    # Train model
    perf = train_conv_net_pytorch(
        datasets=datasets,
        embeddings=embeddings,
        vocab_size=vocab_size,
        num_classes=num_classes,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        static_embeddings=static_embeddings,
        optimizer_type=args.optimizer,
        learning_rate=args.lr,
        early_stopping_patience=10,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    print(f"Final test performance: {perf:.4f}")
    # print to error
    print(f"Final test performance: {perf:.4f}", file=sys.stderr)


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

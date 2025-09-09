"""
PyTorch implementation of CNN for Sentence Classification
With GPU support and proper backpropagation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys


class HiddenLayer(nn.Module):
    """
    PyTorch implementation of HiddenLayer
    """

    def __init__(self, n_in, n_out, activation=None, use_bias=True, dropout_rate=0.0):
        super(HiddenLayer, self).__init__()
        self.linear = nn.Linear(n_in, n_out, bias=use_bias)
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

        # Initialize weights similar to original implementation
        if activation == F.relu:
            nn.init.normal_(self.linear.weight, mean=0, std=0.01)
        else:
            nn.init.xavier_uniform_(self.linear.weight)

        if use_bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        output = self.linear(x)

        if self.activation is not None:
            output = self.activation(output)

        if self.dropout is not None:
            output = self.dropout(output)

        return output


class MLPDropout(nn.Module):
    """
    Multi-layer perceptron with dropout for classification
    """

    def __init__(self, layer_sizes, dropout_rates, activations):
        super(MLPDropout, self).__init__()

        self.layers = nn.ModuleList()

        # Build hidden layers
        for i in range(len(layer_sizes) - 1):
            n_in, n_out = layer_sizes[i], layer_sizes[i + 1]

            if i < len(activations):
                activation = activations[i]
            else:
                activation = None

            if i < len(dropout_rates):
                dropout_rate = dropout_rates[i]
            else:
                dropout_rate = 0.0

            layer = HiddenLayer(
                n_in=n_in, n_out=n_out, activation=activation, dropout_rate=dropout_rate
            )
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Conv1dPoolLayer(nn.Module):
    """
    1D Convolutional layer with max pooling for sentence classification
    """

    def __init__(self, num_filters, embed_dim, filter_size, activation="relu"):
        super(Conv1dPoolLayer, self).__init__()

        self.conv = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=num_filters,
            kernel_size=filter_size,
            padding=0,
        )

        if activation == "relu":
            self.activation = F.relu
        elif activation == "tanh":
            self.activation = torch.tanh
        else:
            self.activation = None

        # Initialize weights
        nn.init.uniform_(self.conv.weight, -0.01, 0.01)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        # x shape: (batch_size, seq_len, embed_dim)
        # Conv1d expects: (batch_size, embed_dim, seq_len)
        x = x.transpose(1, 2)  # (batch_size, embed_dim, seq_len)

        # Convolution
        conv_out = self.conv(x)  # (batch_size, num_filters, conv_len)

        # Apply activation
        if self.activation is not None:
            conv_out = self.activation(conv_out)

        # Global max pooling
        pooled = F.max_pool1d(
            conv_out, kernel_size=conv_out.size(2)
        )  # (batch_size, num_filters, 1)
        pooled = pooled.squeeze(2)  # (batch_size, num_filters)

        return pooled


class CNNSentenceClassifier(nn.Module):
    """
    Complete CNN model for sentence classification
    """

    def __init__(
        self,
        vocab_size,
        embed_dim=300,
        filter_sizes=[3, 4, 5],
        num_filters=100,
        num_classes=2,
        dropout_rate=0.5,
        static_embeddings="static",
        pretrained_embeddings=None,
        embeddings_type="word2vec",
    ):
        super(CNNSentenceClassifier, self).__init__()

        self.static_embeddings = static_embeddings

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Initialize embeddings
        if embeddings_type == "word2vec":
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
        else:
            variance = (
                pretrained_embeddings.std()
                if pretrained_embeddings is not None
                else 0.25
            )
            print("Embedding variance:", variance, file=sys.stderr)

            nn.init.uniform_(self.embedding.weight, -variance, variance)

        # Set padding token to zero
        self.embedding.weight.data[0].fill_(0)

        # Freeze embeddings if static
        if static_embeddings == "static":
            self.embedding.weight.requires_grad = False
        elif static_embeddings == "multichannel":
            self.embedding_multi = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.embedding_multi.weight.data.copy_(
                torch.from_numpy(pretrained_embeddings)
            )
            self.embedding_multi.weight.requires_grad = False

        # Convolutional layers
        self.conv_layers = nn.ModuleList(
            [
                Conv1dPoolLayer(num_filters, embed_dim, filter_size)
                for filter_size in filter_sizes
            ]
        )

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

        # Fully connected layer
        # For multichannel: each filter produces 2*num_filters output
        fc_input_dim = len(filter_sizes) * num_filters
        if static_embeddings == "multichannel":
            fc_input_dim = len(filter_sizes) * num_filters * 2  # 2 channels

        self.fc = nn.Linear(fc_input_dim, num_classes)

        # Initialize FC layer
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        # x shape: (batch_size, seq_len)

        # Embedding lookup
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        if self.static_embeddings == "multichannel":
            embedded_multi = self.embedding_multi(x)
            # Reset padding embeddings to zero for both channels
            # self.embedding.weight.data[0].fill_(0)
            self.embedding_multi.weight.data[0].fill_(0)
            embedded = torch.stack([embedded, embedded_multi], dim=1)

        # Reset padding embeddings to zero (for non-static case)
        if self.static_embeddings == "nonstatic":
            self.embedding.weight.data[0].fill_(0)

        # Apply convolutions
        conv_outputs = []
        if self.static_embeddings == "multichannel":
            # embedded shape: (batch_size, 2, seq_len, embed_dim)
            embedded_ch1 = embedded[:, 0, :, :]  # (batch_size, seq_len, embed_dim)
            embedded_ch2 = embedded[:, 1, :, :]  # (batch_size, seq_len, embed_dim)

            for conv_layer in self.conv_layers:
                conv_out_ch1 = conv_layer(embedded_ch1)  # (batch_size, num_filters)
                conv_out_ch2 = conv_layer(embedded_ch2)  # (batch_size, num_filters)
                conv_out = torch.cat(
                    [conv_out_ch1, conv_out_ch2], dim=1
                )  # (batch_size, 2*num_filters)
                conv_outputs.append(conv_out)
        else:
            for conv_layer in self.conv_layers:
                conv_out = conv_layer(embedded)  # (batch_size, num_filters)
                conv_outputs.append(conv_out)

        # Concatenate all conv outputs
        concat_output = torch.cat(
            conv_outputs, dim=1
        )  # (batch_size, len(filter_sizes) * num_filters)

        # Apply dropout
        dropped = self.dropout(concat_output)

        # Final classification layer
        logits = self.fc(dropped)  # (batch_size, num_classes)

        return logits

    def predict_proba(self, x):
        """Get class probabilities"""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

    def predict(self, x):
        """Get class predictions"""
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)


class CNNTrainer:
    """
    Trainer class for CNN sentence classifier
    """

    def __init__(self, model, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, data_loader, optimizer, clip_grad_norm=None):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            logits = self.model(batch_x)

            # Ensure target is 1D for CrossEntropyLoss
            batch_y = batch_y.squeeze()
            loss = self.criterion(logits, batch_y)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad_norm)

            # Update weights
            optimizer.step()

            # Statistics
            total_loss += loss.item()
            predicted = torch.argmax(logits, dim=1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        avg_loss = total_loss / len(data_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def evaluate(self, data_loader):
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device).squeeze()

                # Forward pass
                logits = self.model(batch_x)
                loss = self.criterion(logits, batch_y)

                # Statistics
                total_loss += loss.item()
                predicted = torch.argmax(logits, dim=1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        avg_loss = total_loss / len(data_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def train(
        self,
        train_loader,
        val_loader,
        optimizer,
        num_epochs,
        clip_grad_norm=9.0,
        early_stopping_patience=10,
    ):
        """
        Complete training loop with early stopping
        """
        best_val_acc = 0
        patience_counter = 0
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []

        for epoch in range(num_epochs):
            # Training
            train_loss, train_acc = self.train_epoch(
                train_loader, optimizer, clip_grad_norm
            )

            # Validation
            val_loss, val_acc = self.evaluate(val_loader)

            # Record metrics
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            # print(f"Epoch {epoch+1}/{num_epochs}:")
            # print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            # print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), "best_model.pth")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping after {epoch+1} epochs")
                    break

        # Load best model
        self.model.load_state_dict(torch.load("best_model.pth"))

        return {
            "train_losses": train_losses,
            "train_accs": train_accs,
            "val_losses": val_losses,
            "val_accs": val_accs,
            "best_val_acc": best_val_acc,
        }


def create_model(vocab_size, embeddings=None, static=False, device="cuda"):
    """
    Factory function to create CNN model
    """
    model = CNNSentenceClassifier(
        vocab_size=vocab_size,
        embed_dim=300,
        filter_sizes=[3, 4, 5],
        num_filters=100,
        num_classes=2,
        dropout_rate=0.5,
        static_embeddings=static,
        pretrained_embeddings=embeddings,
    )

    return model.to(device)


# Activation functions for backward compatibility
def ReLU(x):
    return F.relu(x)


def Sigmoid(x):
    return torch.sigmoid(x)


def Tanh(x):
    return torch.tanh(x)


def Iden(x):
    return x

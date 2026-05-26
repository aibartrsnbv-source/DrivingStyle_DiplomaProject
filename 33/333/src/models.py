"""
Machine Learning Models Module for Driving Style ML Project
============================================================

This module implements both classical machine learning models and deep learning
models using PyTorch for driving style classification and accident risk prediction.

Models Implemented:
- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost (optional)
- MLP (Multi-Layer Perceptron) - PyTorch
- LSTM (Long Short-Term Memory) - PyTorch

Author: [Your Name]
Project: Bachelor Diploma - Driving Style Assessment and Accident Risk Prediction
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

sys.path.append(str(Path(__file__).parent.parent))

from src.config import (
    ML_MODELS_CONFIG,
    DL_CONFIG,
    RANDOM_SEED,
    TORCH_SEED,
)

np.random.seed(RANDOM_SEED)
torch.manual_seed(TORCH_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(TORCH_SEED)


# =============================================================================
# CLASSICAL MACHINE LEARNING MODELS
# =============================================================================

class ModelFactory:
    """
    Factory class for creating machine learning models.

    This class provides a centralized way to instantiate different ML models
    with their configured hyperparameters.
    """

    @staticmethod
    def get_model(model_name: str, **kwargs) -> BaseEstimator:
        """
        Get a configured model instance.

        Parameters:
        -----------
        model_name : str
            Name of the model to create
        **kwargs : dict
            Additional parameters to override defaults

        Returns:
        --------
        BaseEstimator
            Configured model instance
        """
        model_configs = ML_MODELS_CONFIG.copy()

        if model_name == "logistic_regression":
            config = {**model_configs["logistic_regression"], **kwargs}
            return LogisticRegression(**config)

        elif model_name == "random_forest":
            config = {**model_configs["random_forest"], **kwargs}
            return RandomForestClassifier(**config)

        elif model_name == "gradient_boosting":
            config = {**model_configs["gradient_boosting"], **kwargs}
            return GradientBoostingClassifier(**config)

        elif model_name == "xgboost":
            try:
                from xgboost import XGBClassifier
                config = {**model_configs["xgboost"], **kwargs}
                return XGBClassifier(**config)
            except ImportError:
                print("⚠ XGBoost not installed. Using Gradient Boosting instead.")
                config = {**model_configs["gradient_boosting"], **kwargs}
                return GradientBoostingClassifier(**config)

        else:
            raise ValueError(f"Unknown model: {model_name}")

    @staticmethod
    def get_all_models() -> Dict[str, BaseEstimator]:
        """
        Get all available ML models including a soft-voting ensemble.

        The voting ensemble combines RF + GB + XGBoost with equal weights using
        soft voting (predicted class probabilities are averaged). This typically
        gives +2-3% F1 over the best single model because each base learner
        makes different kinds of mistakes, and averaging corrects some of them.

        Returns:
        --------
        Dict[str, BaseEstimator]
            Dictionary of model name to model instance
        """
        from sklearn.ensemble import VotingClassifier

        rf = ModelFactory.get_model("random_forest")
        gb = ModelFactory.get_model("gradient_boosting")

        models = {
            "Logistic Regression": ModelFactory.get_model("logistic_regression"),
            "Random Forest": rf,
            "Gradient Boosting": gb,
        }

        xgb = None
        try:
            from xgboost import XGBClassifier  # noqa: F401
            xgb = ModelFactory.get_model("xgboost")
            models["XGBoost"] = xgb
        except ImportError:
            pass

        # Build voting ensemble from the strongest base learners.
        # Use fresh instances so they can be fitted independently by VotingClassifier.
        estimators = [
            ("rf", ModelFactory.get_model("random_forest")),
            ("gb", ModelFactory.get_model("gradient_boosting")),
        ]
        if xgb is not None:
            estimators.append(("xgb", ModelFactory.get_model("xgboost")))

        models["Voting Ensemble"] = VotingClassifier(
            estimators=estimators,
            voting="soft",     # use predict_proba, better for imbalanced
            n_jobs=-1,
        )

        return models


# =============================================================================
# PYTORCH DATASETS
# =============================================================================

class TabularDataset(Dataset):
    """
    PyTorch Dataset for tabular driving behavior data.

    This dataset wraps pandas DataFrames or numpy arrays for use with
    PyTorch DataLoaders.

    Attributes:
        X (torch.Tensor): Feature tensor
        y (torch.Tensor): Label tensor
    """

    def __init__(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
    ):
        """
        Initialize the TabularDataset.

        Parameters:
        -----------
        X : DataFrame or ndarray
            Feature data
        y : Series or ndarray
            Labels
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample."""
        return self.X[idx], self.y[idx]


class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for time-series driving data.

    This dataset creates sequences of sensor readings for use with
    recurrent neural networks like LSTM.

    Attributes:
        X (torch.Tensor): Sequence tensor of shape (N, seq_len, features)
        y (torch.Tensor): Label tensor
    """

    def __init__(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        sequence_length: int = 50,
    ):
        """
        Initialize the TimeSeriesDataset.

        Parameters:
        -----------
        X : DataFrame or ndarray
            Feature data
        y : Series or ndarray
            Labels
        sequence_length : int
            Length of sequences to create
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        self.sequence_length = sequence_length
        self.X, self.y = self._create_sequences(X, y)

    def _create_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create sequences from the data.

        Parameters:
        -----------
        X : ndarray
            Feature data
        y : ndarray
            Labels

        Returns:
        --------
        Tuple[Tensor, Tensor]
            Sequences and corresponding labels
        """
        sequences = []
        labels = []

        for i in range(len(X) - self.sequence_length + 1):
            seq = X[i:i + self.sequence_length]
            label = y[i + self.sequence_length - 1]  # Label at end of sequence
            sequences.append(seq)
            labels.append(label)

        return (
            torch.FloatTensor(np.array(sequences)),
            torch.LongTensor(np.array(labels)),
        )

    def __len__(self) -> int:
        """Return the number of sequences."""
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sequence."""
        return self.X[idx], self.y[idx]


# =============================================================================
# DEEP LEARNING MODELS (PyTorch)
# =============================================================================

class MLP(nn.Module):
    """
    Multi-Layer Perceptron for tabular data classification.

    This neural network architecture is suitable for classification tasks
    on structured/tabular driving behavior data.

    Architecture:
    - Input layer
    - Multiple hidden layers with ReLU activation
    - Batch normalization (optional)
    - Dropout for regularization
    - Output layer with softmax

    Attributes:
        layers (nn.ModuleList): List of linear layers
        batch_norms (nn.ModuleList): List of batch normalization layers
        dropout (nn.Dropout): Dropout layer
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_layers: List[int] = None,
        dropout_rate: float = 0.3,
        use_batch_norm: bool = True,
    ):
        """
        Initialize the MLP.

        Parameters:
        -----------
        input_dim : int
            Number of input features
        num_classes : int
            Number of output classes
        hidden_layers : list
            Sizes of hidden layers
        dropout_rate : float
            Dropout probability
        use_batch_norm : bool
            Whether to use batch normalization
        """
        super(MLP, self).__init__()

        if hidden_layers is None:
            hidden_layers = DL_CONFIG["mlp"]["hidden_layers"]

        self.use_batch_norm = use_batch_norm
        self.dropout = nn.Dropout(dropout_rate)

        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim

        self.output_layer = nn.Linear(prev_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Parameters:
        -----------
        x : Tensor
            Input tensor of shape (batch_size, input_dim)

        Returns:
        --------
        Tensor
            Output logits of shape (batch_size, num_classes)
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropout(x)

        x = self.output_layer(x)
        return x

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get probability predictions.

        Parameters:
        -----------
        x : Tensor
            Input tensor

        Returns:
        --------
        Tensor
            Probability predictions
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)


class LSTM(nn.Module):
    """
    LSTM Network for time-series driving data classification.

    This recurrent neural network architecture is suitable for sequential
    sensor data from driving sessions.

    Architecture:
    - LSTM layers (optionally bidirectional)
    - Dropout for regularization
    - Fully connected output layer

    Attributes:
        lstm (nn.LSTM): LSTM layer(s)
        fc (nn.Linear): Fully connected output layer
        dropout (nn.Dropout): Dropout layer
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_size: int = None,
        num_layers: int = None,
        dropout: float = None,
        bidirectional: bool = None,
    ):
        """
        Initialize the LSTM.

        Parameters:
        -----------
        input_dim : int
            Number of input features per timestep
        num_classes : int
            Number of output classes
        hidden_size : int
            Size of LSTM hidden state
        num_layers : int
            Number of LSTM layers
        dropout : float
            Dropout probability
        bidirectional : bool
            Whether to use bidirectional LSTM
        """
        super(LSTM, self).__init__()

        lstm_config = DL_CONFIG["lstm"]
        hidden_size = hidden_size or lstm_config["hidden_size"]
        num_layers = num_layers or lstm_config["num_layers"]
        dropout = dropout if dropout is not None else lstm_config["dropout"]
        bidirectional = bidirectional if bidirectional is not None else lstm_config["bidirectional"]

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(dropout)

        fc_input_dim = hidden_size * self.num_directions
        self.fc = nn.Linear(fc_input_dim, num_classes)

        # Attention mechanism (optional)
        self.attention = nn.Linear(fc_input_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        use_attention: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass through the network.

        Parameters:
        -----------
        x : Tensor
            Input tensor of shape (batch_size, seq_len, input_dim)
        use_attention : bool
            Whether to use attention mechanism

        Returns:
        --------
        Tensor
            Output logits of shape (batch_size, num_classes)
        """
        lstm_out, (hidden, cell) = self.lstm(x)
        # lstm_out shape: (batch_size, seq_len, hidden_size * num_directions)

        if use_attention:
            # Attention mechanism
            attention_weights = F.softmax(self.attention(lstm_out), dim=1)
            context = torch.sum(attention_weights * lstm_out, dim=1)
            out = context
        else:
            if self.bidirectional:
                out = torch.cat((hidden[-2], hidden[-1]), dim=1)
            else:
                out = hidden[-1]

        out = self.dropout(out)
        out = self.fc(out)

        return out

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get probability predictions.

        Parameters:
        -----------
        x : Tensor
            Input tensor

        Returns:
        --------
        Tensor
            Probability predictions
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)


class CNN1D(nn.Module):
    """
    1D Convolutional Neural Network for time-series data.

    This architecture applies 1D convolutions along the time dimension
    to extract local patterns from sensor data.

    Architecture:
    - Multiple 1D convolutional layers
    - Max pooling
    - Fully connected layers
    - Output layer
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        seq_length: int = 50,
        num_filters: List[int] = None,
        kernel_sizes: List[int] = None,
        dropout: float = 0.3,
    ):
        """
        Initialize the CNN1D.

        Parameters:
        -----------
        input_dim : int
            Number of input features per timestep
        num_classes : int
            Number of output classes
        seq_length : int
            Length of input sequences
        num_filters : list
            Number of filters in each conv layer
        kernel_sizes : list
            Kernel sizes for each conv layer
        dropout : float
            Dropout probability
        """
        super(CNN1D, self).__init__()

        if num_filters is None:
            num_filters = [64, 128, 64]
        if kernel_sizes is None:
            kernel_sizes = [3, 3, 3]

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        in_channels = input_dim
        for filters, kernel in zip(num_filters, kernel_sizes):
            self.convs.append(
                nn.Conv1d(in_channels, filters, kernel, padding=kernel // 2)
            )
            self.batch_norms.append(nn.BatchNorm1d(filters))
            in_channels = filters

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters:
        -----------
        x : Tensor
            Input of shape (batch_size, seq_len, input_dim)

        Returns:
        --------
        Tensor
            Output logits
        """
        # Reshape for Conv1d: (batch, channels, seq_len)
        x = x.permute(0, 2, 1)

        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x)
            x = bn(x)
            x = F.relu(x)

        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


# =============================================================================
# PYTORCH MODEL WRAPPER
# =============================================================================

class PyTorchClassifier:
    """
    Sklearn-compatible wrapper for PyTorch models.

    This wrapper allows PyTorch models to be used with the same interface
    as scikit-learn classifiers.

    Attributes:
        model (nn.Module): PyTorch model
        device (torch.device): Device to run on
        is_fitted (bool): Whether the model has been trained
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = None,
        batch_size: int = None,
        epochs: int = None,
        early_stopping_patience: int = None,
        device: str = None,
    ):
        """
        Initialize the wrapper.

        Parameters:
        -----------
        model : nn.Module
            PyTorch model to wrap
        learning_rate : float
            Learning rate for optimizer
        batch_size : int
            Batch size for training
        epochs : int
            Maximum number of training epochs
        early_stopping_patience : int
            Patience for early stopping
        device : str
            Device to use ('cuda' or 'cpu')
        """
        self.model = model

        train_config = DL_CONFIG["training"]
        self.learning_rate = learning_rate or train_config["learning_rate"]
        self.batch_size = batch_size or train_config["batch_size"]
        self.epochs = epochs or train_config["epochs"]
        self.early_stopping_patience = (
            early_stopping_patience or train_config["early_stopping_patience"]
        )

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model.to(self.device)
        self.is_fitted = False
        self.training_history = {"train_loss": [], "val_loss": [], "val_acc": []}

    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        class_weights: Optional[torch.Tensor] = None,
        verbose: bool = True,
    ) -> "PyTorchClassifier":
        """
        Train the PyTorch model.

        Parameters:
        -----------
        X_train : DataFrame or ndarray
            Training features
        y_train : Series or ndarray
            Training labels
        X_val : DataFrame or ndarray, optional
            Validation features
        y_val : Series or ndarray, optional
            Validation labels
        class_weights : Tensor, optional
            Class weights for imbalanced data
        verbose : bool
            Whether to print training progress

        Returns:
        --------
        self
            Fitted classifier
        """
        train_dataset = TabularDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        val_loader = None
        if X_val is not None and y_val is not None:
            val_dataset = TabularDataset(X_val, y_val)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
            )

        if class_weights is not None:
            class_weights = class_weights.to(self.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=DL_CONFIG["training"]["weight_decay"],
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=DL_CONFIG["training"]["lr_scheduler_factor"],
            patience=DL_CONFIG["training"]["lr_scheduler_patience"],
        )

        best_val_loss = float("inf")
        patience_counter = 0

        if verbose:
            print("\n" + "=" * 60)
            print("TRAINING NEURAL NETWORK")
            print("=" * 60)
            print(f"Device: {self.device}")
            print(f"Epochs: {self.epochs}, Batch size: {self.batch_size}")
            print(f"Learning rate: {self.learning_rate}")
            print("-" * 60)

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            self.training_history["train_loss"].append(train_loss)

            if val_loader is not None:
                val_loss, val_acc = self._validate(val_loader, criterion)
                self.training_history["val_loss"].append(val_loss)
                self.training_history["val_acc"].append(val_acc)

                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1

                if verbose and (epoch + 1) % 10 == 0:
                    print(
                        f"Epoch {epoch+1:3d}/{self.epochs} | "
                        f"Train Loss: {train_loss:.4f} | "
                        f"Val Loss: {val_loss:.4f} | "
                        f"Val Acc: {val_acc:.4f}"
                    )

                if patience_counter >= self.early_stopping_patience:
                    if verbose:
                        print(f"\nEarly stopping at epoch {epoch+1}")
                    break
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1:3d}/{self.epochs} | Train Loss: {train_loss:.4f}")

        if hasattr(self, "best_model_state"):
            self.model.load_state_dict(self.best_model_state)

        self.is_fitted = True

        if verbose:
            print("-" * 60)
            print("✓ Training complete")

        return self

    def _validate(
        self,
        val_loader: DataLoader,
        criterion: nn.Module,
    ) -> Tuple[float, float]:
        """
        Validate the model.

        Parameters:
        -----------
        val_loader : DataLoader
            Validation data loader
        criterion : nn.Module
            Loss function

        Returns:
        --------
        Tuple[float, float]
            Validation loss and accuracy
        """
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        val_loss /= len(val_loader)
        val_acc = correct / total

        return val_loss, val_acc

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions.

        Parameters:
        -----------
        X : DataFrame or ndarray
            Input features

        Returns:
        --------
        ndarray
            Predicted class labels
        """
        self.model.eval()

        if isinstance(X, pd.DataFrame):
            X = X.values

        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)

        return predicted.cpu().numpy()

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Get probability predictions.

        Parameters:
        -----------
        X : DataFrame or ndarray
            Input features

        Returns:
        --------
        ndarray
            Class probabilities
        """
        self.model.eval()

        if isinstance(X, pd.DataFrame):
            X = X.values

        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            probas = F.softmax(outputs, dim=1)

        return probas.cpu().numpy()

    def get_training_history(self) -> Dict[str, List[float]]:
        """Get the training history."""
        return self.training_history


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_mlp_classifier(
    input_dim: int,
    num_classes: int,
    **kwargs,
) -> PyTorchClassifier:
    """
    Create an MLP classifier with default settings.

    Parameters:
    -----------
    input_dim : int
        Number of input features
    num_classes : int
        Number of output classes
    **kwargs : dict
        Additional parameters for PyTorchClassifier

    Returns:
    --------
    PyTorchClassifier
        Configured MLP classifier
    """
    model = MLP(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_layers=DL_CONFIG["mlp"]["hidden_layers"],
        dropout_rate=DL_CONFIG["mlp"]["dropout_rate"],
        use_batch_norm=DL_CONFIG["mlp"]["batch_norm"],
    )

    return PyTorchClassifier(model, **kwargs)


def create_lstm_classifier(
    input_dim: int,
    num_classes: int,
    **kwargs,
) -> PyTorchClassifier:
    """
    Create an LSTM classifier with default settings.

    Parameters:
    -----------
    input_dim : int
        Number of input features per timestep
    num_classes : int
        Number of output classes
    **kwargs : dict
        Additional parameters for PyTorchClassifier

    Returns:
    --------
    PyTorchClassifier
        Configured LSTM classifier
    """
    model = LSTM(
        input_dim=input_dim,
        num_classes=num_classes,
    )

    return PyTorchClassifier(model, **kwargs)


def compute_class_weights(y: Union[pd.Series, np.ndarray]) -> torch.Tensor:
    """
    Compute class weights for imbalanced data.

    Parameters:
    -----------
    y : Series or ndarray
        Class labels

    Returns:
    --------
    Tensor
        Class weights
    """
    if isinstance(y, pd.Series):
        y = y.values

    classes, counts = np.unique(y, return_counts=True)
    weights = len(y) / (len(classes) * counts)

    # Normalize weights
    weights = weights / weights.sum() * len(classes)

    return torch.FloatTensor(weights)


# =============================================================================
# MODEL SUMMARY
# =============================================================================

def print_model_summary(model: nn.Module, input_size: Tuple[int, ...]) -> None:
    """
    Print a summary of the PyTorch model.

    Parameters:
    -----------
    model : nn.Module
        PyTorch model
    input_size : tuple
        Input size (excluding batch dimension)
    """
    print("\n" + "=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    print(f"\nModel: {model.__class__.__name__}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Input size: {input_size}")

    print("\nLayers:")
    print("-" * 60)
    for name, module in model.named_modules():
        if name:
            print(f"  {name}: {module.__class__.__name__}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MODELS MODULE DEMONSTRATION")
    print("=" * 60)

    np.random.seed(RANDOM_SEED)
    n_samples = 500
    n_features = 20
    n_classes = 3

    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, n_classes, n_samples)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=RANDOM_SEED
    )

    print(f"\nData shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val: {X_val.shape}")
    print(f"  X_test: {X_test.shape}")

    print("\n" + "-" * 60)
    print("Testing Classical ML Models")
    print("-" * 60)

    for name, model in ModelFactory.get_all_models().items():
        model.fit(X_train, y_train)
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        print(f"  {name}: Train Acc = {train_acc:.4f}, Test Acc = {test_acc:.4f}")

    print("\n" + "-" * 60)
    print("Testing MLP (PyTorch)")
    print("-" * 60)

    mlp_classifier = create_mlp_classifier(n_features, n_classes)
    print_model_summary(mlp_classifier.model, (n_features,))

    mlp_classifier.fit(X_train, y_train, X_val, y_val, verbose=True)
    mlp_preds = mlp_classifier.predict(X_test)
    mlp_acc = (mlp_preds == y_test).mean()
    print(f"\nMLP Test Accuracy: {mlp_acc:.4f}")

    print("\n✓ Models module demonstration complete!")

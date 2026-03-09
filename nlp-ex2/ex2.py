
###################################################
# Exercise 2 - Natural Language Processing 67658  #
###################################################

import numpy as np

# subset of categories that we will use
category_dict = {'comp.graphics': 'computer graphics',
                 'rec.sport.baseball': 'baseball',
                 'sci.electronics': 'science, electronics',
                 'talk.politics.guns': 'politics, guns'
                 }

def get_data(categories=None, portion=1.):
    """
    Get data for given categories and portion
    :param portion: portion of the data to use
    :return:
    """
    # get data
    from sklearn.datasets import fetch_20newsgroups
    data_train = fetch_20newsgroups(categories=categories, subset='train', remove=('headers', 'footers', 'quotes'),
                                    random_state=21)
    data_test = fetch_20newsgroups(categories=categories, subset='test', remove=('headers', 'footers', 'quotes'),
                                   random_state=21)

    # train
    train_len = int(portion *len(data_train.data))
    x_train = np.array(data_train.data[:train_len])
    y_train = data_train.target[:train_len]
    # remove empty entries
    non_empty = x_train != ""
    x_train, y_train = x_train[non_empty].tolist(), y_train[non_empty].tolist()

    # test
    x_test = np.array(data_test.data)
    y_test = data_test.target
    non_empty = np.array(x_test) != ""
    x_test, y_test = x_test[non_empty].tolist(), y_test[non_empty].tolist()
    return x_train, y_train, x_test, y_test


# Q1,2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def MLP_classification(portion=1., model_type="linear"):
    """
    Train and evaluate a model for text classification.

    :param portion: float, portion of the dataset to use (e.g., 0.1, 0.2, etc.).
    :param model_type: str, either "linear" for a single-layer perceptron or "mlp" for a multi-layer perceptron.
    """
    # Step 1: Load and preprocess data
    x_train, y_train, x_test, y_test = get_data(categories=category_dict.keys(), portion=portion)
    vectorizer = TfidfVectorizer(max_features=2000)
    X_train = vectorizer.fit_transform(x_train).toarray()
    X_test = vectorizer.transform(x_test).toarray()

    # Convert data to PyTorch datasets and loaders
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test))
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Step 2: Define the model
    input_dim = 2000  # TF-IDF features
    num_classes = 4  # Number of output classes

    if model_type == "linear":
        # Single-layer perceptron (Task 1)
        model = nn.Linear(input_dim, num_classes)
    elif model_type == "mlp":
        # Multi-layer perceptron (Task 2)
        hidden_dim = 500  # Hidden layer size

        class MLPModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_classes):
                super(MLPModel, self).__init__()
                self.hidden = nn.Linear(input_dim, hidden_dim)
                self.activation = nn.ReLU()
                self.output = nn.Linear(hidden_dim, num_classes)

            def forward(self, x):
                x = self.hidden(x)
                x = self.activation(x)
                x = self.output(x)
                return x

        model = MLPModel(input_dim, hidden_dim, num_classes)
    else:
        raise ValueError("Invalid model_type. Choose either 'linear' or 'mlp'.")

    # Step 3: Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Step 4: Train the model
    num_epochs = 20
    train_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(train_loader))

        # Evaluate on the test set
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.numpy())
                all_labels.extend(labels.numpy())

        val_accuracy = accuracy_score(all_labels, all_preds)
        val_accuracies.append(val_accuracy)

        # print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_losses[-1]:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # Step 5: Plot results
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Loss ({model_type.capitalize()}, Portion: {portion})")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Validation Accuracy ({model_type.capitalize()}, Portion: {portion})")
    plt.legend()

    plt.tight_layout()
    plt.show()

    return val_accuracies[2] if len(val_accuracies) >= 3 else None


# Q3
def transformer_classification(portion=1.):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from torch.utils.data import DataLoader
    import evaluate
    from tqdm import tqdm

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

        def __len__(self):
            return len(self.labels)

    def train_epoch(model, data_loader, optimizer, dev='cpu'):
        model.train()
        total_loss = 0.
        for batch in tqdm(data_loader, desc="Training"):
            input_ids = batch['input_ids'].to(dev)
            attention_mask = batch['attention_mask'].to(dev)
            labels = batch['labels'].to(dev)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(data_loader)
        return avg_loss

    def evaluate_model(model, data_loader, dev='cpu', metric=None):
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(dev)
                attention_mask = batch['attention_mask'].to(dev)
                labels = batch['labels'].to(dev)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        metric_result = metric.compute(predictions=all_preds, references=all_labels)
        return metric_result

    # Load data
    x_train, y_train, x_test, y_test = get_data(categories=category_dict.keys(), portion=portion)

    # Parameters
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_labels = len(category_dict)
    epochs = 3
    batch_size = 16
    learning_rate = 5e-5

    # Model, tokenizer, and metric
    model = AutoModelForSequenceClassification.from_pretrained('distilroberta-base', num_labels=num_labels).to(dev)
    tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
    metric = evaluate.load("accuracy")
    # Count the total and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())  # Total parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # Trainable parameters

    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    # Datasets and DataLoaders
    train_dataset = Dataset(tokenizer(x_train, truncation=True, padding=True), y_train)
    val_dataset = Dataset(tokenizer(x_test, truncation=True, padding=True), y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    train_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, dev)
        train_losses.append(train_loss)
        val_metric = evaluate_model(model, val_loader, dev, metric)
        val_accuracy = val_metric["accuracy"]
        val_accuracies.append(val_accuracy)

    import matplotlib.pyplot as plt

    # Ensure epochs matches the length of train_losses
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 6))

    # Plot Training Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Loss (Transformer classification, Portion: {portion})")
    plt.legend()

    # Plot Validation Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Validation Accuracy (Transformer classification, Portion: {portion})")
    plt.legend()

    plt.tight_layout()
    plt.show()

    return val_accuracies[2] if len(val_accuracies) >= 3 else None



if __name__ == "__main__":
    portions = [0.1, 0.2, 0.5, 1.]
    single_layer_accuracies = []
    multi_layer_accuracies = []
    transformer_accuracies = []
    # Q1 - single layer MLP
    for portion in portions:
        print("portion:", portion)
        single_layer_accuracies.append(MLP_classification(portion, "linear"))

    # Q2 - multi-layer MLP
    for portion in portions:
        print("portion:", portion)
        multi_layer_accuracies.append(MLP_classification(portion, "mlp"))

    # Q3 - Transformer
    print("\nTransformer results:")
    for p in portions[:2]:
        print(f"Portion: {p}")
        transformer_accuracies.append(transformer_classification(portion=p))


    # Create the accuracies vs training data portion size
    plt.figure(figsize=(8, 6))
    plt.plot(portions, single_layer_accuracies, marker='o', linestyle='-', label='LogLinear Model', linewidth=2)
    plt.plot(portions, multi_layer_accuracies, marker='s', linestyle='--', label='MLP Model', linewidth=2)
    plt.plot(portions[:2], transformer_accuracies, marker='^', linestyle='-.', label='Transformer Model', linewidth=2)

    # Adding labels, legend, and title
    plt.xlabel('Portions', fontsize=12)
    plt.ylabel(' Accuracies', fontsize=12)
    plt.title('Comparison of Model Accuracies vs Portions', fontsize=14)
    plt.legend(fontsize=10)

    # Add grid
    plt.grid(True)

    # Show the plot
    plt.show()

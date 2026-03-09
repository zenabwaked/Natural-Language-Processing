import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import exercise_blanks
from data_loader import SentimentTreeBank  # Import the provided dataset class
import matplotlib.pyplot as plt
import data_loader
# Define tokenizer and model name
MODEL_NAME = "distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# Load dataset
dataset = SentimentTreeBank()
# Custom PyTorch dataset wrapper with neutral filtering
class SentimentDataset(Dataset):
    def __init__(self, sentences):
        # Filter out neutral sentiment sentences
        self.sentences = [sent for sent in sentences if sent.sentiment_class in [0.0, 1.0]]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        text = " ".join(sentence.text)  # Convert list of tokens to a string

        # Convert sentiment to binary label (0 = Negative, 1 = Positive)
        sentiment_label = 1 if sentence.sentiment_class == 1.0 else 0

        # Tokenize the text
        encoding = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['label'] = torch.tensor(sentiment_label, dtype=torch.long)
        return item


# Create train, validation, and test datasets
train_dataset = SentimentDataset(dataset.get_train_set())
val_dataset = SentimentDataset(dataset.get_validation_set())
test_dataset = SentimentDataset(dataset.get_test_set())

# Create DataLoaders
BATCH_SIZE = 64
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# Sentiment Model class
class SentimentClassifier:
    def __init__(self, model_name="distilroberta-base", num_labels=2, learning_rate=1e-5, weight_decay=0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = torch.nn.CrossEntropyLoss()
        # To store metrics for plotting
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def train(self, train_loader, val_loader, num_epochs=2):
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            total = 0
            correct = 0
            for batch in train_loader:
                self.optimizer.zero_grad()
                input_ids, attention_mask, labels = batch['input_ids'].to(self.device), batch['attention_mask'].to(
                    self.device), batch['label'].to(self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask)
                loss = self.criterion(outputs.logits, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                # Calculate training accuracy
                predictions = torch.argmax(outputs.logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

            avg_train_loss = total_loss / len(train_loader)
            train_accuracy = correct / total

            # Evaluate on validation set
            val_loss, val_accuracy = self.evaluate(val_loader, 'Validation')

            # Store metrics for plotting
            self.train_losses.append(avg_train_loss)
            self.train_accuracies.append(train_accuracy)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)
            print(f"Epoch {epoch + 1}, Training Loss: {avg_train_loss:.4f}")

    def evaluate(self, data_loader, data_type):
        self.model.eval()
        correct, total = 0, 0
        total_loss = 0
        with torch.no_grad():
            for batch in data_loader:
                input_ids, attention_mask, labels = batch['input_ids'].to(self.device), batch['attention_mask'].to(
                    self.device), batch['label'].to(self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=1)
                loss = self.criterion(outputs.logits, labels)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                total_loss += loss.item()
        loss = total_loss / total
        accuracy = correct / total
        print(f"{data_type} Accuracy: {accuracy:.2%}")
        print(f"{data_type} Loss: {loss:.4f}")

        return loss, accuracy

    def predict(self, inputs):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(inputs['input_ids'].to(self.device),
                                 attention_mask=inputs['attention_mask'].to(self.device))
            predictions = torch.argmax(outputs.logits, dim=1).item()
        return predictions

    def plot_metrics(self):
        # Plot Loss
        plt.figure(figsize=(12, 6))
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss over Epochs")
        plt.legend()
        plt.show()

        # Plot Accuracy
        plt.figure(figsize=(12, 6))
        plt.plot(self.train_accuracies, label="Train Accuracy")
        plt.plot(self.val_accuracies, label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy over Epochs")
        plt.legend()
        plt.show()


# Initialize and train the model
classifier = SentimentClassifier()
classifier.train(train_loader, val_loader)
classifier.evaluate(val_loader, 'Validation')
classifier.evaluate(test_loader, 'Test')



def get_predictions_for_data(model, data_iter, device):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in data_iter:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Extract logits from SequenceClassifierOutput
            logits = outputs.logits

            # Convert logits to predictions
            predictions = torch.argmax(logits, dim=1)

            all_predictions.append(predictions)
            all_labels.append(labels)

    return torch.cat(all_predictions), torch.cat(all_labels)

def evaluate_special_subsets(model, dataset, test_iterator, device):
    """
    Evaluate the model on specific subsets of the test data: negated polarity and rare words.
    :param model: Trained model
    :param data_manager: DataManager instance
    :param device: Device for computation (e.g., 'cpu' or 'cuda')
    """
    test_sentences = dataset.get_test_set()
    # Get subset indices
    negated_polarity_indices = torch.tensor(data_loader.get_negated_polarity_examples(test_sentences), dtype=torch.long)
    rare_words_indices = torch.tensor(data_loader.get_rare_words_examples(test_sentences, dataset), dtype=torch.long)
    print(get_predictions_for_data(model, test_iterator))
    # Get predictions and labels for the test set
    predictions, labels = get_predictions_for_data(model, test_iterator)

    predictions = torch.tensor(predictions, dtype=torch.float32).to(device)
    labels = torch.tensor(labels, dtype=torch.float32).to(device)
    # Filter predictions and labels for each subset
    negated_polarity_preds = predictions[negated_polarity_indices]
    negated_polarity_labels = labels[negated_polarity_indices]
    rare_words_preds = predictions[rare_words_indices]
    rare_words_labels = labels[rare_words_indices]

    # Calculate accuracies for special subsets
    negated_polarity_accuracy = exercise_blanks.special_binary_accuracy(negated_polarity_preds,
                                                                        negated_polarity_labels).item()
    rare_words_accuracy = exercise_blanks.special_binary_accuracy(rare_words_preds, rare_words_labels).item()
    print(f"Accuracy on Negated Polarity Subset: {negated_polarity_accuracy:.4f}")
    print(f"Accuracy on Rare Words Subset: {rare_words_accuracy:.4f}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_special_subsets(model, dataset, test_iterator, device):
    """
    Evaluate the model on specific subsets of the test data: negated polarity and rare words.
    :param model: Trained model
    :param data_manager: DataManager instance
    :param device: Device for computation (e.g., 'cpu' or 'cuda')
    """
    test_sentences = dataset.get_test_set()
    # Get subset indices
    negated_polarity_indices = torch.tensor(data_loader.get_negated_polarity_examples(test_sentences), dtype=torch.long)
    rare_words_indices = torch.tensor(data_loader.get_rare_words_examples(test_sentences, dataset), dtype=torch.long)

    # Get predictions and labels for the test set
    predictions, labels = get_predictions_for_data(model, test_iterator, device)

    predictions = torch.tensor(predictions, dtype=torch.float32).to(device)
    labels = torch.tensor(labels, dtype=torch.float32).to(device)
    # Filter predictions and labels for each subset
    negated_polarity_preds = predictions[negated_polarity_indices]
    negated_polarity_labels = labels[negated_polarity_indices]
    rare_words_preds = predictions[rare_words_indices]
    rare_words_labels = labels[rare_words_indices]

    # Calculate accuracies for special subsets
    negated_polarity_accuracy = exercise_blanks.special_binary_accuracy(negated_polarity_preds, negated_polarity_labels).item()
    rare_words_accuracy = exercise_blanks.special_binary_accuracy(rare_words_preds, rare_words_labels).item()
    print(f"Accuracy on Negated Polarity Subset: {negated_polarity_accuracy:.4f}")
    print(f"Accuracy on Rare Words Subset: {rare_words_accuracy:.4f}")

classifier.plot_metrics()
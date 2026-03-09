import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset, Dataset
import operator
import data_loader
import pickle
import tqdm

# ------------------------------------------- Constants ----------------------------------------

SEQ_LEN = 52
W2V_EMBEDDING_DIM = 300

ONEHOT_AVERAGE = "onehot_average"
W2V_AVERAGE = "w2v_average"
W2V_SEQUENCE = "w2v_sequence"

TRAIN = "train"
VAL = "val"
TEST = "test"


# ------------------------------------------ Helper methods and classes --------------------------

def get_available_device():
    """
    Allows training on GPU if available. Can help with running things faster when a GPU with cuda is
    available but not a most...
    Given a device, one can use module.to(device)
    and criterion.to(device) so that all the computations will be done on the GPU.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_model(model, path, epoch, optimizer):
    """
    Utility function for saving checkpoint of a model, so training or evaluation can be executed later on.
    :param model: torch module representing the model
    :param optimizer: torch optimizer used for training the module
    :param path: path to save the checkpoint into
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()}, path)


def load(model, path, optimizer):
    """
    Loads the state (weights, paramters...) of a model which was saved with save_model
    :param model: should be the same model as the one which was saved in the path
    :param path: path to the saved checkpoint
    :param optimizer: should be the same optimizer as the one which was saved in the path
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch


# ------------------------------------------ Data utilities ----------------------------------------

def load_word2vec():
    """ Load Word2Vec Vectors
        Return:
            wv_from_bin: All 3 million embeddings, each lengh 300
    """
    import gensim.downloader as api
    wv_from_bin = api.load("word2vec-google-news-300")
    vocab = list(wv_from_bin.key_to_index.keys())
    # print(wv_from_bin.key_to_index[vocab[0]])
    # print("Loaded vocab size %i" % len(vocab))
    return wv_from_bin


def create_or_load_slim_w2v(words_list, cache_w2v=True):
    """
    returns word2vec dict only for words which appear in the dataset.
    :param words_list: list of words to use for the w2v dict
    :param cache_w2v: whether to save locally the small w2v dictionary
    :return: dictionary which maps the known words to their vectors
    """
    w2v_path = "w2v_dict.pkl"
    if not os.path.exists(w2v_path):
        full_w2v = load_word2vec()
        w2v_emb_dict = {k: full_w2v[k] for k in words_list if k in full_w2v}
        if cache_w2v:
            save_pickle(w2v_emb_dict, w2v_path)
    else:
        w2v_emb_dict = load_pickle(w2v_path)
    return w2v_emb_dict


def get_w2v_average(sent, word_to_vec, embedding_dim=300):
    """
    This method gets a sentence and returns the average word embedding of the words consisting
    the sentence.
    :param sent: the sentence object
    :param word_to_vec: a dictionary mapping words to their vector embeddings
    :param embedding_dim: the dimension of the word embedding vectors
    :return The average embedding vector as numpy ndarray.
    """
    word2vec_embeddings = []
    known = 0

    for word in sent.text:
        if word in word_to_vec:
            word2vec_embeddings.append(word_to_vec[word])
            known += 1
        else:
            word2vec_embeddings.append(np.zeros(embedding_dim))

    if known > 0:
        # Sum the vectors and divide by the count of known words
        avg_vector = np.sum(word2vec_embeddings, axis=0) / known
    else:
        # If no known words, return a zero vector
        avg_vector = np.zeros(embedding_dim)

    return avg_vector


#6.3.a
def get_one_hot(size, ind):
    """
    this method returns a one-hot vector of the given size, where the 1 is placed in the ind entry.
    :param size: the size of the vector
    :param ind: the entry index to turn to 1
    :return: numpy ndarray which represents the one-hot vector
    """
    one_hot = np.zeros(size)
    one_hot[ind] = 1
    return one_hot



#6.3.b
def average_one_hots(sent, word_to_ind):
    """
    this method gets a sentence, and a mapping between words to indices, and returns the average
    one-hot embedding of the tokens in the sentence.
    :param sent: a sentence object.
    :param word_to_ind: a mapping between words to indices
    :return:
    """
    size = len(word_to_ind)
    avg = np.zeros(size)
    text = sent.text
    for word in text:
        avg[word_to_ind[word]] += 1
    return avg / len(text)



#6.2
def get_word_to_ind(words_list):
    """
    this function gets a list of words, and returns a mapping between
    words to their index.
    :param words_list: a list of words
    :return: the dictionary mapping words to the index
    """
    return {word: idx for idx, word in enumerate(words_list)}



def sentence_to_embedding(sent, word_to_vec, seq_len, embedding_dim=300):
    """
    this method gets a sentence and a word to vector mapping, and returns a list containing the
    words embeddings of the tokens in the sentence.
    :param sent: a sentence object
    :param word_to_vec: a word to vector mapping.
    :param seq_len: the fixed length for which the sentence will be mapped to.
    :param embedding_dim: the dimension of the w2v embedding
    :return: numpy ndarray of shape (seq_len, embedding_dim) with the representation of the sentence
    """
    # Initialize a zero matrix for the sentence embedding
    sentence_embedding = np.zeros((seq_len, embedding_dim))

    # Iterate through the words in the sentence
    for i, word in enumerate(sent.text[:seq_len]):  # Consider only the first `seq_len` words
        if word in word_to_vec:  # If the word is in the vocabulary, use its embedding
            sentence_embedding[i] = word_to_vec[word]

    return sentence_embedding




class OnlineDataset(Dataset):
    """
    A pytorch dataset which generates model inputs on the fly from sentences of SentimentTreeBank
    """

    def __init__(self, sent_data, sent_func, sent_func_kwargs):
        """
        :param sent_data: list of sentences from SentimentTreeBank
        :param sent_func: Function which converts a sentence to an input datapoint
        :param sent_func_kwargs: fixed keyword arguments for the state_func
        """
        self.data = sent_data
        self.sent_func = sent_func
        self.sent_func_kwargs = sent_func_kwargs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent = self.data[idx]
        sent_emb = self.sent_func(sent, **self.sent_func_kwargs)
        sent_label = sent.sentiment_class
        return sent_emb, sent_label


class DataManager():
    """
    Utility class for handling all data management task. Can be used to get iterators for training and
    evaluation.
    """

    def __init__(self, data_type=ONEHOT_AVERAGE, use_sub_phrases=True, dataset_path="stanfordSentimentTreebank", batch_size=50,
                 embedding_dim=None):
        """
        builds the data manager used for training and evaluation.
        :param data_type: one of ONEHOT_AVERAGE, W2V_AVERAGE and W2V_SEQUENCE
        :param use_sub_phrases: if true, training data will include all sub-phrases plus the full sentences
        :param dataset_path: path to the dataset directory
        :param batch_size: number of examples per batch
        :param embedding_dim: relevant only for the W2V data types.
        """

        # load the dataset
        self.sentiment_dataset = data_loader.SentimentTreeBank(dataset_path, split_words=True)
        # map data splits to sentences lists
        self.sentences = {}
        if use_sub_phrases:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set_phrases()
        else:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set()

        self.sentences[VAL] = self.sentiment_dataset.get_validation_set()
        self.sentences[TEST] = self.sentiment_dataset.get_test_set()

        # map data splits to sentence input preperation functions
        words_list = list(self.sentiment_dataset.get_word_counts().keys())
        if data_type == ONEHOT_AVERAGE:
            self.sent_func = average_one_hots
            self.sent_func_kwargs = {"word_to_ind": get_word_to_ind(words_list)}
        elif data_type == W2V_SEQUENCE:
            self.sent_func = sentence_to_embedding

            self.sent_func_kwargs = {"seq_len": SEQ_LEN,
                                     "word_to_vec": create_or_load_slim_w2v(words_list),
                                     "embedding_dim": embedding_dim
                                     }
        elif data_type == W2V_AVERAGE:
            self.sent_func = get_w2v_average
            words_list = list(self.sentiment_dataset.get_word_counts().keys())
            self.sent_func_kwargs = {"word_to_vec": create_or_load_slim_w2v(words_list),
                                     "embedding_dim": embedding_dim
                                     }
        else:
            raise ValueError("invalid data_type: {}".format(data_type))
        # map data splits to torch datasets and iterators
        self.torch_datasets = {k: OnlineDataset(sentences, self.sent_func, self.sent_func_kwargs) for
                               k, sentences in self.sentences.items()}
        self.torch_iterators = {k: DataLoader(dataset, batch_size=batch_size, shuffle=k == TRAIN)
                                for k, dataset in self.torch_datasets.items()}

    def get_torch_iterator(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: torch batches iterator for this part of the datset
        """
        return self.torch_iterators[data_subset]

    def get_labels(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: numpy array with the labels of the requested part of the datset in the same order of the
        examples.
        """
        return np.array([sent.sentiment_class for sent in self.sentences[data_subset]])

    def get_input_shape(self):
        """
        :return: the shape of a single example from this dataset (only of x, ignoring y the label).
        """
        return self.torch_datasets[TRAIN][0][0].shape




# ------------------------------------ Models ----------------------------------------------------

class LSTM(nn.Module):
    """
An LSTM for sentiment analysis with architecture as described in the exercise description.
"""

    def __init__(self, embedding_dim, hidden_dim, dropout):
        """
        Initialize the bi-directional LSTM model.
        :param embedding_dim: Dimension of input embeddings (e.g., Word2Vec).
        :param hidden_dim: Dimension of the LSTM hidden state.
        :param dropout: Dropout rate for regularization.
        """
        super(LSTM, self).__init__()

        # Bidirectional LSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        # Fully connected layer
        # Input size is 2 * hidden_dim because of bi-directionality
        self.fc = nn.Linear(2 * hidden_dim, 1)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Sigmoid activation for binary classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, text):
        """
        Forward pass through the LSTM.
        :param text: Input tensor of shape (batch_size, seq_len, embedding_dim).
        :return: Logits for sentiment predictions.
        """
        # Pass input through the LSTM
        lstm_out, (hidden, cell) = self.lstm(text)

        # Extract the final forward and backward hidden states
        # hidden[-2] -> Forward direction
        # hidden[-1] -> Backward direction
        final_hidden_state = torch.cat((hidden[-2], hidden[-1]), dim=1)

        # Apply dropout
        final_hidden_state = self.dropout(final_hidden_state)

        # Pass through the fully connected layer
        logits = self.fc(final_hidden_state)
        return logits

    def predict(self, text):
        """
        Predict probabilities for the input text.
        :param text: Input tensor of shape (batch_size, seq_len, embedding_dim).
        :return: Probabilities for each sample in the batch.
        """
        logits = self.forward(text)
        probabilities = self.sigmoid(logits)
        return probabilities



class LogLinear(nn.Module):
    """
    general class for the log-linear models for sentiment analysis.
    """
    def __init__(self, embedding_dim):
        super(LogLinear, self).__init__()
        self.linear = nn.Linear(embedding_dim, 1)


    def forward(self, x):
        return self.linear(x)


    def predict(self, x):
        x = x.to(next(self.parameters()).device, dtype=torch.float32)  # Ensure correct dtype and device
        logits = self.forward(x)
        return torch.sigmoid(logits)



# ------------------------- training functions -------------

#5.1
def special_binary_accuracy(preds, y):
    """
    This method returns tha accuracy of the predictions, relative to the labels.
    You can choose whether to use numpy arrays or tensors here.
    :param preds: a vector of predictions
    :param y: a vector of true labels
    :return: scalar value - (<number of accurate predictions> / <number of examples>)
    """
    rounded_preds = torch.round(preds)  # Round probabilities to {0, 1}
    correct = (rounded_preds.int() == y.int())
    accuracy = correct.sum().float() / len(y)  # Ensure accuracy is a float
    return accuracy


def binary_accuracy_for_models(preds, y):
    """
    This method returns tha accuracy of the predictions, relative to the labels.
    You can choose whether to use numpy arrays or tensors here.
    :param preds: a vector of predictions
    :param y: a vector of true labels
    :return: scalar value - (<number of accurate predictions> / <number of examples>)
    """
    preds = torch.sigmoid(preds)
    rounded_preds = torch.round(preds)
    correct = (rounded_preds.int() == y.int())
    accuracy = correct.sum() / len(y)
    # if 'evaluate' in string:
        # print(rounded_preds)
        # print(y)
        # print(string + f"accuracy over single batch in binary accuracy function {accuracy:.2f}")
    return accuracy

def train_epoch(model, data_iterator, optimizer, criterion, device):
    """
    Train the model for one epoch.
    :param model: The PyTorch model to train.
    :param data_iterator: DataLoader for the training data.
    :param optimizer: Optimizer for the model.
    :param criterion: Loss function.
    :param device: Device to use for training (e.g., 'cpu' or 'cuda').
    :return: Average loss and accuracy for the epoch.
    """
    model.train()
    epoch_loss = 0
    epoch_acc = 0

    for batch in data_iterator:
        inputs, labels = batch

        # Move tensors to the appropriate device and ensure consistent dtype
        inputs, labels = inputs.to(device, dtype=torch.float32), labels.to(device, dtype=torch.float32)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        predictions = model(inputs).squeeze()

        # Compute loss
        loss = criterion(predictions, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Compute accuracy
        acc = binary_accuracy_for_models(predictions, labels)

        # Accumulate loss and accuracy
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(data_iterator), epoch_acc / len(data_iterator)

def evaluate(model, data_iterator, criterion, device):
    """
    Evaluate the model on the validation or test set.
    :param model: The PyTorch model to evaluate.
    :param data_iterator: DataLoader for the validation/test data.
    :param criterion: Loss function.
    :param device: Device to use for evaluation (e.g., 'cpu' or 'cuda').
    :return: Average loss and accuracy for the evaluation.
    """
    model.eval()
    epoch_loss = 0
    epoch_acc = 0

    with torch.no_grad():
        for batch in data_iterator:
            inputs, labels = batch

            # Move tensors to the appropriate device and ensure consistent dtype
            inputs, labels = inputs.to(device, dtype=torch.float32), labels.to(device, dtype=torch.float32)

            # Forward pass
            predictions = model(inputs).squeeze()

            # Compute loss
            loss = criterion(predictions, labels)

            # Compute accuracy
            acc = binary_accuracy_for_models(predictions, labels)

            # Accumulate loss and accuracy
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(data_iterator), epoch_acc / len(data_iterator)

def train_model(model, data_manager, n_epochs, lr, weight_decay, device):
    """
    Train the model for a specified number of epochs and evaluate on validation set.
    :param model: The PyTorch model to train.
    :param data_manager: DataManager instance for handling data.
    :param n_epochs: Number of epochs to train.
    :param lr: Learning rate for the optimizer.
    :param weight_decay: Weight decay for regularization.
    :param device: Device to use for training and evaluation (e.g., 'cpu' or 'cuda').
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()

    train_iter = data_manager.get_torch_iterator()
    val_iter = data_manager.get_torch_iterator(VAL)

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    for epoch in range(n_epochs):
        # Train for one epoch
        epoch_loss, epoch_acc = train_epoch(model, train_iter, optimizer, criterion,device)
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        # Evaluate on validation set
        val_loss, val_acc = evaluate(model, val_iter, criterion,device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

    return train_losses, train_accuracies, val_losses, val_accuracies


#6.4.d
def train_log_linear_with_one_hot():
    print("[One-Hot] Initializing model...")
    data_manager = DataManager(batch_size=64)
    model = LogLinear(data_manager.get_input_shape()[0]).to(device=get_available_device())

    print("[One-Hot] Training...")
    n_epochs = 20
    lr = 0.01
    weight_decay = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_losses, train_accuracies, val_losses, val_accuracies = train_model(
        model, data_manager, n_epochs, lr, weight_decay, device
    )

    print("[One-Hot] Plotting results...")
    plot_training_results(train_losses, val_losses, "Loss", "One-Hot")
    plot_training_results(train_accuracies, val_accuracies, "Accuracy", "One-Hot")

    print("[One-Hot] Testing...")
    test_iter = data_manager.get_torch_iterator(TEST)
    test_loss, test_acc = evaluate(model, test_iter, torch.nn.BCEWithLogitsLoss(), device)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Validation Accuracy (Last Epoch): {val_accuracies[-1]:.4f}")  # Log final validation accuracy
    print("[One-Hot] Testing on Special Subsets...")
    evaluate_special_subsets(model, data_manager, device)





def get_predictions_for_data(model, data_iter):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in data_iter:
            inputs, labels = batch
            inputs = inputs.to(get_available_device(), dtype=torch.float32)
            labels = labels.to(get_available_device(), dtype=torch.float32)
            preds = model.predict(inputs).squeeze()
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_predictions), np.array(all_labels)

    # tensors = []
    # for batch_X, batch_y in data_iter:
    #     tensors.append(model.predict(batch_X))
    # return torch.cat(tensors)


def evaluate_special_subsets(model, data_manager, device):
    """
    Evaluate the model on specific subsets of the test data: negated polarity and rare words.
    :param model: Trained model
    :param data_manager: DataManager instance
    :param device: Device for computation (e.g., 'cpu' or 'cuda')
    """
    dataset = data_loader.SentimentTreeBank()
    test_iter = data_manager.get_torch_iterator(data_subset=TEST)
    test_sentences = dataset.get_test_set()

    # Get subset indices
    negated_polarity_indices = torch.tensor(data_loader.get_negated_polarity_examples(test_sentences), dtype=torch.long)
    rare_words_indices = torch.tensor(data_loader.get_rare_words_examples(test_sentences, dataset), dtype=torch.long)
    # print(f"negated_polarity_indices for current model: {negated_polarity_indices[:10]}")
    # print(f"rare_words_indices for current model: {rare_words_indices[:10]}")


    # Get predictions and labels for the test set
    predictions, labels = get_predictions_for_data(model, test_iter)
    # print(f"Predictions for current model: {predictions[:10]}")
    # print(f"labels for current model: {labels[:10]}")


    predictions = torch.tensor(predictions, dtype=torch.float32).to(device)
    labels = torch.tensor(labels, dtype=torch.float32).to(device)

    # Filter predictions and labels for each subset
    negated_polarity_preds = predictions[negated_polarity_indices]
    negated_polarity_labels = labels[negated_polarity_indices]


    rare_words_preds = predictions[rare_words_indices]
    rare_words_labels = labels[rare_words_indices]
    # print(f"rare_words_preds for current model: {rare_words_preds[:10]}")
    # print(f"rare_words_labels for current model: {rare_words_labels[:10]}")

    # Calculate accuracies for special subsets
    negated_polarity_accuracy = special_binary_accuracy(negated_polarity_preds, negated_polarity_labels).item()
    rare_words_accuracy = special_binary_accuracy(rare_words_preds, rare_words_labels).item()

    print(f"Accuracy on Negated Polarity Subset: {negated_polarity_accuracy:.4f}")
    print(f"Accuracy on Rare Words Subset: {rare_words_accuracy:.4f}")



def plot_training_results(train_values, val_values, metric_name, model_name):
    """
    Plot training and validation results (loss or accuracy).
    :param train_values: List of training values (loss/accuracy).
    :param val_values: List of validation values (loss/accuracy).
    :param metric_name: Name of the metric (e.g., "Loss" or "Accuracy").
    :param model_name: Name of the model to include in the plot title.
    """
    import matplotlib.pyplot as plt

    epochs = range(1, len(train_values) + 1)
    plt.plot(epochs, train_values, label=f"Train {metric_name}")
    plt.plot(epochs, val_values, label=f"Validation {metric_name}")
    plt.title(f"{model_name} {metric_name} Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel(metric_name)
    plt.legend()
    plt.show()



def train_log_linear_with_w2v():
    """
    Train and evaluate the log-linear model with word embeddings.
    """
    print("[Word2Vec] Initializing model...")
    # Data preparation
    data_manager = DataManager(data_type=W2V_AVERAGE, batch_size=64, embedding_dim=300)
    model = LogLinear(300).to(device=get_available_device())

    # Hyperparameters
    n_epochs = 20
    lr = 0.01
    weight_decay = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("[Word2Vec] Training...")
    # Train the model and record losses/accuracies
    train_losses, train_accuracies, val_losses, val_accuracies = train_model(
        model, data_manager, n_epochs, lr, weight_decay, device
    )

    print("[Word2Vec] Plotting results...")
    # Plot training/validation results
    plot_training_results(train_losses, val_losses, "Loss", "Word2Vec")
    plot_training_results(train_accuracies, val_accuracies, "Accuracy", "Word2Vec")

    print("[Word2Vec] Testing...")
    # Evaluate on the test set
    test_iter = data_manager.get_torch_iterator(TEST)
    test_loss, test_acc = evaluate(model, test_iter, torch.nn.BCEWithLogitsLoss(), device)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Validation Accuracy (Last Epoch): {val_accuracies[-1]:.4f}")  # Log final validation accuracy

    print("[Word2Vec] Testing on Special Subsets...")
    evaluate_special_subsets(model, data_manager, device)








def train_lstm_with_w2v():
    """
    Train and evaluate the LSTM model with Word2Vec embeddings.
    """

    print("[LSTM] Initializing model...")
    # Data preparation
    data_manager = DataManager(data_type=W2V_SEQUENCE, batch_size=64, embedding_dim=300)  # Prepare data
    model = LSTM(embedding_dim=300, hidden_dim=100, dropout=0.5).to(device=get_available_device())  # Instantiate LSTM model

    n_epochs = 4
    lr = 0.001
    weight_decay = 0.0001

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Train the model and record losses/accuracies
    train_losses, train_accuracies, val_losses, val_accuracies = train_model(
       model, data_manager, n_epochs, lr, weight_decay , device
    )

    # Plot training/validation results
    plot_training_results(train_losses, val_losses, "Loss","LSTM")
    plot_training_results(train_accuracies, val_accuracies, "Accuracy","LSTM")

    # Evaluate on the test set
    test_iter = data_manager.get_torch_iterator(data_subset=TEST)
    test_loss, test_acc = evaluate(model, test_iter, torch.nn.BCEWithLogitsLoss(),device)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Validation Accuracy (Last Epoch): {val_accuracies[-1]:.4f}")  # Log final validation accuracy
    print("[Word2Vec] Testing on Special Subsets...")
    evaluate_special_subsets(model, data_manager, device)




if __name__ == '__main__':
    print("[One-Hot] started...")
    train_log_linear_with_one_hot()
    print("[One-Hot] finished...\n")

    print("[Word2Vec] started...")
    train_log_linear_with_w2v()
    print("[Word2Vec] finished...\n")

    print("[LSTM] started...")
    train_lstm_with_w2v()
    print("[LSTM] finished...\n")

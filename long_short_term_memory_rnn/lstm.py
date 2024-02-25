import matplotlib.pyplot as plt # Graph plotting
import numpy as np
import time
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import random
import csv

# Reproducability
torch.manual_seed(294124015)
np.random.seed(734787823)
random.seed(647878234)

# Constants
PLOT_SHOW = False
PLOT_SAVE = True
WRITE_TO_FILE = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -- PREPROCESSING 
tokenizer = get_tokenizer("basic_english")

TEXT_PIPELINE = None
LABEL_PIPELINE = None
N_CLASSES = 4

# -- TRAINING HYPERPARAMETERS
TRAINING_PARTITION_RATE = 0.8
LEARNING_RATE = 1e-4
LOSS_FUNCTION = torch.nn.CrossEntropyLoss()
OPTIMIZER = torch.optim.Adam
TRAINING_EPOCHS = 80
BATCH_SIZE = 100
SHUFFLE = False

class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        tag_scores = self.fc(embedded) 
        tag_scores = F.softmax(tag_scores, dim=1)
        return tag_scores

class LSTMTextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_class, lstm_layers):
        super(LSTMTextClassificationModel, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, num_layers = lstm_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        batch_size = embedded.size()[0]
        # LSTM Input Shape (with batch_first=True): Batch, Sequence, Feature
        hidden, _ = self.lstm(embedded.view(batch_size, -1, self.embed_dim))
        # FC Input Shape: Batch, Sequence
        tag_space = self.fc(hidden.view(batch_size, self.hidden_dim))
        tag_scores = F.softmax(tag_space, dim=1)
        return tag_scores

def train(model, dataloader_train, optimizer_fn=OPTIMIZER, criterion=LOSS_FUNCTION, epoch_id=TRAINING_EPOCHS):
    optimizer = optimizer_fn(params=model.parameters(), lr=LEARNING_RATE)
    model.train()
    total_acc, total_loss, total_count = 0, 0, 0
    log_interval = 500

    for idx, (label, text, offsets) in enumerate(dataloader_train):
        optimizer.zero_grad()
        predicted_label = model(text, offsets)
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_loss += loss
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches "
                "| accuracy {:8.3f}".format(
                    epoch_id, idx, len(dataloader_train), total_acc / total_count
                )
            )
            # total_acc, total_count = 0, 0
    return model, total_acc/total_count, total_loss/total_count

def evaluate(model, dataloader, criterion=LOSS_FUNCTION):
    model.eval()
    total_acc, total_loss, total_count = 0, 0, 0

    # Confusion Matrix, rows: truth, columns: prediction
    confusion_mat = torch.zeros(N_CLASSES,N_CLASSES).to(DEVICE)
    label_count = torch.zeros(N_CLASSES)

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets)
            
            total_loss += criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
            for pred, truth in zip(predicted_label, label):
                confusion_mat[truth] += pred
                label_count[truth] += 1
            
        # Normalize matrix along rows
        for i in range(N_CLASSES):
            confusion_mat[i] /= label_count[i]

    return total_acc / total_count, total_loss / total_count, confusion_mat

def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for sample in batch:
        _label = sample[0]
        _text = sample[1:]
        entire_string = ""
        for word in _text:
            entire_string += word
        label_list.append(LABEL_PIPELINE(_label))
        processed_text = torch.tensor(TEXT_PIPELINE(entire_string), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(DEVICE), text_list.to(DEVICE), offsets.to(DEVICE)

def yield_tokens(data_iter):
    for text in data_iter:
        entire_string = ""
        text = text[1:]
        for word in text:
            entire_string += word
        yield tokenizer(entire_string)

def main():
    if not PLOT_SHOW:
        print("No figures will be shown. Set PLOT_SHOW to True to enable this function.")
    
    if not PLOT_SAVE:
        print("No figures will be saved to the Figures folder. Set PLOT_SAVE to True to enable this function.")

    if not WRITE_TO_FILE:
        print("No result scores will be written to the results file. Set WRITE_TO_FILE to True to enable this function.")

    with open('data3forEx11/train.csv') as train_data, \
        open('data3forEx11/test.csv') as test_data:
        train_data = csv.reader(train_data)
        test_data = csv.reader(test_data)

        train_dataset = []
        test_dataset = []
        classes = set()
        for row in train_data:
            classes.add(row[0])
            train_dataset.append(row)
        
        global N_CLASSES
        N_CLASSES = len(classes)

        for row in test_data:
            test_dataset.append(row)
        
        vocab = build_vocab_from_iterator(yield_tokens(train_dataset), specials=["<unk>"])
        vocab.set_default_index(vocab["<unk>"])
        global VOCAB_SIZE
        VOCAB_SIZE = len(vocab)

        global TEXT_PIPELINE
        global LABEL_PIPELINE
        TEXT_PIPELINE = lambda x: vocab(tokenizer(x))
        LABEL_PIPELINE = lambda x: int(x) - 1
        
        
        num_train = int(len(train_dataset) * TRAINING_PARTITION_RATE)
        train_split, cv_split = random_split(
            train_dataset, [num_train, len(train_dataset) - num_train]
        )

        train_dataloader = DataLoader(
           train_split, batch_size=BATCH_SIZE, shuffle=SHUFFLE, collate_fn=collate_batch
        )

        cv_dataloader = DataLoader(
           cv_split, batch_size=BATCH_SIZE, shuffle=SHUFFLE, collate_fn=collate_batch
        )

        test_dataloader = DataLoader(
           test_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE, collate_fn=collate_batch
        )
        
        baseline = {
            "EMBEDDING_DIM": 16,
            "HIDDEN_DIM": 16,
            "LSTM_LAYERS": 1,
            "NOLSTM": False
        }
        
        strongembed = {
            "EMBEDDING_DIM": 32,
            "HIDDEN_DIM": 16,
            "LSTM_LAYERS": 1,
            "NOLSTM": False
        }
        
        stronglstm = {
            "EMBEDDING_DIM": 16,
            "HIDDEN_DIM": 32,
            "LSTM_LAYERS": 1,
            "NOLSTM": False
        }
        
        multilayer = {
            "EMBEDDING_DIM": 16,
            "HIDDEN_DIM": 16,
            "LSTM_LAYERS": 3,
            "NOLSTM": False
        }
        
        withoutlstm = {
            "EMBEDDING_DIM": 16,
            "NOLSTM": True
        }
        
        minimal = {
            "EMBEDDING_DIM": 6,
            "HIDDEN_DIM": 6,
            "LSTM_LAYERS": 1,
            "NOLSTM": False
        }

        experiment_models = [baseline, strongembed, stronglstm, multilayer, withoutlstm, minimal]
        experiment_model_names = ["Baseline", "Double Embedding", "Double LSTM", "Multilayer", "No LSTM", "Minimal"]
        
        with open("results.txt", "a") as res_file:
            for combi, combi_name in zip(experiment_models, experiment_model_names):
                print("\r\nSelected hyperparameters in this iteration:")
                hyperParamHeader = f"Experiment Model: {combi_name}, Time: {time.strftime('%a %d %b %Y, %I:%M%p')}\r\nParameters: "
                fileHeader = f"{combi_name}__"
                for param_name in combi:
                    fileHeader += f"{param_name}={combi[param_name]}_"
                    hyperParamHeader += f"{param_name}: {combi[param_name]}, "
                    print(f"{param_name}: {combi[param_name]}, ")
                
                if WRITE_TO_FILE:
                    res_file.write(f"\r\n{hyperParamHeader}\r\n")
                fileHeader += f"{int(time.time())}"

                print("\r\n--- TRAINING PHASE ---")
                model = None
                if combi["NOLSTM"]:
                    model = TextClassificationModel(
                        VOCAB_SIZE, 
                        combi["EMBEDDING_DIM"], 
                        N_CLASSES).to(DEVICE)
                else:
                    model = LSTMTextClassificationModel(
                        VOCAB_SIZE, 
                        combi["EMBEDDING_DIM"],
                        combi["HIDDEN_DIM"], 
                        N_CLASSES,
                        combi["LSTM_LAYERS"], ).to(DEVICE)

                log_accuracy = []
                log_cv_accuracy = []
                log_loss = []
                log_cv_loss = []
                for epoch in range(TRAINING_EPOCHS):
                    model, avg_acc, avg_loss = train(model, train_dataloader, epoch_id=epoch+1)
                    cv_acc, cv_loss, _ = evaluate(model, cv_dataloader)
                    log_accuracy.append(avg_acc)
                    log_cv_accuracy.append(cv_acc)
                    log_loss.append(avg_loss.detach().cpu().numpy())
                    log_cv_loss.append(cv_loss.detach().cpu().numpy())
                
                    resString = f"Epoch\t{epoch+1:>2}, Average Loss:{avg_loss:2.4f}, Average Acc: {avg_acc:.4f}, CV Loss:{cv_loss:2.4f}, CV Acc: {cv_acc:.4f}"
                    if WRITE_TO_FILE:
                        res_file.write(f"{resString}\r\n")
                
                plt.plot(log_accuracy, label='Training')
                plt.plot(log_cv_accuracy, label='Cross Validation')
                plt.xlabel('Train Epoch')
                plt.ylabel('Accuracy')
                plt.legend()
                plt.title(f"Training and CV Accuracy during Training")
                if PLOT_SAVE:
                    plt.savefig(f"Figures/learning_curve_{fileHeader}.png")
                if PLOT_SHOW:
                    plt.show()
                plt.clf()

                # Test
                print("\r\n--- TEST PHASE ---")

                test_acc, test_loss, confusion_mat = evaluate(model, test_dataloader)
                
                if WRITE_TO_FILE:
                    res_file.write("\r\n-- TEST RESULTS --\r\n")
                    res_file.write(f"Test Accuracy: {test_acc:0.4f}, Test Loss: {test_loss.detach().cpu().numpy():0.4f}\r\n")
                    res_file.write("Confusion matrix (truth,prediction)\r\n")
                    for i in range(N_CLASSES):
                        for j in range(N_CLASSES):
                            res_file.write(f"{confusion_mat[i][j]:.4f}\t")
                        res_file.write("\r\n")
                
            res_file.flush()
        res_file.close()
    return 0
    

if __name__=="__main__":
    main()
import matplotlib.pyplot as plt # Graph plotting
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import numpy as np
import idx2numpy
import time
import torch, torch.nn as nn
from typing import Union
import random
import os

# Reproducability
torch.manual_seed(294124015)
np.random.seed(734787823)
random.seed(647878234)

# Constants
PLOT_SHOW = False
PLOT_SAVE = True
WRITE_TO_FILE = True
DEVICE = 'cuda:2'

# -- TRAINING HYPERPARAMETERS
PARTITION_ORDER = 10
LEARNING_RATE = 1e-4
LOSS_FUNCTION = torch.nn.CrossEntropyLoss()
TRAINING_EPOCHS = 80
BATCH_SIZE = 100

# -- CNN HYPERPARAMETERS
BIAS = True
DILATION = 1
STRIDE = 1
PADDING = 0
KERNEL_SIZE = 4
POOL_KERNEL_SIZE = 4

LOG_HYPERPARAMS = {
    "BIAS": BIAS,
    "DIL": DILATION,
    "STRIDE": STRIDE,
    "PAD": PADDING,
    "KRNLSIZE": KERNEL_SIZE,
    "POOLKRNLSIZE": POOL_KERNEL_SIZE
}

# Train the model with each combination of the experiment values: 

EXPERIMENT_VALUES = {
    "BIAS": [True],
    "DIL": [1],
    "STRIDE": [1],
    "PAD": [0],
    "KRNLSIZE": [2,4,8],
    "POOLKRNLSIZE": [4,8]
}

# Visualization
color_idxs = [
    mcolors.CSS4_COLORS['indianred'],
    mcolors.CSS4_COLORS['orange'],
    mcolors.CSS4_COLORS['goldenrod'],
    mcolors.CSS4_COLORS['palegreen'],
    mcolors.CSS4_COLORS['seagreen'],
    mcolors.CSS4_COLORS['turquoise'],
    mcolors.CSS4_COLORS['skyblue'],
    mcolors.CSS4_COLORS['steelblue'],
    mcolors.CSS4_COLORS['slateblue'],
    mcolors.CSS4_COLORS['darkorchid'],
    mcolors.CSS4_COLORS['mediumvioletred'],
    mcolors.CSS4_COLORS['deeppink'],
    mcolors.CSS4_COLORS['gold'],
    mcolors.CSS4_COLORS['greenyellow'],
    mcolors.CSS4_COLORS['navy']
]
color_gray = mcolors.CSS4_COLORS['gray']
legend_patches = [mpatches.Patch(color = color_idx, label = f"{i}") for i, color_idx in enumerate(color_idxs)]
POINT_SIZE = plt.rcParams['lines.markersize'] / 16

class CNN(nn.Module):

    def __init__(self, layer_depth=5, 
                 channel: Union[int, list] = 1, 
                 kernel_size: Union[int, list] = 1,
                 pool_kernel_size: Union[int, list] = 1,
                 padding: Union[int, list] = 0,
                 dilation: Union[int, list] = 1,
                 stride: Union[int, list] = 1,
                 input_size=84,
                 bias=True,
                 f_act=nn.ReLU(),
                 fc_layer_width: int = 625,
                 output_size: int = 10):
        super(CNN, self).__init__()

        # Input validation and processing
        if isinstance(channel,int):
            channel = [channel] * (layer_depth + 1)
        else:
            assert len(channel) == layer_depth + 1
        self.channel = channel

        if isinstance(kernel_size,int):
            kernel_size = [kernel_size] * (layer_depth)
        else:
            assert len(kernel_size) == layer_depth
        self.kernel_size = kernel_size

        if isinstance(pool_kernel_size,int):
            pool_kernel_size = [pool_kernel_size] * (layer_depth)
        else:
            assert len(pool_kernel_size) == layer_depth
        self.pool_kernel_size = pool_kernel_size

        if isinstance(padding,int):
            padding = [padding] * (layer_depth)
        else:
            assert len(padding) == layer_depth
        self.padding = padding    

        if isinstance(dilation,int):
            dilation = [dilation] * (layer_depth)
        else:
            assert len(dilation) == layer_depth
        self.dilation = dilation    

        if isinstance(stride,int):
            stride = [stride] * (layer_depth)
        else:
            assert len(stride) == layer_depth
        self.stride = stride   

        self.layer_depth = layer_depth
        self.conv_layers = list()
        cur_size = input_size

        i_l = 0
        self.repr_str = "CNN(torch.nn.Module)\r\n"
        print("\r\nCreating Convolution Layers...")
        for i in range(layer_depth):
            # Adjust the kernel parameters with a few heuristics to meet the input size.
            field_width = 1 + (kernel_size[i]-1) * dilation[i]
            overlap = (cur_size + 2*padding[i] - field_width) % stride[i]
            if overlap > 0:
                if stride - overlap % 2 == 0:
                    padding[i] += (stride - overlap) // 2
                elif overlap % dilation[i] == 0:
                    kernel_size[i] += overlap
                elif overlap % (kernel_size[i] - 1) == 0:
                    dilation[i] += overlap
                elif stride[i] % 2 == 1:
                    padding[i] += (2*stride[i] - overlap) // 2
                else:
                    pass
                field_width = 1 + (kernel_size[i]-1) * dilation[i]
            out_size = 1 + (cur_size + 2*padding[i] - field_width) // stride[i]
            desc = f"Conv Layer {i+1}. Input size:\t{channel[i]}x{cur_size}x{cur_size}, Output size:\t{channel[i+1]}x{out_size}x{out_size}, " \
                  f"Kernel size:\t{kernel_size[i]}, Stride:\t{stride[i]}, Padding:\t{padding[i]}, Dilation:\t{dilation[i]}" 
            self.repr_str += f"{desc}\r\n"
            print(desc)
            cur_size = out_size

            # For pooling, if necessary increase padding to meet input size.
            # Padding with zeros will unlikely affect max pooling.
            # If some paddings are ignored at the corner, it doesn't matter
            pool_padding = 0
            overlap = (cur_size) % pool_kernel_size[i]
            if overlap > 0:
                pool_padding += (overlap+1) // 2 
            out_size = (cur_size + 2*pool_padding) // pool_kernel_size[i]
            i_l += 1
            layer = nn.Sequential(
                nn.Conv2d(channel[i], channel[i+1], kernel_size[i],
                          stride=stride[i], padding=padding[i], dilation=dilation[i], bias=bias),
                f_act,
                nn.MaxPool2d(pool_kernel_size[i], padding=pool_padding)
            )
            setattr(self, f"layer{i_l}", layer)
            self.conv_layers.append(layer)
            
            desc = f"Pool Layer {i+1}. Input size:\t{channel[i+1]}x{cur_size}x{cur_size}, Output size:\t{channel[i+1]}x{out_size}x{out_size}, " \
                  f"Kernel size:\t{pool_kernel_size[i]}, Padding:\t{pool_padding}"
            self.repr_str += f"{desc}\r\n"
            print(desc)

            cur_size = out_size

        print("\r\nCreating Fully Connected Layers...")
        # Fully Connected Final Layer
        self.fc_layers = list()
        input_widths = [
            cur_size * cur_size * channel[layer_depth],
            fc_layer_width,
            output_size
        ]
        desc = f"Final FC Layers. Input:{channel[layer_depth]}x{cur_size}x{cur_size}\t, Intermediate:{input_widths[1]}\t, Output:{input_widths[2]}\t, Bias:{bias}"
        self.repr_str += f"{desc}\r\n"
        print(desc)

        for i in range(len(input_widths)-1):
            i_l += 1
            fc = nn.Linear(input_widths[i], input_widths[i+1], bias=bias)
            nn.init.xavier_uniform_(fc.weight)
            layer = nn.Sequential(
                fc,
                f_act
            )
            setattr(self, f"layer{i_l}", layer)
            self.fc_layers.append(layer)

            self.softmax = nn.Softmax(dim=1)
    
    def __repr__(self):
        return self.repr_str

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = torch.flatten(x, start_dim=1)
        for layer in self.fc_layers:
            x = layer(x)
        x = self.softmax(x)
        return x

def train(model, data, labels, crossval_data, crossval_labels, res_file, loss_fn=LOSS_FUNCTION, epochs=TRAINING_EPOCHS, batch_size=BATCH_SIZE):
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    
    # Data is Torch Tensor
    assert isinstance(data, torch.Tensor)
    assert isinstance(crossval_data, torch.Tensor)
    
    model.to(DEVICE)
    if WRITE_TO_FILE:
        res_file.write("\r\n-- TRAINING PROGRESS --\r\n")
    batch_num = data.size()[0] // batch_size
    Y = None
    pred = None
    log_accuracy = []
    log_cv_accuracy = []
    for epoch in range(epochs):
        avg_loss = 0.0
        avg_acc = 0.0
        for i_b in range(batch_num):
            optimizer.zero_grad()
            X = data[batch_size * i_b : batch_size * i_b + batch_size]
            Y = labels[batch_size * i_b : batch_size * i_b + batch_size]
            pred = model(X)
            loss = loss_fn(pred, Y)
            loss.backward()
            optimizer.step()

            accuracy = torch.mul(pred, Y).sum(dim=1).mean()
            avg_loss += loss.item() / batch_num
            avg_acc += accuracy / batch_num
            # print(f"Epoch\t{epoch+1:>2}, Batch: \t{i_b+1:>3}, Loss:{loss.item():2.4f}, Accuracy: {accuracy:.4f}")
            

        cv_pred = model(crossval_data)
        cv_acc = torch.mul(cv_pred, crossval_labels).sum(dim=1).mean()
        log_accuracy.append(avg_acc.detach().cpu().numpy())
        log_cv_accuracy.append(cv_acc.detach().cpu().numpy())
    
        resString = f"Epoch\t{epoch+1:>2}, Average Loss:{avg_loss:2.4f}, Average Acc: {avg_acc:.4f}, CV Acc: {cv_acc:.4f}"
        if WRITE_TO_FILE:
            res_file.write(f"{resString}\r\n")

        # print(resString)
    
    return log_accuracy, log_cv_accuracy

def main():
    if not PLOT_SHOW:
        print("No figures will be shown. Set PLOT_SHOW to True to enable this function.")
    
    if not PLOT_SAVE:
        print("No figures will be saved to the Figures folder. Set PLOT_SAVE to True to enable this function.")

    if not WRITE_TO_FILE:
        print("No result scores will be written to the results file. Set WRITE_TO_FILE to True to enable this function.")

    images = idx2numpy.convert_from_file("data2forEx8+/train-images.idx3-ubyte")
    labels = idx2numpy.convert_from_file("data2forEx8+/train-labels.idx1-ubyte")

    N_IMAGES = images.shape[0]
    image_size = images.shape[1]

    labels_onehot = np.zeros((N_IMAGES,10))
    for i, label in enumerate(labels):
        labels_onehot[i][int(label)] = 1

    baseline = {
        "LAYER_DEPTH": 4,
        "CHANNEL": [1, 4, 16, 64, 128],
        "KRNLSIZE": 3,
        "POOLKRNLSIZE": 2,
        "PAD": 1,
        "DIL": 1,
        "STRIDE": 1,
        "BIAS": True,
        "ACT":nn.ReLU(),
        "FCLAYERWIDTH":128
    }

    strongerfc = {
        "LAYER_DEPTH": 4,
        "CHANNEL": [1, 4, 16, 64, 128],
        "KRNLSIZE": 3,
        "POOLKRNLSIZE": 2,
        "PAD": 1,
        "DIL": 1,
        "STRIDE": 1,
        "BIAS": True,
        "ACT":nn.ReLU(),
        "FCLAYERWIDTH":256
    }

    dilated = {
        "LAYER_DEPTH": 3,
        "CHANNEL": [1, 4, 16, 64],
        "KRNLSIZE": 3,
        "POOLKRNLSIZE": 2,
        "PAD": 1,
        "DIL": 2,
        "STRIDE": 1,
        "BIAS": True,
        "ACT":nn.ReLU(),
        "FCLAYERWIDTH":128
    }

    elu = {
        "LAYER_DEPTH": 4,
        "CHANNEL": [1, 4, 16, 64, 128],
        "KRNLSIZE": 3,
        "POOLKRNLSIZE": 2,
        "PAD": 1,
        "DIL": 1,
        "STRIDE": 1,
        "BIAS": True,
        "ACT":nn.ELU(),
        "FCLAYERWIDTH":128
    }
    experiment_models = [baseline, strongerfc, dilated, elu]
    experiment_model_names = ["Baseline", "Stronger FC", "Dilated", "ELU"]
    
    with open("results.txt", "a") as res_file:
        
        print("\r\n--- PREPROCESSING PHASE ---")
        
        # Divide data into training, validation and test datasets
        
        # Randomly sample partition chunk for CV and Test
        rs = RandomState(MT19937(SeedSequence(329598739)))
        cvindex = rs.randint(0,PARTITION_ORDER)
        test_index = rs.randint(0, PARTITION_ORDER - 1)
        test_index = (cvindex + 1 + test_index) % PARTITION_ORDER
        
        partition_size = N_IMAGES // PARTITION_ORDER
        # Convert to float for compatibility with weights
        # Add new 'channel' dimension (unsqueeze)
        images = torch.from_numpy(images.copy()).float().unsqueeze(1).to(DEVICE)
        labels_onehot = torch.from_numpy(labels_onehot.copy()).to(DEVICE)
        partitions_data = torch.split(images, partition_size, dim=0)
        partitions_labels = torch.split(labels_onehot, partition_size, dim=0)
        
        train_data = list()
        train_labels = list()
        
        crossval_data = partitions_data[cvindex]
        crossval_labels = partitions_labels[cvindex]

        test_data = partitions_data[test_index]
        test_labels = partitions_labels[test_index]
        
        for i in range(PARTITION_ORDER):
            if i != cvindex and i != test_index:
                train_data.append(partitions_data[i])
                train_labels.append(partitions_labels[i])
        train_data = torch.cat(train_data)
        train_labels = torch.cat(train_labels)

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
            cnn = CNN(
                layer_depth = combi["LAYER_DEPTH"],
                channel = combi["CHANNEL"],
                kernel_size=combi["KRNLSIZE"],
                pool_kernel_size=combi["POOLKRNLSIZE"],
                padding = combi["PAD"],
                dilation = combi["DIL"],
                stride=combi["STRIDE"],
                input_size=image_size,
                bias=combi["BIAS"],
                f_act=combi["ACT"],
                fc_layer_width=combi["FCLAYERWIDTH"],
                output_size=10
            )
            
            log_accuracy, log_cv_accuracy = train(cnn, train_data, train_labels, crossval_data, crossval_labels, res_file)
            
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

            test_pred = cnn(test_data)
            test_acc = torch.mul(test_pred, test_labels).sum(dim=1).mean()
            if WRITE_TO_FILE:
                res_file.write("\r\n-- TEST RESULTS --\r\n")
                res_file.write(f"Test Accuracy: {test_acc:0.4f}\r\n")


            # Confusion Matrix, rows: truth, columns: prediction
            confusion_mat = torch.zeros(10,10).to(DEVICE)
            label_count = torch.zeros(10)
            for pred, label in zip(test_pred, test_labels):
                for i, j in enumerate(label.to(int)):
                    if j == 0:
                        continue
                    confusion_mat[i] += pred
                    label_count[i] += 1
            
            # Normalize matrix along rows
            if WRITE_TO_FILE:
                res_file.write("Confusion matrix (truth,prediction)\r\n")
            for i in range(10):
                confusion_mat[i] /= label_count[i]
                if WRITE_TO_FILE:
                    for j in range(10):
                        res_file.write(f"{confusion_mat[i][j]:.4f}\t")
                    res_file.write("\r\n")
            
        res_file.flush()
        res_file.close()
    return 0
    

if __name__=="__main__":
    main()
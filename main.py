import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

class Preprocesing:
    def __init__(self):
        self.amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        self.secondaries = 'CHE'

    def one_hot_encode(self, sequence, primary = True):
        # Initialize an array of zeros with shape (length of data, number of amino acids)
        one_hot_encoded_sequence = np.zeros((len(sequence), len(self.amino_acids)), dtype=int)

        for i, value in enumerate(sequence):
            if primary:
                inde = self.amino_acids.index(value)
            else:
                inde = self.secondaries.index(value)
            one_hot_encoded_sequence[i, inde] = 1

        return one_hot_encoded_sequence

    def preprocess(self):
        pass


seq = 'AIKWWWC'
new = Preprocesing()
data = pd.read_csv("2018-06-06-ss.cleaned.csv")
filtered_data = data[data['has_nonstd_aa'] == False]
sequence = filtered_data['seq']

#filter out sequences with non-standard amino acids

sequences = []
for i in sequence:
    sequences.append(new.one_hot_encode(i))



class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(Model,self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        # Forward propagate the LSTM
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

        


class AminoAcidDataset(Dataset):
    def __init__(self, sequences, labels, lengths):
        self.sequences = sequences 
        self.lengths = lengths # A list of encoded amino acid sequences (as tensors)
        self.labels = labels        # Corresponding labels or targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.lengths[idx], self.labels[idx]


# dataset = AminoAcidDataset(sequences, labels)
# dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)

# for batch in dataloader:
#     sequence_batch, label_batch = batch
#     # Your training loop here


# from dataset import ImageDataset
# from transforms import get_transforms_val, get_transforms_train
# from cnn_network import CNN, get_loss_function, get_optimizer

# # *********************************************************** #
# # set all training parameters. You can play around with these
# # *********************************************************** #

# batch_size = 12          # Number of images in each batch
# learning_rate = 0.001    # Learning rate in the optimizer
# momentum = 0.8            # Momentum in SGD
# num_epochs = 5          # Number of passes over the entire dataset
# print_every_iters = 100  # Print training loss every X mini-batches


# # *********************************************************** #
# # Initialize all the training components, e.g. model, cost function
# # *********************************************************** #

# # Get transforms
# transform_train = get_transforms_train()
# transform_val = get_transforms_val()

# # Generate our data loaders
# dataset_train = ImageDataset(data_path/'train.csv', data_path/'train.hdf5', get_transforms_train())
# dataset_valid = ImageDataset(data_path/'val.csv', data_path/'val.hdf5', get_transforms_val())

# train_loader = DataLoader(dataset_train, batch_size, shuffle=True, num_workers=2)
# valid_loader = DataLoader(dataset_valid, batch_size, shuffle=True, num_workers=2)

# # create CNN model
# cnn_network = CNN()

# # Get optimizer and loss functions
# criterion = get_loss_function()
# optimizer = get_optimizer(cnn_network, lr=learning_rate, momentum=momentum)






# # *********************************************************** #
# # The main training loop. You dont need to change this
# # *********************************************************** #
# training_loss_per_epoch = []
# val_loss_per_epoch = []
# for epoch in range(num_epochs):  # loop over the dataset multiple times
#     # First we loop over training dataset
#     running_loss = 0.0
#     for i, data in enumerate(train_loader, 0):
#         # get the inputs; data is a list of [inputs, labels]
#         inputs, labels = data

#         # zero the parameter gradients
#         optimizer.zero_grad()  # zero the gradients from previous iteration

#         # forward + backward + optimize
#         outputs = cnn_network(inputs)  # forward pass to obtain network outputs
#         loss = criterion(outputs, labels)  # compute loss with respect to labels
#         loss.backward()  # compute gradients with backpropagation (autograd)
#         optimizer.step()  # optimize network parameters

#         # print statistics
#         running_loss += loss.item()
#         if (i + 1) % print_every_iters == 0:
#             print(
#                 f'[Epoch: {epoch + 1} / {num_epochs},'
#                 f' Iter: {i + 1:5d} / {len(train_loader)}]'
#                 f' Training loss: {running_loss / (i + 1):.3f}'
#             )

#     mean_loss = running_loss / len(train_loader)
#     training_loss_per_epoch.append(mean_loss)

#     # Next we loop over validation dataset
#     running_loss = 0.0
#     for i, data in enumerate(valid_loader, 0):
#         # get the inputs; data is a list of [inputs, labels]
#         inputs, labels = data

#         # on validation dataset, we only do forward, without computing gradients
#         with torch.no_grad():
#             outputs = cnn_network(inputs)  # forward pass to obtain network outputs
#             loss = criterion(outputs, labels)  # compute loss with respect to labels

#         # print statistics
#         running_loss += loss.item()

#     mean_loss = running_loss / len(valid_loader)
#     val_loss_per_epoch.append(mean_loss)

#     print(
#         f'[Epoch: {epoch + 1} / {num_epochs}]'
#         f' Validation loss: {mean_loss:.3f}'
#     )

# print('Finished Training')

# # Plot the training curves
# plt.figure()
# plt.plot(np.array(training_loss_per_epoch))
# plt.plot(np.array(val_loss_per_epoch))
# plt.legend(['Training loss', 'Val loss'])
# plt.xlabel('Epoch')
# plt.show()
# plt.close()
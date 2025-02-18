import torch
from torch import nn

########RNN################
class RNNUnit(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, features, input_size, hidden_size, num_layers=2):
        super(RNNUnit, self).__init__()

        self.encoder = nn.Sequential(nn.Linear(features, input_size))
        self.rnn = nn.RNN(input_size, hidden_size, num_layers)
        self.decoder = nn.Sequential(nn.Linear(hidden_size, features))
        # self.out = nn.Linear(hidden_size, features)

    def forward(self, x, prev_hidden):

        L, B, F = x.shape
        output = x.reshape(L * B, -1)
        output = self.encoder(output) #(L*B,F)->(L*B,input_size)
        #
        output = output.reshape(L, B, -1) #(L*B,input_size)->(L,B,input_size)
        #NN.RNN(input_size, hidden_size, num_layers)
        #input: (seq_len, batch, input_size)
        #output: (seq_len, batch, hidden_size)
        #cur_hidden: (num_layers * num_directions, batch, hidden_size)
        # prev_hidden:initial state
        output, cur_hidden = self.rnn(output, prev_hidden) #output:(L,B,hidden_size) cur_hidden:(num_layers,B,hidden_size)
        #
        output = output.reshape(L * B, -1) #(L,B,hidden_size)->(L*B,hidden_size)
        #
        output = self.decoder(output) #(L*B,hidden_size)->(L*B,features)
        # output = self.out(torch.cos(output))
        output = output.reshape(L, B, -1) #(L*B,features)->(L,B,features)

        return output, cur_hidden


class RNN(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, features, input_size, hidden_size, num_layers=2):

        super(RNN, self).__init__()
        self.num_layers = num_layers ## the number of layers in the hidden layer inside the RNN
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.features = features
        self.model = RNNUnit(features, input_size, hidden_size, num_layers=self.num_layers)

    def train_pro(self, x, pred_len, device):

        BATCH_SIZE, seq_len, _ = x.shape ##input:[batch_size,seq_len,feature_size]
        prev_hidden = torch.zeros(self.num_layers, BATCH_SIZE, self.hidden_size).to(device) ##initialize the hidden layer parameters
        outputs = []
        #make the next prediction
        #seq_len: input sequence length
        #pred_len: predicted sequence length
        for idx in range(seq_len + pred_len - 1):
            if idx < seq_len:
                # output:(seq_len,batch,hidden_size)=(1,batchsize,hidden_size)
                output, prev_hidden = self.model(x[:, idx:idx + 1, ...].permute(1, 0, 2).contiguous(), prev_hidden)
            else:
                #output:(seq_len,batch,hidden_size)=(1,batchsize,hidden_size)
                output, prev_hidden = self.model(output, prev_hidden)
            if idx >= seq_len - 1:
                outputs.append(output)
        outputs = torch.cat(outputs, dim=0).permute(1, 0, 2).contiguous()

        return outputs

    def forward(self, x, pred_len, device):
        # x: (batch_size, seq_len, feature_size)
        # pred_len: predicted sequence length
        # output: (batch_size, pred_len, feature_size)
        return self.train_pro(x, pred_len, device)

###########GRU###############
class GRUUnit(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, features, input_size, hidden_size, num_layers=2):
        super(GRUUnit, self).__init__()

        self.encoder = nn.Sequential(nn.Linear(features, input_size))
        self.gru = nn.GRU(input_size, hidden_size, num_layers)
        self.decoder = nn.Sequential(nn.Linear(hidden_size, features))
        # self.out = nn.Linear(hidden_size, features)

    def forward(self, x, prev_hidden):

        L, B, F = x.shape
        output = x.reshape(L * B, -1)
        output = self.encoder(output)
        #
        output = output.reshape(L, B, -1)
        output, cur_hidden = self.gru(output, prev_hidden)
        #
        output = output.reshape(L * B, -1)
        #
        output = self.decoder(output)
        # output = self.out(torch.cos(output))
        output = output.reshape(L, B, -1)

        return output, cur_hidden


class GRU(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, features, input_size, hidden_size, num_layers=2):

        super(GRU, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.features = features
        self.model = GRUUnit(features, input_size, hidden_size, num_layers=self.num_layers)

    def train_pro(self, x, pred_len, device):

        BATCH_SIZE, seq_len, _ = x.shape
        prev_hidden = torch.zeros(self.num_layers, BATCH_SIZE, self.hidden_size).to(device)
        outputs = []
        for idx in range(seq_len + pred_len - 1):
            if idx < seq_len:
                output, prev_hidden = self.model(x[:, idx:idx + 1, ...].permute(1, 0, 2).contiguous(), prev_hidden)
            else:
                output, prev_hidden = self.model(output, prev_hidden)
            if idx >= seq_len - 1:
                outputs.append(output)
        outputs = torch.cat(outputs, dim=0).permute(1, 0, 2).contiguous()

        return outputs

    def forward(self, x, pred_len, device):
        return self.train_pro(x, pred_len, device)

#########LSTM###########
class LSTMUnit(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, features, input_size, hidden_size, num_layers=2):
        super(LSTMUnit, self).__init__()

        self.encoder = nn.Sequential(nn.Linear(features, input_size))
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.decoder = nn.Sequential(nn.Linear(hidden_size, features))
        # self.out = nn.Linear(hidden_size, features)

    def forward(self, x, prev_hidden, prev_cell):

        L, B, F = x.shape
        output = x.reshape(L * B, -1)
        output = self.encoder(output)
        #
        output = output.reshape(L, B, -1)
        output, (cur_hidden, cur_cell) = self.lstm(output, (prev_hidden, prev_cell)) ##pre_hidden:隐藏层 prev_cell:细胞状态
        #
        output = output.reshape(L * B, -1)
        #
        output = self.decoder(output)
        # output = self.out(torch.cos(output))
        output = output.reshape(L, B, -1)

        return output, cur_hidden, cur_cell


class LSTM(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, features, input_size, hidden_size, num_layers=2):

        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.features = features
        self.model = LSTMUnit(features, input_size, hidden_size, num_layers=self.num_layers)

    def train_pro(self, x, pred_len, device):

        BATCH_SIZE, seq_len, _ = x.shape
        prev_hidden = torch.zeros(self.num_layers, BATCH_SIZE, self.hidden_size).to(device)
        prev_cell = torch.zeros(self.num_layers, BATCH_SIZE, self.hidden_size).to(device)
        outputs = []
        for idx in range(seq_len + pred_len - 1):
            if idx < seq_len:
                output, prev_hidden, prev_cell = self.model(x[:, idx:idx + 1, ...].permute(1, 0, 2).contiguous(),
                                                            prev_hidden, prev_cell)
            else:
                output, prev_hidden, prev_cell = self.model(output, prev_hidden, prev_cell)
            if idx >= seq_len - 1:
                outputs.append(output)
        outputs = torch.cat(outputs, dim=0).permute(1, 0, 2).contiguous()

        return outputs

    def forward(self, x, pred_len, device):
        return self.train_pro(x, pred_len, device)
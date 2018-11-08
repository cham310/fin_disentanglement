import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        out = self.fc(hn[-1])
        return out


class MLP(nn.Module):
    def __init__(self, layers):
        super(MLP, self).__init__()
        self.layers = layers
        net = []
        for n, (inp, outp) in enumerate(zip(layers, layers[1:])):
            net.append(nn.Linear(inp, outp))
            net.append(nn.ReLU(inplace=True))
        net = nn.ModuleList(net[:-1])
        self.net = nn.Sequential(*net)
        print(self.net)

    def forward(self, x):
        x = self.net(x)
        return x


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, sequence_length):
        super(Decoder, self).__init__()
        self.sequence_length = sequence_length
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, c, s):
        x = torch.cat([c, s], 1).unsqueeze(1).expand(-1, self.sequence_length, -1)
        out, (hn, cn) = self.lstm(x)
        out = self.fc(out)
        return out.squeeze(2)

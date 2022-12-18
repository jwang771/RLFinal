import numpy as np
import random
import torch
import torch.nn.functional as F

    
class LSTMencoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMencoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.net = torch.nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, input, hidden):   
        output, hidden = self.net(torch.FloatTensor(input).view(1, -1, self.input_size), hidden)
        return output[-1].reshape(-1, self.hidden_size)

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_size),
                torch.zeros(1, 1, self.hidden_size))


class encdecoder_actor(torch.nn.Module):
    def __init__(self, input_size, hidden_size, linear_size):
        super(encdecoder_actor, self).__init__()
        self.AbRNN = LSTMencoder(int(input_size), int(hidden_size))
        self.cutsRNN = LSTMencoder(int(input_size), int(hidden_size))
        self.model = torch.nn.Sequential(
                    #input layer of input size obssize
                    torch.nn.Linear(hidden_size, linear_size),
                    torch.nn.ReLU(),
                    #intermediate layers
                    #output layer of output size actsize
                    torch.nn.Linear(linear_size, linear_size)
                )
    def forward(self, Ab, cuts):
        #compute projections
        gj = self.model(self.AbRNN.forward(torch.FloatTensor(Ab), self.AbRNN.init_hidden()))
        hi = self.model(self.cutsRNN.forward(torch.FloatTensor(cuts), self.cutsRNN.init_hidden()))
        #return score
        return torch.mean(torch.mm(hi, gj.T), axis=1)

class encdecoder_critic(torch.nn.Module):
    def __init__(self, input_size, hidden_size, linear_size):
        super(encdecoder_critic, self).__init__()
        self.AbRNN = LSTMencoder(int(input_size), int(hidden_size))
        self.cutsRNN = LSTMencoder(int(input_size), int(hidden_size))
        self.model = torch.nn.Sequential(
                    #input layer of input size obssize
                    torch.nn.Linear(hidden_size, linear_size),
                    torch.nn.ReLU(),
                    #intermediate layers
                )
        self.linear = torch.nn.Linear(128, 1)

    def forward(self, Ab, cuts):
        #compute projections
        gj = self.model(self.AbRNN.forward(torch.FloatTensor(Ab), self.AbRNN.init_hidden()))
        hi = self.model(self.cutsRNN.forward(torch.FloatTensor(cuts), self.cutsRNN.init_hidden()))
        #return score
        score = torch.mean(torch.mm(hi, gj.T), axis=1)
        score_pad = F.pad(score, pad=(0, 128 - score.shape[0]))
        Q_value = self.linear(score_pad)
        return Q_value


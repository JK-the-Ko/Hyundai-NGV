import torch
from torch import nn


class LSTM(nn.Module) :
    def __init__(self, opt) :
        # Inheritance
        super(LSTM, self).__init__()

        # Create LSTM Layer Instance
        self.lstm = nn.LSTM(opt.hid_channels, opt.hid_channels, num_layers=opt.num_layer, bidirectional=False, batch_first=True, dropout=opt.p)
        self.bilstm = nn.LSTM(opt.hid_channels, opt.hid_channels//2, num_layers=opt.num_layer, bidirectional=True, batch_first=True, dropout=opt.p)

        # Create FC Layer Instance
        self.input2lstm = nn.Linear(opt.in_channels, opt.hid_channels)
        self.input2bilstm = nn.Linear(opt.in_channels, opt.hid_channels)
        self.input2output = nn.Linear(opt.in_channels, opt.hid_channels)
        self.fc0 = nn.Linear(opt.hid_channels*2, opt.hid_channels, bias=False)
        self.fc1 = nn.Linear(opt.hid_channels, opt.hid_channels, bias=False)
        self.fc2 = nn.Linear(opt.hid_channels, opt.out_channels)
        
        # Create Layer Normalization Layer Instance
        self.norm0 = nn.LayerNorm(opt.hid_channels)
        self.norm1 = nn.LayerNorm(opt.hid_channels)

        # Create Activation Layer Instance
        self.act = nn.ReLU(inplace=True)

    def forward(self, input) :
        lstmOutput, _ = self.lstm(self.input2lstm(input))
        bilstmOutput, _ = self.bilstm(self.input2bilstm(input))

        output = self.norm0(self.act(self.fc0(torch.cat([lstmOutput, bilstmOutput], dim=-1))))
        output = self.norm1(self.act(self.fc1(output))) + self.input2output(input)
        output = self.fc2(output)

        return output
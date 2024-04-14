from .unet_model import UNet
import torch
import torch.nn as nn


def outer_concatenation(x, y):
    y_expand = y.unsqueeze(1)
    x_expand = x.unsqueeze(2)
    y_repeat = y_expand.repeat(1, y_expand.shape[2], 1, 1)
    x_repeat = x_expand.repeat(1, 1, x_expand.shape[1], 1)
    outer_concat = torch.cat((x_repeat, y_repeat), 3)
    return outer_concat.permute(0, 3, 1, 2).contiguous()


class Bilstm_ProposedModel(nn.Module):
    def __init__(self, input_ch):
        super().__init__()
        self.D = 1
        self.num_layers = 1

        self.conv1 = nn.Conv1d(in_channels =input_ch,out_channels=input_ch,kernel_size =3)
        self.bidirectional_lstm = nn.LSTM(input_size = input_ch,hidden_size =input_ch,bidirectional=True,batch_first =True, num_layers=self.num_layers)

        self.unet = UNet(n_channels=input_ch*2, n_classes=1)

    def forward(self, data1, data2):
        data1 = self.conv1(data1)
        data2 = self.conv1(data2)
        
        hidden1 = (torch.randn(self.D*self.num_layers, batch_size, input_ch), torch.randn(self.D*self.num_layers, batch_size, input_ch))
        data1 = self.bidirectional_lstm(data1)

        hidden2 = (torch.randn(self.D*self.num_layers, batch_size, input_ch), torch.randn(self.D*self.num_layers, batch_size, input_ch))
        data2 = self.bidirectional_lstm(data2)

        pred_path = self.unet(outer_concatenation(data1, data2))
        return pred_path.squeeze(1)

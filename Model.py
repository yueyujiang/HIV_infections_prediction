import torch
import torchvision
from torch import nn

class G0_model(nn.Module):
    def __init__(self, input_length):
        super(G0_model, self).__init__()
        self.rnn1 = nn.LSTMCell(1, 20)
        self.rnn2 = nn.LSTMCell(20, 20)
        self.linear1 = nn.Linear(20,64)
        self.linear2 = nn.Linear(64, 1)
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.input_length = input_length

    def forward(self, diagnosis):
        x = diagnosis.unsqueeze(2)
        # l = x.shape[1]
        for i in range(0, self.input_length):
            # for i in range(30):
                if i == 0:
                    h1 = self.rnn1(x[:, i, :])
                    h2 = self.rnn2(h1[0])
                else:
                    h1 = self.rnn1(x[:, i, :], h1)
                    h2 = self.rnn2(h1[0], h2)
        y1 = self.linear1(h2[0])
        y1 = self.relu(y1)
        y2 = self.linear2(y1)
        y2 = self.relu(y2)
        return y2

class p_model(nn.Module):
    def __init__(self, output_length):
        super(p_model, self).__init__()
        self.rnn1_g0 = nn.LSTMCell(1, 20)
        self.rnn2_g0 = nn.LSTMCell(20, 20)

        self.rnn1 = nn.LSTMCell(2, 20)
        self.rnn2 = nn.LSTMCell(20, 20)
        # self.rnn3 = nn.LSTMCell(20, 20)
        self.linear1_x = nn.Linear(20, 64)
        self.linear2_x = nn.Linear(64, 1)
        self.linear1_y = nn.Linear(20, 64)
        self.linear2_y = nn.Linear(64, 1)
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.output_length = output_length
        self.rnn1_p = nn.LSTMCell(1, 20)
        self.rnn2_p = nn.LSTMCell(20, 20)

        self.linear1_p = nn.Linear(20, 64)
        self.linear2_p = nn.Linear(64, 1)

    def forward(self, diagnosis, train=True):
        length = diagnosis.shape[1]
        x = diagnosis.unsqueeze(2)
        for i in range(1, diagnosis.shape[1]):
            if i == 1:
                h0 = self.rnn1_g0(x[:, i, :])
            else:
                h0 = self.rnn1_g0(x[:, i, :], h0)
        g0 = self.linear1_y(h0[0])
        g0 = self.linear2_y(g0)
        # g0 = torch.zeros((diagnosis.shape[0],))
        # infection = infection.unsqueeze(2)
        output = torch.zeros((diagnosis.shape[0], length - 1))
        output[:, 0] = g0.squeeze(1)
        tmp = torch.cat((x[:, 1, :], g0), 1)
        for i in range(1, length - 1):
            if i == 1:
                h1 = self.rnn1(tmp)
                h2 = self.rnn2(h1[0])
            else:
                h1 = self.rnn1(tmp, h1)
                h2 = self.rnn2(h1[0], h2)
                # h3 = self.rnn3(h2[0], h3)
                # h1 = (nn.functional.normalize(h1[0]), nn.functional.normalize(h1[1]))
            y0 = self.linear1_x(h2[0])
            y0 = self.relu(y0)
            y1 = self.linear2_x(y0)
            tmp = torch.cat((x[:, i + 1, :], y1), 1)
            output[:, i] = y1.squeeze(1) + output[:, i - 1]
        pre = output.clone()
        prediction = torch.zeros((diagnosis.shape[0], self.output_length))
        prediction[:, 0] = pre[:, -1]
        pre_len = pre.shape[0]
        for i in range(1, self.output_length + pre_len):
            if i == 1:
                h1 = self.rnn1_p(pre[:, i].unsqueeze(1))
                h2 = self.rnn2_p(h1[0])
            else:
                h1 = self.rnn1_p(pre[:, i].unsqueeze(1), h1)
                h2 = self.rnn2_p(h1[0], h2)
                # h3 = self.rnn3(h2[0], h3)
                # h1 = (nn.functional.normalize(h1[0]), nn.functional.normalize(h1[1]))
            if i > pre_len:
                y0 = self.linear1_p(h2[0])
                y0 = self.relu(y0)
                y1 = self.linear2_p(y0)
                prediction[:, i - pre_len] = y1.squeeze(1) + prediction[:, i - pre_len - 1]
                pre = torch.cat((pre, prediction[:, i - pre_len].unsqueeze(1)), 1)
        return output, g0, prediction

# class G0_model(nn.Module):
#     def __init__(self, input_length):
#         super(G0_model, self).__init__()
#         self.rnn1 = nn.LSTMCell(1, 32)
#         self.rnn2 = nn.LSTMCell(32, 32)
#         self.linear1 = nn.Linear(32,64)
#         self.linear2 = nn.Linear(64, 1)
#         self.softmax = nn.Softmax()
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout()
#         self.input_length = input_length
#
#     def forward(self, diagnosis):
#         x = diagnosis.unsqueeze(2)
#         # l = x.shape[1]
#         for i in range(0, self.input_length):
#             # for i in range(30):
#                 if i == 0:
#                     h1 = self.rnn1(x[:, i, :])
#                     h2 = self.rnn2(h1[0])
#                 else:
#                     h1 = self.rnn1(x[:, i, :], h1)
#                     h2 = self.rnn2(h1[0], h2)
#         y1 = self.linear1(h2[0])
#         y1 = self.relu(y1)
#         y2 = self.linear2(y1)
#         y2 = self.relu(y2)
#         return y2
#
# class p_model(nn.Module):
#     def __init__(self, output_length):
#         super(p_model, self).__init__()
#         self.rnn1_g0 = nn.LSTMCell(1, 32)
#         self.rnn2_g0 = nn.LSTMCell(32, 32)
#
#         self.rnn1 = nn.LSTMCell(2, 32)
#         self.rnn2 = nn.LSTMCell(32, 32)
#         # self.rnn3 = nn.LSTMCell(20, 20)
#         self.linear1_x = nn.Linear(32, 64)
#         self.linear2_x = nn.Linear(64, 1)
#         self.linear1_y = nn.Linear(32, 64)
#         self.linear2_y = nn.Linear(64, 1)
#         self.softmax = nn.Softmax()
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout()
#         self.output_length = output_length
#         self.rnn1_p = nn.LSTMCell(1, 32)
#         self.rnn2_p = nn.LSTMCell(32, 32)
#
#         self.linear1_p = nn.Linear(32, 64)
#         self.linear2_p = nn.Linear(64, 1)
#
#     def forward(self, diagnosis, train=True):
#         length = diagnosis.shape[1]
#         x = diagnosis.unsqueeze(2)
#         for i in range(1, diagnosis.shape[1]):
#             if i == 1:
#                 h0 = self.rnn1_g0(x[:, i, :])
#             else:
#                 h0 = self.rnn1_g0(x[:, i, :], h0)
#         g0 = self.linear1_y(h0[0])
#         g0 = self.linear2_y(g0)
#         # g0 = torch.zeros((diagnosis.shape[0],))
#         # infection = infection.unsqueeze(2)
#         output = torch.zeros((diagnosis.shape[0], length - 1))
#         output[:, 0] = g0.squeeze(1)
#         tmp = torch.cat((x[:, 1, :], g0), 1)
#         for i in range(1, length - 1):
#             if i == 1:
#                 h1 = self.rnn1(tmp)
#                 h2 = self.rnn2(h1[0])
#             else:
#                 h1 = self.rnn1(tmp, h1)
#                 h2 = self.rnn2(h1[0], h2)
#                 # h3 = self.rnn3(h2[0], h3)
#                 # h1 = (nn.functional.normalize(h1[0]), nn.functional.normalize(h1[1]))
#             y0 = self.linear1_x(h2[0])
#             y0 = self.relu(y0)
#             y1 = self.linear2_x(y0)
#             tmp = torch.cat((x[:, i + 1, :], y1), 1)
#             output[:, i] = y1.squeeze(1) + output[:, i - 1]
#         pre = output.clone()
#         prediction = torch.zeros((diagnosis.shape[0], self.output_length))
#         prediction[:, 0] = pre[:, -1]
#         pre_len = pre.shape[0]
#         for i in range(1, self.output_length + pre_len):
#             if i == 1:
#                 h1 = self.rnn1_p(pre[:, i].unsqueeze(1))
#                 h2 = self.rnn2_p(h1[0])
#             else:
#                 h1 = self.rnn1_p(pre[:, i].unsqueeze(1), h1)
#                 h2 = self.rnn2_p(h1[0], h2)
#                 # h3 = self.rnn3(h2[0], h3)
#                 # h1 = (nn.functional.normalize(h1[0]), nn.functional.normalize(h1[1]))
#             if i > pre_len:
#                 y0 = self.linear1_p(h2[0])
#                 y0 = self.relu(y0)
#                 y1 = self.linear2_p(y0)
#                 prediction[:, i - pre_len] = y1.squeeze(1) + prediction[:, i - pre_len - 1]
#                 pre = torch.cat((pre, prediction[:, i - pre_len].unsqueeze(1)), 1)
#         return output, g0, prediction
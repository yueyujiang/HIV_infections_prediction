import argparse
import os
import utils
import torch.optim
import Model
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torch import nn
from tensorboardX import SummaryWriter
from data import dataset

parser = argparse.ArgumentParser(description='Test!')
parser.add_argument('--lr', type=float, default=5e-3,
                    help="learning rate")
parser.add_argument('-b', '--batch-size', type=int, default=10,
                    help="batch size")
parser.add_argument('-epochs', type=int, default=800,
                    help="epochs size")
parser.add_argument('-p', '--print-frequency', default=10,
                    help="the frequency of printing the value of loss")
parser.add_argument('-i', '--img-frequency', default=10,
                    help="the frequency of showing the image")
parser.add_argument('--summary-dir', default='./summary', metavar='DIR',
                    help='path to save summary')
parser.add_argument('--save-checkpoint-frequency', type=int, default=20,
                    help='the frequency to save checkpoint')
parser.add_argument('--checkpoint-path', type=int, default=None,
                    help='the path of the checkpoint')
parser.add_argument('--load-path-A', type=str,
                    help='the path of the pretrained parameter of Generator')
parser.add_argument('--load-path-C', type=str,
                    help='the path of the pretrained parameter of Distriminator')

train_time = 50
sequence_length = 450
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model_p = Model.p_model(output_length=sequence_length)
model_p = model_p.to(device)
optimizer_p = optim.Adam(model_p.parameters(), lr=args.lr)

checkpoint_path_g0 = './checkpoint/g0'
if not os.path.exists(checkpoint_path_g0):
    os.mkdir(checkpoint_path_g0)

checkpoint_path_p = './checkpoint/p'
if not os.path.exists(checkpoint_path_p):
    os.mkdir(checkpoint_path_p)

# checkpoint_path_C_load = './checkpoint/Component/m-1799-10.pth.tar'
# checkpoint_path_A_load = './checkpoint/Attention/m-1999-10.pth.tar'
# checkpoint_path_load = None

utils.load_checkpoint(model_p, checkpoint_path_p + '/m-13-13.pth.tar', optimizer_p)

testset = dataset(train=False)

testloader = DataLoader(testset, batch_size=args.batch_size,
                        shuffle=True, num_workers=8)

mseloss = nn.MSELoss().to(device)

writer = SummaryWriter(args.summary_dir)

alpha = torch.tensor(np.load('alpha.npy'))


def test(dataloader, model_p):
    print('testing!')
    counter = 0
    for batch_idx, sample in enumerate(dataloader):
        diagnosis = sample['diagnosis'][:, :train_time].to(device).float()
        gt_infection = sample['infection'][:, :train_time - 1].to(device).float()
        gt_prediction = sample['infection'][:, train_time - 2: train_time - 2 + sequence_length]
        infection, g0, prediction = model_p(diagnosis)
        if counter > 300:
            break
        for i in range(args.batch_size):
            counter += 1
            utils.draw(infection[i, :], diagnosis[i, :], gt_infection[i, :], prediction[i, :],
                       gt_prediction[i, :], False, counter)

# np.save('train_error.npy', np.array(train_error_record))
# np.save('test_error.npy', np.array(test_error_record))
test(testloader, model_p)
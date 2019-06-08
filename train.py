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
checkpoint_path_load = None

trainset = dataset(train=True)

trainloader = DataLoader(trainset, batch_size=args.batch_size,
                         shuffle=True, num_workers=8)

testset = dataset(train=False)

testloader = DataLoader(testset, batch_size=args.batch_size,
                        shuffle=True, num_workers=8)

mseloss = nn.MSELoss().to(device)

writer = SummaryWriter(args.summary_dir)

alpha = torch.tensor(np.load('alpha.npy'))


def test(dataloader, model_p, writer, train):
    error1 = 0
    error2 = 0
    print('testing!')
    counter = 0
    for batch_idx, sample in enumerate(dataloader):
        diagnosis = sample['diagnosis'][:, :train_time].to(device).float()
        gt_infection = sample['infection'][:, :train_time - 1].to(device).float()
        gt_prediction = sample['infection'][:, train_time - 2: train_time - 2 + sequence_length]
        infection, g0, prediction = model_p(diagnosis)
        infection_tmp = infection.detach()
        error1 += torch.sum(
            torch.abs(infection_tmp[gt_infection != 0] - gt_infection[gt_infection != 0]) / (
                torch.abs(gt_infection[gt_infection != 0]))) / (infection.shape[1])
        prediction_tmp = prediction.detach()
        error2 += torch.sum(
            torch.abs(prediction_tmp.float() - gt_prediction.float()) / (torch.abs(gt_prediction.float()))) / (
                      prediction.shape[1])
        if batch_idx % 20 == 0:
            for i in range(args.batch_size):
                img = utils.draw(infection[i, :], diagnosis[i, :], gt_infection[i, :], prediction[i, :],
                                 gt_prediction[i, :], train, i)
                writer.add_figure('img_test', img)
            counter += 1
            if counter == 3:
                break
    return error1 / (20 * 3 * 10), error2 / (20 * 3 * 10)


train_error_record = []
test_error_record = []

for epoch in range(15):  # loop over the dataset multiple times
    running_loss = 0
    train_error = 0
    test_error1, test_error2 = test(testloader, model_p, writer, train=False)
    train_error1, train_error2 = test(trainloader, model_p, writer, train=True)
    print('after ' + str(epoch) + ' epoch(s)')
    print('train_error for approximation', train_error1.item())
    print('test_error for approximation', test_error1.item())
    print('train_error for prediction', train_error2.item())
    print('test_error for prediction', test_error2.item())
    # test_error_record.append(test_error)
    # train_error_record.append(train_error)
    for batch_idx, sample in enumerate(trainloader):
        diagnosis = sample['diagnosis'][:, :train_time].to(device).float()
        gt_infection = sample['infection'][:, :train_time - 1].to(device).float()
        gt_prediction = sample['infection'][:, train_time - 2: train_time - 2 + sequence_length].to(device).float()
        gt_g0 = gt_infection[:, 0]
        infection, g0, prediction = model_p(diagnosis)
        loss = mseloss(infection, gt_infection) + mseloss(g0, gt_g0.unsqueeze(1)) + mseloss(prediction, gt_prediction)
        ###########################################################################################################
        loss.backward()
        nn.utils.clip_grad_norm_(model_p.parameters(), 1)
        optimizer_p.step()
        running_loss += loss
        infection_tmp = infection.detach()
        train_error += torch.sum(
            torch.abs(infection_tmp[gt_infection != 0] - gt_infection[gt_infection != 0]) / (
                torch.abs(gt_infection[gt_infection != 0]))) / (infection.shape[1])
        if batch_idx % 20 == 0:
            print('epoch:', epoch, 'batch:', batch_idx, 'loss:', running_loss.item())
            running_loss = 0
            for i in range(args.batch_size):
                img = utils.draw(infection[i, :], diagnosis[i, :], gt_infection[i, :], prediction[i, :],
                                 gt_prediction[i, :], True, 0)
                writer.add_figure('img', img)
    writer.add_scalar('loss', running_loss, epoch)
    print(running_loss.item())
    if epoch % args.print_frequency == args.print_frequency - 1:  # print every 2000 mini-batches
        print('[%d, %5d] loss of Generator and Encoder: %.6f' %
              (epoch + 1, epoch + args.print_frequency, running_loss))

    utils.save_checkpoint(model_p, epoch, epoch, optimizer_p, checkpoint_PATH=checkpoint_path_p)

# np.save('train_error.npy', np.array(train_error_record))
# np.save('test_error.npy', np.array(test_error_record))

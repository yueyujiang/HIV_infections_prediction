import torch
import matplotlib.pyplot as plt
import numpy as np
import os

def save_checkpoint(model, time_step, epochs, optimizer, checkpoint_PATH='./checkpoint'):
    torch.save({'epoch': epochs + 1, 'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()},
               checkpoint_PATH + '/m-' + str(epochs) + '-' + str(time_step) + '.pth.tar')

def load_checkpoint(model, checkpoint_PATH, optimizer):
    if checkpoint_PATH != None:
        model_CKPT = torch.load(checkpoint_PATH)
        model.load_state_dict(model_CKPT['state_dict'])
        print('loading checkpoint!')
        optimizer.load_state_dict(model_CKPT['optimizer'])
    return model, optimizer

def cross_entropy(result, target):
    t = torch.zeros(result.shape)
    for i in range(max(target) + 1):
        t[target==i, i] = 1

    return - torch.sum(t * torch.log(result + 1e-12))

def draw(infection, diagnosis, gt_infection, prediction, gt_prediction, train, n):
    b = infection.detach()
    prediction = prediction.detach()
    fig = plt.figure()
    pre_len = len(infection)
    plt.plot(np.array(b))
    plt.plot(np.array(diagnosis))
    plt.plot(np.array(gt_infection))
    plt.plot(np.arange(pre_len - 1, pre_len + len(gt_prediction) - 1), np.array(gt_prediction))
    plt.plot(np.arange(pre_len - 1, pre_len + len(gt_prediction) - 1), np.array(prediction))
    plt.legend(['approximated infection', 'diagnosis', 'gt infection', 'gt prediction', 'prediction'])
    if not train:
        if not os.path.exists('./test_img'):
            os.mkdir('./test_img')
        plt.savefig('./test_img/' + str(n) + '.png')
        plt.close()
    return fig


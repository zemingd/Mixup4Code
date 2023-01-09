import numpy as np
import torch


##### Mixup-Tensorflow ################
def mixup_data( x, y, alpha, runs):
    if runs is None:
        runs = 1
    output_x = []
    output_y = []
    batch_size = x.shape[0]
    for i in range(runs):
        lam_vector = np.random.beta(alpha, alpha, batch_size)
        index = np.random.permutation(batch_size)
        mixed_x = (x.T * lam_vector).T + (x[index, :].T * (1.0 - lam_vector)).T
        output_x.append(mixed_x)
        if y is None:
            return np.concatenate(output_x, axis=0)
        mixed_y = (y.T * lam_vector).T + (y[index].T * (1.0 - lam_vector)).T
        output_y.append(mixed_y)
    return np.concatenate(output_x, axis=0), np.concatenate(output_y, axis=0)



def mixup_data_refactor( x, y, x_refactor, y_refactor, alpha, runs):
    if runs is None:
        runs = 1
    output_x = []
    output_y = []
    batch_size = x.shape[0]
    for i in range(runs):
        lam_vector = np.random.beta(alpha, alpha, batch_size)
        index = np.random.permutation(batch_size)

        mixed_x = (x.T * lam_vector).T + (x_refactor[index, :].T * (1.0 - lam_vector)).T
        output_x.append(mixed_x)
        if y is None:
            return np.concatenate(output_x, axis=0)
        mixed_y = (y.T * lam_vector).T + (y_refactor[index].T * (1.0 - lam_vector)).T
        output_y.append(mixed_y)
    return np.concatenate(output_x, axis=0), np.concatenate(output_y, axis=0)




##### Mixup-Pytorch ################
def mixup_data(x, y, alpha=0.1, runs, use_cuda=True):
    for i in range(runs):
        output_x = torch.Tensor(0)
        output_x= output_x.numpy().tolist()
        output_y = torch.Tensor(0)
        output_y = output_y.numpy().tolist()
        batch_size = x.size()[0]
        if alpha > 0.:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.

        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        mixed_y = lam * y + (1 - lam) * y[index, :]
        output_x.append(mixed_x)
        output_y.append(mixed_y)
    return torch.cat(output_x,dim=0), torch.cat(output_y,dim=0)


def mixup_data_refactor( x, y, x_refactor, y_refactor, alpha, runs, use_cuda=True):
    for i in range(runs):
        output_x = torch.Tensor(0)
        output_x= output_x.numpy().tolist()
        output_y = torch.Tensor(0)
        output_y = output_y.numpy().tolist()
        batch_size = x.size()[0]
        if alpha > 0.:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.
        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)
        mixed_x = lam * x + (1 - lam) * x_refactor[index, :]
        mixed_y = lam * y + (1 - lam) * y_refactor[index, :]
        output_x.append(mixed_x)
        output_y.append(mixed_y)
    return torch.cat(output_x,dim=0), torch.cat(output_y,dim=0)

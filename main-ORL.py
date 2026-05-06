import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from post_clustering import spectral_clustering, acc, nmi, purity
import scipy.io as sio
import math
from block import *
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.cluster.spectral import spectral_clustering_1


class Conv2dSamePad(nn.Module):
    def __init__(self, kernel_size, stride):
        super(Conv2dSamePad, self).__init__()
        self.kernel_size = kernel_size if type(kernel_size) in [list, tuple] else [kernel_size, kernel_size]
        self.stride = stride if type(stride) in [list, tuple] else [stride, stride]

    def forward(self, x):
        in_height = x.size(2)
        in_width = x.size(3)
        out_height = math.ceil(float(in_height) / float(self.stride[0]))
        out_width = math.ceil(float(in_width) / float(self.stride[1]))
        pad_along_height = ((out_height - 1) * self.stride[0] + self.kernel_size[0] - in_height)
        pad_along_width = ((out_width - 1) * self.stride[1] + self.kernel_size[1] - in_width)
        pad_top = math.floor(pad_along_height / 2)
        pad_left = math.floor(pad_along_width / 2)
        pad_bottom = pad_along_height - pad_top
        pad_right = pad_along_width - pad_left
        return F.pad(x, [pad_left, pad_right, pad_top, pad_bottom], 'constant', 0)


class ConvTranspose2dSamePad(nn.Module):
    def __init__(self, kernel_size, stride):
        super(ConvTranspose2dSamePad, self).__init__()
        self.kernel_size = kernel_size if type(kernel_size) in [list, tuple] else [kernel_size, kernel_size]
        self.stride = stride if type(stride) in [list, tuple] else [stride, stride]

    def forward(self, x):
        in_height = x.size(2)
        in_width = x.size(3)
        pad_height = self.kernel_size[0] - self.stride[0]
        pad_width = self.kernel_size[1] - self.stride[1]
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        return x[:, :, pad_top:in_height - pad_bottom, pad_left: in_width - pad_right]


class ConvAE(nn.Module):
    def __init__(self, channels, kernels, T):
        super(ConvAE, self).__init__()
        assert isinstance(channels, list) and isinstance(kernels, list)

        self.encoder1 = nn.Sequential()
        self.encoder1.add_module('pad%d' % 1, Conv2dSamePad(kernels[0], 2))
        self.encoder1.add_module('conv%d' % 1,nn.Conv2d(channels[0], channels[1], kernel_size=kernels[0], stride=2))
        self.encoder1.add_module('relu%d' % 1, nn.ReLU(True))

        self.encoder2 = nn.Sequential()
        self.encoder2.add_module('pad%d' % 2, Conv2dSamePad(kernels[1], 2))
        self.encoder2.add_module('conv%d' % 2,nn.Conv2d(channels[1], channels[2], kernel_size=kernels[1], stride=2))
        self.encoder2.add_module('relu%d' % 2, nn.ReLU(True))

        self.encoder3 = nn.Sequential()
        self.encoder3.add_module('pad%d' % 3, Conv2dSamePad(kernels[2], 2))
        self.encoder3.add_module('conv%d' % 3,nn.Conv2d(channels[2], channels[3], kernel_size=kernels[2], stride=2))
        self.encoder3.add_module('relu%d' % 3, nn.ReLU(True))

        channels = list(reversed([channels[i] - T[i] for i in range(0, len(channels))]))
        kernels = list(reversed(kernels))

        self.decoder3 = nn.Sequential()
        self.decoder3.add_module('deconv%d' % 1, nn.ConvTranspose2d(channels[0], channels[1], kernel_size=kernels[0], stride=2))
        self.decoder3.add_module('padd%d' % 1, ConvTranspose2dSamePad(kernels[0], 2))
        self.decoder3.add_module('relud%d' % 1, nn.ReLU(True))

        self.decoder2 = nn.Sequential()
        self.decoder2.add_module('deconv%d' % 2, nn.ConvTranspose2d(channels[1], channels[2], kernel_size=kernels[1], stride=2))
        self.decoder2.add_module('padd%d' % 2, ConvTranspose2dSamePad(kernels[1], 2))
        self.decoder2.add_module('relud%d' % 2, nn.ReLU(True))

        self.decoder1 = nn.Sequential()
        self.decoder1.add_module('deconv%d' % 3, nn.ConvTranspose2d(channels[2], channels[3], kernel_size=kernels[2], stride=2))
        self.decoder1.add_module('padd%d' % 3, ConvTranspose2dSamePad(kernels[2], 2))
        self.decoder1.add_module('relud%d' % 3, nn.ReLU(True))

    def forward(self, x):
        h = self.encoder1(x)
        h = self.encoder2(h)
        h = self.encoder3(h)
        y = self.decoder3(h)
        y = self.decoder2(y)
        y = self.decoder1(y)
        return y


class SelfExpression(nn.Module):
    def __init__(self, n):
        super(SelfExpression, self).__init__()
        self.Coefficient = nn.Parameter(1.0e-8 * torch.ones(n, n, dtype=torch.float32), requires_grad=True)

    def forward(self, x):  # shape=[n, d]
        y = torch.matmul(self.Coefficient, x)
        return y


class DSCNet(nn.Module):
    def __init__(self, channels, kernels, num_sample, T):
        super(DSCNet, self).__init__()
        self.n = num_sample
        self.ae = ConvAE(channels, kernels, T)
        self.self_expression1 = SelfExpression(self.n)
        self.self_expression2 = SelfExpression(self.n)
        self.self_expression3 = SelfExpression(self.n)

        self.Reject1 = Reject(channels[1], T[1])
        self.Reject2 = Reject(channels[2], T[2])
        self.Reject3 = Reject(channels[3], T[3])

        self.Sample = Sample(3, 40)
        self.SCM = SCM(3, 1)
        self.SAS = SAS(num_sample, 2)

        self.conv2d = nn.Conv2d(9, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        Z1 = self.ae.encoder1(x)
        Z2 = self.ae.encoder2(Z1)
        Z3 = self.ae.encoder3(Z2)

        Z1 = self.Reject1(Z1)
        shape = Z1.shape
        Z1 = Z1.view(self.n, -1)  # shape=[n, d]
        Z_recon_1 = self.self_expression1(Z1)  # shape=[n, d]
        Z1_1 = Z_recon_1.view(shape)

        Z2 = self.Reject2(Z2)
        shape = Z2.shape
        Z2 = Z2.view(self.n, -1)  # shape=[n, d]
        Z_recon_2 = self.self_expression2(Z2)  # shape=[n, d]
        Z2_1 = Z_recon_2.view(shape)

        Z3 = self.Reject3(Z3)
        shape = Z3.shape
        Z3 = Z3.view(self.n, -1)  # shape=[n, d]
        Z_recon_3 = self.self_expression3(Z3)  # shape=[n, d]
        Z3_1 = Z_recon_3.view(shape)

        x_recon = self.ae.decoder1(self.ae.decoder2(self.ae.decoder3(Z3_1) + Z2_1) + Z1_1)

        return x_recon, Z1, Z_recon_1, Z2, Z_recon_2, Z3, Z_recon_3


    def loss_fn(self, x,  x_recon, Z1, Z_recon_1, Z2, Z_recon_2, Z3, Z_recon_3, weight_coef, weight_selfExp):
        loss_ae = F.mse_loss(x_recon, x, reduction='sum')

        loss_coef1 = torch.sum(torch.pow(self.self_expression1.Coefficient, 2))
        loss_coef2 = torch.sum(torch.pow(self.self_expression2.Coefficient, 2))
        loss_coef3 = torch.sum(torch.pow(self.self_expression3.Coefficient, 2))
        # loss_coef = torch.sum(torch.abs(self.self_expression.Coefficient))

        loss_selfExp1 = F.mse_loss(Z_recon_1, Z1, reduction='sum')
        loss_selfExp2 = F.mse_loss(Z_recon_2, Z2, reduction='sum')
        loss_selfExp3 = F.mse_loss(Z_recon_3, Z3, reduction='sum')

        loss = loss_ae + weight_coef * (loss_coef1 + loss_coef2 + loss_coef3) + weight_selfExp * (loss_selfExp1 + loss_selfExp2 + loss_selfExp3)

        return loss

    def get_C(self):

        C1 = self.self_expression1.Coefficient
        C2 = self.self_expression2.Coefficient
        C3 = self.self_expression3.Coefficient

        C = torch.stack([C1, C2, C3])
        C = torch.unsqueeze(C, dim=0)
        C = C.cuda()

        C11 = self.Sample(C)
        C22 = self.SCM(C)
        C33 = self.SAS(C)

        C = torch.cat([C11,C22,C33], dim=1)
        C = self.conv2d(C)
        C = torch.squeeze(C)
        C = C.detach().to('cpu').numpy()
        return C

    def save_metrics_to_csv(slef, file_name, metrics):
        df = pd.DataFrame(metrics, columns=['epoch', 'loss', 'acc', 'nmi', 'pur'])
        df.to_csv(file_name, index=False)


def train(model,  # type: DSCNet
          x, y, epochs, lr=1e-3, weight_coef=1.0, weight_selfExp=150, device='cuda', alpha=0.04, dim_subspace=12, ro=8, show=10):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32, device=device)
    x = x.to(device)
    if isinstance(y, torch.Tensor):
        y = y.to('cpu').numpy()
    K = len(np.unique(y))
    metrics = []
    min_acc = 0
    for epoch in range(1, epochs + 1):
        x_recon, Z1, Z_recon_1, Z2, Z_recon_2, Z3, Z_recon_3 = model(x)
        loss = model.loss_fn(x, x_recon, Z1, Z_recon_1, Z2, Z_recon_2, Z3, Z_recon_3, weight_coef=weight_coef, weight_selfExp=weight_selfExp)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % show == 0 or epoch == epochs - 1:
            C = model.get_C()
            y_pred = spectral_clustering(C, K, dim_subspace, alpha, ro)
            print('Epoch %02d: loss=%.4f, acc=%.4f, nmi=%.4f, pur=%.4f' %
                  (epoch, loss.item() / y_pred.shape[0], acc(y, y_pred), nmi(y, y_pred), purity(y, y_pred)))
            metrics.append([epoch, round(float(loss.item() / y_pred.shape[0]), 4), round(acc(y, y_pred), 4), round(nmi(y, y_pred), 4), round(purity(y, y_pred), 4)])

            # if acc(y, y_pred) > min_acc:
            #     min_acc = acc(y, y_pred)
            #     torch.save(model.state_dict(), 'H:/5/results/ORL/orl(0.2).pkl')

            # torch.save(model.state_dict(), 'H:/5/results\ORL/1\orl-%d.pkl' % epoch)
            # torch.save(model.state_dict(), 'H:/5/results\ORL/2\orl-%d.pkl' % epoch)
            # torch.save(model.state_dict(), 'H:/5/results\ORL/3\orl-%d.pkl' % epoch)

            # model.save_metrics_to_csv(os.path.join('H:/5/results/metrics', '%s-10.csv' % args.db), metrics)



if __name__ == "__main__":
    import argparse
    import warnings

    parser = argparse.ArgumentParser(description='DSCNet')
    parser.add_argument('--db', default='orl',
                        choices=['coil20','orl'])
    parser.add_argument('--show-freq', default=1, type=int)
    parser.add_argument('--ae-weights', default=None)
    parser.add_argument('--save-dir', default='results')
    args = parser.parse_args()
    print(args)
    import os

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    db = args.db

    if db == 'orl':
        # load data
        data = sio.loadmat('dataset/ORL_32x32.mat')
        # print(data.get('gnd').shape)
        x, y = data['fea'].reshape((-1, 1, 32, 32)), data['gnd']
        y = np.squeeze(y - 1)  # y in [0, 1, ..., K-1]
        num_sample = x.shape[0]

        channels = [1, 30, 30, 30]
        T = [0, 3, 3, 3]
        kernels = [3, 3, 3]
        epochs = 2000
        weight_coef = 10
        weight_selfExp = 10

        # post clustering parameters
        alpha = 0.2 # threshold of C 0.2
        dim_subspace = 3  # dimension of each subspace
        ro = 2  #
        # warnings.warn("You can uncomment line#64 in post_clustering.py to get better result for this dataset!")

        dscnet = DSCNet(num_sample=num_sample, channels=channels, kernels=kernels, T=T)
        dscnet.to(device)

        ae_state_dict = torch.load('D:/5\DSC\pretrained_weights_original/%s-700.pkl' % db)
        # ae_state_dict = torch.load('H:/5/results/ORL/orl(0.3).pkl')
        dscnet.load_state_dict(ae_state_dict)
        print("Pretrained ae weights are loaded successfully.")

        train(dscnet, x, y, epochs, weight_coef=weight_coef, weight_selfExp=weight_selfExp, alpha=alpha, dim_subspace=dim_subspace, ro=ro, show=args.show_freq, device=device)

        # torch.save(dscnet.state_dict(), args.save_dir + '/%s-/%s.pkl' % args.db,% epochs)
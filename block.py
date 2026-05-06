import torch
import torch.nn as nn

class Reject(nn.Module):
    def __init__(self, c, num_channels_to_remove):
        super(Reject, self).__init__()
        self.aag = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fcn = nn.Sequential(
            torch.nn.Linear(c, 2*c),
            nn.ReLU(),
            torch.nn.Linear(2*c, c))
        self.num_channels_to_remove = num_channels_to_remove

    def forward(self, z):
        # Adaptive average pooling
        out = self.aag(z)
        out = torch.squeeze(out)  # Remove dimensions of size 1
        out_fcn = self.fcn(out)

        # Get the indices of the smallest values in out_fcn2
        _, indices = torch.topk(torch.abs(out_fcn), self.num_channels_to_remove, largest=False)

        # Create a mask to keep only the channels not in the indices
        mask = torch.ones(out_fcn.size(), dtype=torch.bool, device=out_fcn.device)
        mask.scatter_(1, indices, False)

        # Apply mask to the input tensor z to keep or remove channels
        z_filtered = torch.masked_select(z, mask.unsqueeze(2).unsqueeze(3)).view(z.size(0), -1, z.size(2), z.size(3))

        return z_filtered


class Sample(nn.Module): #块对角BDAM
    def __init__(self, in_channels, k):
        super().__init__()
        self.Conv1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.avg = nn.AvgPool2d(k)
        self.norm = nn.Sigmoid()
        self.k = k

    def forward(self, U):
        q = self.Conv1x1(U)  # U:[bs,c,h,w] to q:[bs,1,h,w]
        q = self.avg(torch.squeeze(q, dim=0))
        q = torch.unsqueeze(self.norm(q), dim=0)
        q = q.repeat_interleave(self.k, dim=2).repeat_interleave(self.k, dim=3)
        return U * q.expand_as(U)  # 广播机制


class SCM(nn.Module):#自表示系数矩阵间的注意力IMAM
    def __init__(self, c, reduction=2):
        super(SCM, self).__init__()
        self.aag = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fcn = nn.Sequential(
            torch.nn.Linear(c, c // reduction),
            nn.ReLU(inplace=True),
            torch.nn.Linear(c // reduction, c),
            nn.Sigmoid())

    def forward(self, x):
        y = torch.squeeze(self.aag(x))
        y = self.fcn(y)
        y = torch.unsqueeze(torch.unsqueeze(y, dim=-1), dim=-1)
        return x * y.expand_as(x)

class SAS(nn.Module):#样本类别间的注意力ISAM
    def __init__(self, h, reduction=2):
        super(SAS, self).__init__()
        self.aag = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fcn = nn.Sequential(
            torch.nn.Linear(h, h // reduction),
            nn.ReLU(inplace=True),
            torch.nn.Linear(h // reduction, h),
            nn.Sigmoid())

    def forward(self, x):
        z = x.transpose(1, 2)
        y = torch.squeeze(self.aag(z))
        y = self.fcn(y)
        y = torch.unsqueeze(torch.unsqueeze(y, dim=-1), dim=-1)
        return (z * y.expand_as(z)).transpose(1, 2)

# if __name__ == "__main__":
#     bs, c, h, w = 1, 3, 20, 20
#     in_tensor = torch.randn(bs, c, h, w)
#
#     s_se = Sample(c, 5)
#     # s_se = SAS(h, 3)
#     torch.set_printoptions(profile="full")
#     print("in shape:", in_tensor)
#     out_tensor = s_se(in_tensor)
#     torch.set_printoptions(profile="full")
#     print("out shape:", out_tensor)


# if __name__ == '__main__':
#
#     # 定义输入的通道数
#     c = 5
#
#     # 创建随机输入张量，形状为 (batch_size, channels, height, width)
#     input_tensor = torch.randn(5, c, 8, 8)  # 1 个样本，16 个通道，每个通道 8x8
#
#     # 创建 Reject 模块实例
#     model = Reject(c,1)
#
#     # 前向传播，得到输出
#     output_tensor = model(input_tensor)
#
#     # 打印输入和输出张量
#     print("Input Tensor:")
#     torch.set_printoptions(profile="full")
#     print(input_tensor.shape)
#     print("Output Tensor:")
#     torch.set_printoptions(profile="full")
#     print(output_tensor.shape)
#
#     # 检查输出中的通道数量是否小于或等于输入中的通道数量
#     print("Number of channels in input:", input_tensor.shape[1])
#     print("Number of channels in output:", output_tensor.shape[1])
#     print("Channels removed:", input_tensor.shape[1] - output_tensor.shape[1])

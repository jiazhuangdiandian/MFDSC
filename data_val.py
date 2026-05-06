# import scipy.io as sio
# import matplotlib.pyplot as plt
# import numpy as np
#
# # 加载数据
# # data = sio.loadmat('dataset/YaleBCrop025.mat')
# # img = data['Y']
# # I = []
# # Label = []
# # for i in range(img.shape[2]):
# #     for j in range(img.shape[1]):
# #         temp = np.reshape(img[:, j, i], [42, 48])
# #         Label.append(i)
# #         I.append(temp)
# # I = np.array(I)
# # y_total = np.array(Label[:])
# # Img = np.transpose(I, [0, 2, 1])
# # x = np.expand_dims(Img[:], 1)
#
# data = sio.loadmat('dataset/COIL20.mat')
# # print(data.get('gnd').shape)
# x, y = data['fea'].reshape((-1, 1, 32, 32)), data['gnd']
#
# # 自定义显示参数
# n_samples = 100    # 总显示数量
# n_cols = 10        # 每行显示数量
# n_rows = n_samples // n_cols
#
# # 间距控制参数（可调节）
# width_ratio = 1   # 画布宽度利用率 (0-1)
# height_ratio = 1  # 画布高度利用率 (0-1)
# wspace = 0.02       # 横向间距（宽度比例）
# hspace = 0.02       # 纵向间距（高度比例）
#
# plt.figure(figsize=(n_cols*1.5, n_rows*1.5))  # 基础尺寸系数1.5可调
#
# for i in range(n_samples):
#     plt.subplot(n_rows, n_cols, i+1)
#     plt.imshow(x[i][0].T, cmap='gray')
#     plt.axis('off')
#
# # 精细布局调整（核心代码）
# plt.subplots_adjust(left=(1-width_ratio)/2,     # 左侧留空
#                     right=1-(1-width_ratio)/2,  # 右侧留空
#                     bottom=(1-height_ratio)/2,  # 底部留空
#                     top=1-(1-height_ratio)/2,   # 顶部留空
#                     wspace=wspace,              # 横向间距
#                     hspace=hspace)             # 纵向间距
#
# plt.show()


import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

# 加载数据
data = sio.loadmat('dataset/COIL20.mat')
x, y = data['fea'].reshape((-1, 1, 32, 32)), data['gnd'].flatten()

# 获取类别信息
unique_labels = np.unique(y)
n_classes = len(unique_labels)  # COIL20共有20个类别
samples_per_class = 10  # 每个类别显示10个样本

# 创建子图画布（20行×10列）
plt.figure(figsize=(20, 40))  # 宽度=10*2, 高度=20*2

# 遍历每个类别
for row, label in enumerate(unique_labels):
    # 获取当前类别的所有样本索引
    class_indices = np.where(y == label)[0]

    # 随机选择指定数量的样本（可改为选择前10个）
    selected = np.random.choice(class_indices, samples_per_class, replace=False)

    # 绘制当前类别的样本
    for col, idx in enumerate(selected):
        plt.subplot(n_classes, samples_per_class, row * samples_per_class + col + 1)
        plt.imshow(x[idx][0].T, cmap='gray')
        if col == 0:  # 只在第一个位置显示类别标签
            plt.ylabel(f'Class {label}', rotation=0, ha='right', va='center')
        plt.axis('off')

# 调整间距（核心参数）
plt.subplots_adjust(
    left=0.05,  # 左侧留空
    right=0.95,  # 右侧留空
    bottom=0.01,  # 底部留空
    top=0.99,  # 顶部留空
    wspace=0.02,  # 列间距（同一类别内）
    hspace=0.02  # 行间距（不同类别间）
)

plt.show()
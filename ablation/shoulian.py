import matplotlib.pyplot as plt
import pandas as pd

def min_max_normalize(loss_list):
    min_val = min(loss_list)
    max_val = max(loss_list)
    normalized = [(x - min_val) / (max_val - min_val) for x in loss_list]
    return normalized


a = pd.read_csv(r"D:\5\DSC\csv\Umist1.csv")
#
epoch = a['epoch'].tolist()
loss = min_max_normalize((a['loss']).tolist())
acc = a['acc'].tolist()
nmi = a['nmi'].tolist()
pur = a['pur'].tolist()
plt.plot(epoch, loss, 'r', label='loss')
plt.plot(epoch, acc, 'b', label='ACC')
plt.plot(epoch, nmi, 'c', label='NMI')
plt.plot(epoch, pur, 'y', label='PUR')
# plt.title('Training phase')
plt.title('fine-tuning phase')
plt.legend()
plt.show()
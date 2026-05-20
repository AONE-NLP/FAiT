import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import numpy as np

# # setting
save_path = './utils/corr_mat'
os.makedirs(save_path, exist_ok=True)
label_name = 'ECL_corr_mat'
fig_title = 'ECL'
file_path = './dataset/electricity/electricity.csv'  # 替换为您的CSV文件路径

# 1. 读取数据，从第2行第2列开始

data = pd.read_csv(file_path, header=1)  # 从第2行开始读取数据
data = data.iloc[:, 1:]  # 从第2列开始读取数据

# 2. 将数据转换为Tensor并移动到GPU
data_tensor = torch.tensor(data.values, dtype=torch.float32).to('cuda')

# 3. 计算各列之间的相关性矩阵
mean = torch.mean(data_tensor, dim=0)
data_centered = data_tensor - mean
correlation_matrix = torch.matmul(data_centered.T, data_centered) / (data_tensor.shape[0] - 1)
stddev = torch.std(data_tensor, dim=0)
correlation_matrix = correlation_matrix / torch.outer(stddev, stddev)

# 4. 将结果转移回CPU并转换为NumPy数组
correlation_matrix = correlation_matrix.cpu().numpy()

# 5. 绘制相关性矩阵的热力图（无注释）
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', square=True)
plt.xticks([])  # 去除x轴刻度
plt.yticks([])  # 去除y轴刻度
plt.title(fig_title)

# 6. 保存热力图为PDF文件
plt.savefig(os.path.join(save_path, label_name+".pdf"), format="pdf")

# 7. 保存相关性矩阵到文件
np.savetxt(os.path.join(save_path, label_name+".csv"), correlation_matrix, delimiter=',')

# 8. 保存相关性矩阵为NumPy格式
np.save(os.path.join(save_path, label_name+".npy"), correlation_matrix)

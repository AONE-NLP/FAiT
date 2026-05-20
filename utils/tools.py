import os
import shutil

# import functorch.dim
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math
import torch.nn.functional as F

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import time
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import random
import seaborn as sns

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    lr_adjust = {}
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj in ['cosine', 'card']:
        # warmup-cosine
        min_lr = 0
        warmup_epochs = 0
        lr = (min_lr + (args.learning_rate - min_lr) * 0.5 *
              (1. + math.cos(math.pi * (epoch - warmup_epochs) / (args.train_epochs - warmup_epochs))))
        lr_adjust = {epoch: lr}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, save_every_epoch=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_every_epoch = save_every_epoch

    def __call__(self, val_loss, model, path, epoch=None):
        if np.isnan(val_loss):
            self.early_stop = True
            return
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, epoch)
            self.counter = 0
            self.early_stop = False

    def save_checkpoint(self, val_loss, model, path, epoch=None):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        file_path = os.path.join(path, 'checkpoint.pth')
        torch.save(model.state_dict(), file_path)

        # output checkpoint size
        file_size = os.path.getsize(file_path)
        file_size = convert_size(file_size)
        print(f"The size of checkpoint is {file_size}.")

        # delete txt files
        delete_txt_files_in_folder(path)
        file_path = os.path.join(path, f'Epoch_{epoch}.txt')
        # Create the file with the name "epoch_{i}.txt"
        with open(file_path, 'w') as file:
            file.write(f'Current Epoch: {epoch}')
        if self.save_every_epoch:
            if epoch:
                shutil.copy(os.path.join(path, 'checkpoint.pth'), os.path.join(path, f'checkpoint_epoch_{epoch:d}'
                                                                                     f'_val_loss_{val_loss:.5f}.pth'))
            else:
                shutil.copy(os.path.join(path, 'checkpoint.pth'), os.path.join(path, f'checkpoint_val_loss_'
                                                                                     f'{val_loss:.5f}.pth'))
        self.val_loss_min = val_loss


def convert_size(size_bytes):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f}{unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f}PB"


def delete_txt_files_in_folder(path):
    # 遍历路径下的所有文件，并删除以 .txt 结尾的文件
    [os.remove(os.path.join(path, f)) for f in os.listdir(path) if f.endswith('.txt')]


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def write_into_xls(excel_name, mat, columns=None):
    file_extension = os.path.splitext(excel_name)[1]

    if file_extension != ".xls" and file_extension != ".xlsx":
        raise ValueError('excel_name is not right in write_into_xls')

    folder_name = os.path.dirname(excel_name)
    if folder_name:
        os.makedirs(folder_name, exist_ok=True)

    if isinstance(mat, np.ndarray) and mat.ndim > 2:
        mat = mat.reshape(-1, mat.shape[-1])
        mat = mat[:1000]
    if columns is not None:
        dataframe = pd.DataFrame(mat, columns=columns)
    else:
        dataframe = pd.DataFrame(mat)
    # print(dataframe)
    # print(excel_name)
    dataframe.to_excel(excel_name, index=False)


def visual(true, preds=None, name='./pic/test.pdf', imp=False):
    """
    Results visualization
    """
    folder_name = os.path.dirname(name)
    if folder_name:
        os.makedirs(folder_name, exist_ok=True)
    label2 = 'Imputation' if imp else 'Prediction'

    if not isinstance(true, np.ndarray):
        true = true.numpy()
    if not isinstance(preds, np.ndarray):
        preds = preds.numpy()

    plt.figure()
    plt.plot(true, label='Ground Truth', linestyle='--', linewidth=2)
    if preds is not None:
        plt.plot(preds, label=label2, linewidth=2)
    plt.legend()
    plt.grid(linestyle=':', color='lightgray')
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def find_most_recently_modified_subfolder(base_dir, file_name='checkpoint.pth', contain_str=''):
    most_recent_time = 0
    most_recent_folder = None
    most_recent_subfolder = None

    if isinstance(contain_str, list):
        subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and
                   os.path.isfile(os.path.join(base_dir, d, file_name)) and all([cstr in d for cstr in contain_str])]
    else:
        subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and
                   os.path.isfile(os.path.join(base_dir, d, file_name)) and contain_str in d]

    # if not subdirs:
    #     raise ValueError('No such folder found!!! ')

    for subdir in subdirs:
        folder_path = os.path.join(base_dir, subdir)
        current_time = os.path.getmtime(folder_path)

        if current_time > most_recent_time:
            most_recent_time = current_time
            most_recent_folder = folder_path
            most_recent_subfolder = subdir

    return most_recent_folder, most_recent_subfolder


def compare_prefix_before_third_underscore(str1, str2, num=3):
    if str1 is None or str2 is None:
        return False
    prefix1 = ''.join(str1.split("_", num)[:num])
    prefix2 = ''.join(str2.split("_", num)[:num])

    are_prefixes_equal = prefix1 == prefix2

    return are_prefixes_equal


def compute_gradient_norm(model):
    total_norm = 0.0
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        # elif param.requires_grad and param.grad is None:
        #     print('\t param.grad is None...')
    total_norm = total_norm ** 0.5
    return total_norm


def is_not_empty_or_nan(a):
    if isinstance(a, list):
        if not a:
            return False
        if any(isinstance(i, (float, np.float32, np.float64)) and np.isnan(i) for i in a):
            return False
    elif isinstance(a, torch.Tensor):
        if a.numel() == 0:
            return False
        if torch.isnan(a).any():
            return False
    else:
        if isinstance(a, (float, np.float32, np.float64)) and np.isnan(a):
            return False

    return True


def compute_uncert(mask, patch_len=16, temp_stride=8, temporal=True, channel_num=7, softmax=0, tau=1.0, tau2=0.5,
                   Patch_CI=True, eps=1e-5):
    # mask: [b,t,n]
    # return [b,token_num]
    mask = ~mask
    assert channel_num == mask.shape[-1]
    if temporal:
        # [b,t,n] --> [b,token_num,n,patch_len] --> [b,token_num,n]
        token_uncer_weight = mask.unfold(dimension=1, size=patch_len, step=temp_stride).sum(dim=-1)
        if Patch_CI:
            # [b*n, token_num]
            token_uncer_weight = token_uncer_weight.sum(dim=-1).repeat_interleave(repeats=channel_num, dim=0)
        else:
            # [b, token_num]
            token_uncer_weight = token_uncer_weight.sum(dim=-1)
    else:
        token_uncer_weight = mask.sum(dim=1)
    # float
    token_uncer_weight = token_uncer_weight.to(dtype=torch.float)

    tau = F.softplus(torch.tensor(tau)) if tau <= 0 else tau
    tau2 = F.softplus(torch.tensor(tau2)) if tau2 <= 0 else tau2

    if softmax > 0:
        # softmax
        token_uncer_weight = F.softmax(token_uncer_weight / tau, dim=-1)
    elif softmax == 0:
        # pow
        token_uncer_weight = F.normalize(token_uncer_weight.pow(tau2), p=1, dim=-1)
    else:
        # F.normalize
        token_uncer_weight = F.normalize(token_uncer_weight, p=1, dim=-1)

    return token_uncer_weight.clamp(min=eps)


def hier_half_token_weight(token_weight, ratio=2):
    if token_weight is None:
        return None
    # temp_token_weight_time: [b, token_num]
    B, N = token_weight.shape
    if N % ratio != 0:
        tmp = ratio - N % ratio
        token_weight = torch.cat([token_weight, token_weight[:, -tmp:]], dim=-1)
    token_weight = token_weight.reshape(B, -1, ratio).sum(dim=-1)
    return token_weight


def cosine_distance(tensor1, tensor2, keepdims=False):
    assert tensor1.shape == tensor2.shape, "Both tensors must have the same shape in cosine_distance"
    # F.cosine_similarity
    cosine_sim = F.cosine_similarity(tensor1, tensor2, dim=-1)
    # 1 - cosine_sim
    cosine_dist = 1 - cosine_sim

    if keepdims:
        return cosine_dist.unsqueeze(-1)
    else:
        return cosine_dist


def euclidean_distance(tensor1, tensor2, keepdims=False):
    assert tensor1.shape == tensor2.shape, "Both tensors must have the same shape in euclidean_distance"
    diff = tensor1 - tensor2
    squared_diff = diff ** 2
    euclidean_dist = torch.sqrt(squared_diff.sum(-1))
    if keepdims:
        return euclidean_dist.unsqueeze(-1)
    else:
        return euclidean_dist


def get_eval_feat(layer, tensor):
    # tensor: [b, l, n]
    # feat: [b, d_model]

    # [b,l,n] --> [n,b,d_model] --> [b,d_model]
    feat = layer(tensor.permute(2, 0, 1)).sum(dim=0)
    return feat


def undo_unfold(inp, length, stride, fft_flag=False):
    # [b,n,stride,period] --> [b,l,n]
    B, N, num, period = inp.shape
    if fft_flag:
        assert num == length // stride, (f'num:{num}, length:{length}, stride:{stride}. inp.shape: {inp.shape}. '
                                         f'Please check the inputs of undo_unfold().')
    else:
        assert num == (length - period) // stride + 1, 'Please check the inputs of undo_unfold().'

    if stride == period or fft_flag:
        reconstructed = inp.flatten(start_dim=2)
        return reconstructed.transpose(-1, -2)

    reconstructed = torch.zeros(B, N, length, device=inp.device)
    count_overlap = torch.zeros_like(reconstructed)

    for i in range(num):
        start = i * stride
        end = start + period
        reconstructed[:, :, start:end] += inp[:, :, i, :]
        count_overlap[:, :, start:end] += 1

    # average
    mask = count_overlap > 0
    reconstructed[mask] /= count_overlap[mask]

    return reconstructed.transpose(-1, -2)


def send_email(subject='Python Notification', body='Program complete!', to_email=r'xxx',
               from_email=r'xxxx@xxx', password='xxxxx', mail_host='xxx',
               mail_port=465):
    # Create the message

    message = MIMEMultipart()
    message['From'] = from_email
    message['To'] = to_email
    message['Subject'] = subject
    message.attach(MIMEText(body, 'plain', 'utf-8'))  # utf-8 for compatibility

    try:
        # Connect to the SMTP server using SSL (port 465)
        with smtplib.SMTP_SSL(mail_host, mail_port) as server:
            # Login and send the email
            server.login(from_email, password)
            server.send_message(message)
        print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")


def create_sub_diagonal_matrix(n, value=1, offset=0):
    if abs(offset) >= n:
        return None
    vec = torch.ones(n - abs(offset)) * value
    return torch.diag(vec, diagonal=offset)


def plot_mat(mat, str_cat='series_2D', str0='tmp', save_folder='./results'):
    if not isinstance(mat, np.ndarray):
        mat = mat.detach().cpu().numpy()
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)

    # fig, axs = plt.subplots(1, 1)
    # plt.imshow(mat, cmap='viridis', interpolation='nearest', vmin=0.0, vmax=1.0)  # viridis  hot
    # plt.colorbar()

    plt.figure(figsize=(8, 8))
    sns.heatmap(mat, annot=False, cmap='coolwarm', square=True, cbar=True)
    plt.xticks([])  # 去除x轴刻度
    plt.yticks([])  # 去除y轴刻度
    timestamp = time.strftime("%Y%m%d_%H_%M_%S", time.localtime())
    plt.savefig(os.path.join(save_folder, f'{str_cat}_{str0}-{timestamp}.pdf'))
    plt.show()
    # save to excel
    excel_name = os.path.join(save_folder, f'{str_cat}_{str0}-{timestamp}.xlsx')
    write_into_xls(excel_name, mat)
    # save to npy
    np.save(os.path.join(save_folder, f'{str_cat}_{str0}-{timestamp}.npy'), mat)


def create_sin_pos_embed(max_len, d_model):
    pe = torch.zeros(max_len, d_model).float()

    position = torch.arange(0, max_len).float().unsqueeze(1)
    div_term = (torch.arange(0, d_model, 2).float()
                * -(math.log(10000.0) / d_model)).exp()

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    pe = pe.unsqueeze(0)

    #  [1, max_len, d_model]
    return pe


def var2tuple2(x, num=2):
    num = int(num)
    if isinstance(x, tuple):
        if len(x) == num:
            return x
        elif len(x) > num:
            return x[:num]
        else:
            return x + (x[-1],) * (num - len(x))
    return (x,) * num


def compute_weights(alpha, length, stages=None, multiple_flag=True):
    assert alpha <= 0
    if alpha == 0:
        weights = torch.ones(length)
        return weights
    stage_num = 1
    rem = 0
    if stages is not None:
        # assert (length + 1) % stages == 0 or length % stages == 0
        stage_num = (length + 1) // stages
        rem = length + 1 - stage_num * stages

    weights = torch.tensor([i ** alpha for i in range(length + 1, 0, -1)])
    # weights2 = torch.tensor([i ** (alpha / 2) for i in range(length + 1, 0, -1)])

    # iTransformer
    if multiple_flag and stages is not None:
        # on SDA now
        slices = list(range(stage_num - 1, length, stage_num))
        if rem > 0:
            slices = [a + i + 1 if i < rem else a + rem for i, a in enumerate(slices)]
        weights[slices] = torch.minimum(weights[slices] * 1.5, weights[-2])
        # weights[slices] = weights2[slices]

    # remove the first element
    weights = weights[:length]

    return weights


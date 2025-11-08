import numpy as np
import torch
import matplotlib.pyplot as plt
import shutil
from utils.metrics import results_evaluation
from tqdm import tqdm
import os
import pickle
import seaborn as sns

plt.switch_backend('agg')


def save_config(config, filepath):
    # 打印文件路径进行检查
    print(f"Saving config to: {filepath}")

    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(filepath, 'wb') as f:
        pickle.dump(config, f)


def load_config(filepath):
    with open(filepath, 'rb') as f:
        config = pickle.load(f)
    return config


def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True,accelerator=None):
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'PEMS':
        lr_adjust = {epoch: args.learning_rate * (0.95 ** (epoch // 1))}
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout:
            if accelerator is not None:
                accelerator.print('Updating learning rate to {}'.format(lr))
            else:
                print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self,accelerator=None, patience=7, verbose=False, delta=0, save_mode=True):
        self.accelerator = accelerator
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.save_mode = save_mode

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            if self.accelerator is not None:
                self.accelerator.print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
                torch.save(model.state_dict(), path + '/' + 'checkpoint')
            else:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
                torch.save(model.state_dict(), path + '/' + 'checkpoint')
        self.val_loss_min = val_loss


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


def del_files(dir_path):
    shutil.rmtree(dir_path)


def load_content(args):
    if 'ETT' in args.data:
        file = 'ETT'
    else:
        file = args.data
    if args.target == 'Global_horizontal_irradiance':
        with open('./dataset/prompt_bank/sr/{0}.txt'.format(file), 'r', encoding='utf-8') as f:
            content = f.read()
    elif args.target == 'Price':
        with open('./dataset/prompt_bank/price/{0}.txt'.format(file), 'r', encoding='utf-8') as f:
            content = f.read()
    else:
        with open('./dataset/prompt_bank/{0}.txt'.format(file), 'r', encoding='utf-8') as f:
            content = f.read()
    return content


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def heatmap(data,output_file):
    data_coor = data.corr()
    mask = np.zeros_like(data_coor, dtype=bool)
    mask[np.triu_indices_from(mask)] = False
    print(data_coor)
    data_coor.to_csv(output_file+'.csv')
    plt.rcParams.update({'font.size': 8})
    plt.subplots(figsize=(18, 22), dpi=1080, facecolor='w')
    fig = sns.heatmap(data_coor, annot=True, mask=mask, vmin=-1, vmax=1, square=True, cmap="viridis", fmt='.2f', annot_kws={"size": 8},
                      cbar_kws={'shrink': 0.85, 'aspect': 13})
    plt.savefig('{}.png'.format(output_file))
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.show()


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)

if __name__ == '__main__':
    path = r'D:\Time-LLM-main\results\noforecast\1_DLinear_aircon_ftM_sl6_ll6_pl36_sd20_td5_dm32_nh8_el4_dl4_df64_fc3_dropout0.1_ebtimeF_test_0_scale\checkpoints\configs.pkl'
    configs = load_config(path)
    print(configs)
    path = r'D:\Time-LLM-main\results\1_DLinear_aircon_ftM_sl6_ll6_pl36_sd20_td5_dm32_df64_nh8_el4_dl4_ma13_fc3_dropout0.1_ebtimeF_test_0_scale\checkpoints\configs.pkl'
    configs = load_config(path)
    print(configs)
import os
import sys

# from sklearn.utils import gen_even_slices
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
import math
from torch.utils.data import Dataset, Sampler

import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
import numpy as np
import torch
import shutil
import logging
import json
from functools import reduce
from pathlib import Path
import shutil

matplotlib.use('Agg')

def scaler_inverse(scaler, data):
    assert len(data.shape) == 2
    _data = scaler.inverse_transform(data)
    return 

def class_exist(str):
    _exist = False
    try:
        _exist = reduce(getattr, str.split("."), sys.modules[__name__])
    except:
        pass
    
    return _exist

def toTorch(train_input, train_target, test_input, test_target):
    train_input = torch.from_numpy(
        train_input).float()
    train_target = torch.from_numpy(
        train_target).float()
    # --
    test_input = torch.from_numpy(
        test_input).float()
    test_target = torch.from_numpy(
        test_target).float()
    return train_input, train_target, test_input, test_target


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    # dataset = np.insert(dataset, [0] * look_back, 0)
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    dataY = np.array(dataY)
    dataY = np.reshape(dataY, (dataY.shape[0], 1))
    dataset = np.concatenate((dataX, dataY), axis=1)
    return dataset


class scaled_Dataset(Dataset):
    '''
    Packing the input x_data and label_data to torch.dataset
    '''

    def __init__(self, x_data, label_data):
        self.data = x_data.copy()
        self.label = label_data.copy()
        self.samples = self.data.shape[0]
        # logger.info(f'samples: {self.samples}')

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        return (self.data[index, :, :], self.label[index])


class multiClass_Dataset(Dataset):
    '''
    only support multiple class
    '''

    def __init__(self, x_data, label_data, v_data):
        self.data = x_data.copy()
        self.label = label_data.copy()
        self.v = v_data.copy()
        self.test_len = self.data.shape[0]
        # logger.info(f'test_len: {self.test_len}')

    def __len__(self):
        return self.test_len

    def __getitem__(self, index):
        return (self.data[index,:,:-1],int(self.data[index,0,-1]),self.v[index],self.label[index])

def set_logger(log_path, log_name, level = 20, rewrite = True):
    '''Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `task_dir/train.log`.
    Example:
    logging.info('Starting training...')
    Args:
        log_path: (string) where to log
    '''
    
    logger = logging.Logger(log_name)
    if os.path.exists(log_path) and rewrite:
        os.remove(log_path) # os.remove can only delete a file with given file_path; os.rmdir() can delete a directory.
    log_file = Path(log_path)
    log_folder = log_file.parent
    os_makedirs(log_folder)
    log_file.touch(exist_ok=True)


    if level == 50:
        logger.setLevel(logging.CRITICAL)
    else:
        logger.setLevel(logging.INFO)

    fmt = logging.Formatter('[%(asctime)s] %(name)s: %(message)s', '%Y-%m-%d %H:%M:%S')

    class TqdmHandler(logging.StreamHandler):
        def __init__(self, formatter):
            logging.StreamHandler.__init__(self)
            self.setFormatter(formatter)

        def emit(self, record):
            msg = self.format(record)
            tqdm.write(msg)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(TqdmHandler(fmt))

    return logger

def save_checkpoint(state, is_best, epoch, checkpoint, ins_name=-1):
    '''Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
        ins_name: (int) instance index
    '''
    if ins_name == -1:
        filepath = os.path.join(checkpoint, f'epoch_{epoch}.pth.tar')
    else:
        filepath = os.path.join(
            checkpoint, f'epoch_{epoch}_ins_{ins_name}.pth.tar')
    if not os.path.exists(checkpoint):
        print(
            f'Checkpoint Directory does not exist! Making directory {checkpoint}')
        os.mkdir(checkpoint)
    torch.save(state, filepath)
    # logger.info(f'Checkpoint saved to {filepath}')
    print('Checkpoint saved to {}'.format(filepath))
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))
        # logger.info('Best checkpoint copied to best.pth.tar')
        print('Best checkpoint copied to best.pth.tar')


def savebest_checkpoint(state, checkpoint):
    '''Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        checkpoint: (string) folder where parameters are to be saved
    '''
    filepath = os.path.join(checkpoint, 'best.cv{}.pth.tar'.format(state['cv']))
    torch.save(state, filepath)
    # logger.info(f'Checkpoint saved to {filepath}')


def load_checkpoint(checkpoint, model, optimizer=None):
    '''Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
        gpu: which gpu to use
    '''
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"File doesn't exist {checkpoint}")
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint, map_location='cuda')
    else:
        checkpoint = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    if 'epoch' in checkpoint:
        model.epochs -= checkpoint['epoch'] + 1
        if model.epochs < 0:
            model.epochs = 0

    return checkpoint


def save_dict_to_json(d, json_path):
    '''Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    '''
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def plot_all_epoch(variable, save_name, location='./figures/'):
    num_samples = variable.shape[0]
    x = np.arange(start=1, stop=num_samples + 1)
    f = plt.figure()
    plt.plot(x, variable[:num_samples])
    f.savefig(os.path.join(location, save_name + '_summary.png'))
    plt.close()

def plot_xfit(fit_info, save_name, location='./figures/'):
    tloss,vloss = np.array(fit_info.loss_list), np.array(fit_info.vloss_list)
    num_samples = tloss.shape[0]
    x = np.arange(start=1, stop=num_samples + 1)
    f = plt.figure()
    plt.plot(x, tloss[:num_samples], label='Training')
    plt.plot(x, vloss[:num_samples], label='Validation')
    plt.legend()
    f.savefig(os.path.join(location, save_name + '.xfit.png'))
    np.save(os.path.join(location, save_name) + '.loss', (tloss, vloss))
    plt.close()

def plot_xfit_agent(fit_info, save_name, location='./figures_agent/'):
    agentP,v_agentP = np.array(fit_info.agentP), np.array(fit_info.v_agentP)
    num_samples = agentP.shape[0]
    x = np.arange(start=1, stop=num_samples + 1)
    for i in range(agentP.shape[1]):
        f = plt.figure()
        plt.plot(x, agentP[:num_samples,i,0], label='Auto_P')
        plt.plot(x, agentP[:num_samples,i,1], label='MSVR_P')
        if agentP.shape[2] == 3:
            plt.plot(x, agentP[:num_samples,i,2], label='MLP_P')
        plt.xlabel('Epoch')
        plt.ylabel('Percent_Train_H{}'.format(i))
        plt.legend()
        f.savefig(os.path.join(location, save_name + 'Training_H'+ str(i) +'.xfit.png'))
        plt.close()
        
        f = plt.figure()
        plt.plot(x, v_agentP[:num_samples,i,0], label='Auto_P')
        plt.plot(x, v_agentP[:num_samples,i,1], label='MSVR_P')
        if agentP.shape[2] == 3:
            plt.plot(x, v_agentP[:num_samples,i,2], label='MLP_P')
        plt.xlabel('Epoch')
        plt.ylabel('Percent_Valid_H{}'.format(i))
        plt.legend()
        f.savefig(os.path.join(location, save_name + 'Validation_H'+ str(i) +'.xfit.png'))
        plt.close()
    np.save(os.path.join(location, save_name) + '.S2SLoss', (fit_info.agent_s2sloss,fit_info.agent_s2svloss))
    np.save(os.path.join(location, save_name) + '.Precent', (agentP,v_agentP))
        
def plot_hError(H_error, metrics, cid = 0, location = './figures/'):
    
    os_makedirs(location)
    x = np.arange(start=1, stop=H_error.shape[0] + 1)
    
    for i, m in enumerate(metrics):
        f = plt.figure()
        plt.plot(x, H_error[:, i], label = m)
        plt.legend()
        
        # x_range = np.arange(1, horizon + 1)
        # plt.xticks(x_range)
        f.savefig(os.path.join(location, '{}.cv{}.step.error.png'.format(m, cid)))
        plt.close()
        # np.save(os.path.join(location, m + '.step.error'), H_error[:,i])

def plot_eight_windows(plot_dir,
                       predict_values,
                       predict_sigma,
                       labels,
                       window_size,
                       predict_start,
                       plot_num,
                       plot_metrics,
                       sampling=False):

    x = np.arange(window_size)
    f = plt.figure(figsize=(8, 42), constrained_layout=True)
    nrows = 21
    ncols = 1
    ax = f.subplots(nrows, ncols)

    for k in range(nrows):
        if k == 10:
            ax[k].plot(x, x, color='g')
            ax[k].plot(x, x[::-1], color='g')
            ax[k].set_title('This separates top 10 and bottom 90', fontsize=10)
            continue
        m = k if k < 10 else k - 1
        ax[k].plot(x, predict_values[m], color='b')
        ax[k].fill_between(x[predict_start:], predict_values[m, predict_start:] - 2 * predict_sigma[m, predict_start:],
                           predict_values[m, predict_start:] + 2 * predict_sigma[m, predict_start:], color='blue',
                           alpha=0.2)
        ax[k].plot(x, labels[m, :], color='r')
        ax[k].axvline(predict_start, color='g', linestyle='dashed')

        #metrics = utils.final_metrics_({_k: [_i[k] for _i in _v] for _k, _v in plot_metrics.items()})

        plot_metrics_str = f'ND: {plot_metrics["ND"][m]: .3f} ' \
            f'RMSE: {plot_metrics["RMSE"][m]: .3f}'
        if sampling:
            plot_metrics_str += f' rou90: {plot_metrics["rou90"][m]: .3f} ' \
                                f'rou50: {plot_metrics["rou50"][m]: .3f}'

        ax[k].set_title(plot_metrics_str, fontsize=10)

    f.savefig(os.path.join(plot_dir, str(plot_num) + '.png'))
    plt.close()


def de_scale(params, pred):
    _pred = params.scaler.inverse_transform(pred)
    return _pred

def os_makedirs(folder_path):
    try:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    except FileExistsError:
        pass

def os_rmdirs(folder_path):
    try:
        dirPath = Path(folder_path)
        if dirPath.exists() and dirPath.is_dir():
            shutil.rmtree(dirPath)
    except FileExistsError:
        pass
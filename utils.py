# -*- coding:utf-8 -*-
import logging
import os
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
from torch.backends import cudnn
from datetime import datetime

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

def _save_image_grid(images: torch.Tensor, filename: str, save_dir: str, figsize = (9, 9)) -> None:
    fig, axes = plt.subplots(3, 3, figsize=figsize)
    for ax, img in zip(axes.flat, images):
        ax.imshow(img.numpy().squeeze(), cmap='gray')
        ax.axis('off')
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def setup_environment(device: str) -> None:
    torch.cuda.set_device(device)
    cudnn.benchmark = True

def configure_logger(output_path: str) -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(LOG_FORMAT)

    file_handler = logging.FileHandler(output_path)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger

def create_output_structure(args: argparse.Namespace):
    timestamp = current_time()
    output_marker = f"Task{args.task}-{args.output_marker}-{args.combine}"
    output_folder = os.path.join("./result/", f"{timestamp}-{output_marker}")
    
    os.makedirs(output_folder, exist_ok=True)
    logger = configure_logger(os.path.join(output_folder, "training.log"))
    
    logger.info(f"Output directory: {output_folder}")
    logger.info("Configuration parameters:\n%s", vars(args))
    
    return output_folder, logger

def current_time():
    now = datetime.now()
    date_time = now.strftime("%m_%d_%Y, %H:%M:%S")
    return date_time

def to_torch(ndarray):
    if isinstance(ndarray, (int, float)):
        ndarray = torch.tensor([ndarray])  
    elif isinstance(ndarray, (tuple, list)):
        ndarray = torch.tensor(ndarray)
    elif isinstance(ndarray, np.ndarray):
        ndarray = torch.from_numpy(ndarray)
    return ndarray

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def test_cut_type(v):
    if v.lower() in ('false', 'f'):
        return False
    elif isinstance(v, (str, float, int)):
        try:
            v = float(v)
        except:
            raise argparse.ArgumentTypeError('Boolean value or ratio number expected.')
        return [v, round(1 - v, 1)]
    elif isinstance(v, (list, tuple)):
        return [v[i] for i in v]
    else:
        raise argparse.ArgumentTypeError('Boolean value or ratio number expected.')


def str2bool(v):
    if v.lower() in ('true','t'):
        return True
    elif v.lower() in ('false', 'f'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

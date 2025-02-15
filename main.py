# -*- coding: utf-8 -*-
"""
Author: Yuanye Liu
Paper: MERIT: Multi-view Evidential Learning for Reliable and Interpretable Liver Fibrosis Staging
Published in: Medical Image Analysis
Link: https://arxiv.org/abs/2405.02918
Last Modified: 2025-2-15
"""

import argparse
import os
import numpy as np
import torch
from model import MERIT
from test import Tester
from train import Trainer
from torch.utils.tensorboard import SummaryWriter
from dataset import (train_val_split, prepare_dataloaders)
from utils import current_time, str2bool, test_cut_type, create_output_structure, setup_environment

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MERIT Training Configuration")

    # Data parameters
    parser.add_argument('--dataroot', type=str, help="Path to the dataset root directory.")
    parser.add_argument('--num_classes', type=int, help="number of classes in the classification task")
    parser.add_argument('--num_local_views', type=int, help="number of local views")
    parser.add_argument('--task', type=int, help="Task number, which corresponds to different tasks in the experiment.")
    parser.add_argument('--batch_size', type=int, help="Batch size for training and evaluation.")
    parser.add_argument('--num_workers', type=int, help="Number of workers to use for data loading.")
    parser.add_argument('--window_size', type=int, help="Size of the sliding window for cropping the input images. (global view)")
    parser.add_argument('--patch_size', type=int, help="Size of the local views")
    parser.add_argument('--step_size', type=int, help="Step size for sliding window.")

    # Training parameters
    parser.add_argument('--lr', type=float, help="Init learning rate for training.")
    parser.add_argument('--epochs', type=int, help="Number of epochs to train the model.")
    parser.add_argument('--coef', type=int, help="Coefficient for controlling the weight of KL divergence in the loss function.")
    parser.add_argument('--device', type=str, help="Device to run the model on (e.g., 'cpu', 'cuda:X').")
    parser.add_argument('--pretrain', type=str2bool, nargs=2, help="Whether to use pretraining for local and global models. Provide two boolean values, one for each model.")
    parser.add_argument('--glb', type=str2bool, help="Whether to use global view (True or False).")

    # Experimental setup
    parser.add_argument('--output_marker', type=str, help="Marker to name the output directory for the experiment.")
    parser.add_argument('--epochs_per_vali', type=int, help="Number of epochs between each validation check.")
    parser.add_argument('--rand_seed', type=int, help="Random seed for reproducibility.")
    parser.add_argument('--cross_val', type=str2bool, help="Whether to perform cross-validation (True or False).")
    parser.add_argument('--train_prior', type=str2bool, help="Whether to use training prior probabilities (True or False).")
    parser.add_argument('--test_prior', type=str2bool, help="Whether to use testing prior probabilities (True or False).")
    parser.add_argument('--test_cut', type=test_cut_type, help="Test cut range to adjust test set ratio")
    parser.add_argument('--combine', type=str, choices=['hybrid', 'BCF', 'CBF'], help="Method to combine local and global views. Choose from 'hybrid', 'BCF', or 'CBF'.")

    args = parser.parse_args()

    return args

def main():
    args = parse_arguments()
    setup_environment(args.device)
    output_folder, logger = create_output_structure(args)
    writer = SummaryWriter(log_dir=os.path.join(output_folder, "tensorboard"))

    # Initialize metrics storage
    metrics = {"acc": [], "auc": [], "ece": []}
    num_folds = 4

    for fold in range(num_folds):
        logger.info(f"\n{'='*30} Processing Fold {fold+1}/{num_folds} {'='*30}")

        # Data preparation
        train_data, valid_data, test_data = train_val_split(args.dataroot, fold, num_folds, args.rand_seed, args.cross_val)
        dataloaders = prepare_dataloaders(args, train_data, valid_data, test_data)
        
        # Model initialization
        model = MERIT(
            num_classes=args.num_classes,
            num_local_views=args.num_local_views,
            in_channel=1,
            pretrain=args.pretrain,
            train_prior=dataloaders["train_prior"],
            test_prior=dataloaders["test_prior"],
            lambda_epochs=args.coef,
            glb=args.glb,
            combine=args.combine,
            window_size=args.window_size
        )
        model.to_device(args.device)
        
        # Training setup
        checkpoint_dir = os.path.join(output_folder, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        trainer = Trainer(
            model=model,
            train_loader=dataloaders["train"],
            valid_loader=dataloaders["valid"],
            args=args,
            checkpoint_dir=checkpoint_dir,
            writer=writer,
            fold=fold,
            logger=logger
        )
        trainer.train()

        # Evaluation
        fianl_model_path = os.path.join(checkpoint_dir, f"final_model_{fold}.pth")
        checkpoint = torch.load(fianl_model_path)
        model.load_state_dict(checkpoint["model"])
        tester = Tester(model, dataloaders['test'], args.device, logger, save_error_samples=True, error_save_dir=os.path.join(output_folder, "errors"))
        metirc = tester.test(global_step=checkpoint['epoch'])
        metrics["acc"].append(metirc['acc'])
        metrics["auc"].append(metirc['auc'])
        metrics["ece"].append(metirc['ece'])

        log_str = f"Test Fold:{fold} "
        log_str += " ".join([f"{k}:{v:.4f}" for k, v in metirc.items()])
        logger.info(log_str)

        del model, trainer, tester
        torch.cuda.empty_cache()

    # Final report
    logger.info("\n%s Final Results %s", "="*20, "="*20)
    for metric in metrics:
        values = metrics[metric]
        logger.info("%s: %.4f ± %.4f", metric.upper(), np.mean(values), np.std(values))

    logger.info("Training completed at %s", current_time())

if __name__ == '__main__':
    main()
# -*- coding:utf-8 -*-

import numpy as np
import torch
import os
from torch.optim.lr_scheduler import PolynomialLR
from sklearn.metrics import roc_auc_score
from utils import AverageMeter

class Trainer(object):
    def __init__(self, model, train_loader, valid_loader, args, checkpoint_dir, fold, writer, logger):
        self.model = model
        self.device = next(model.parameters()).device
        self.init_lr = args.lr
        self.epochs = args.epochs
        self.checkpoint_dir = checkpoint_dir
        self.writer = writer
        self.logger = logger
        self.fold = fold

        self.iters_per_epoch = len(train_loader)
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.init_lr)
        self.lr_scheduler = PolynomialLR(optimizer=self.optimizer, total_iters=self.epochs)
        self.epochs_per_vali = args.epochs_per_vali

        self.best_acc = 0
        self.best_auc = 0

    def train(self):
        
        for epoch in range(self.epochs):
            self.model.train()
            train_metrics = self._run_epoch(epoch, is_train=True)
            self._log_metrics(epoch, train_metrics, prefix='train')
            self.lr_scheduler.step()

            if (epoch + 1) % self.epochs_per_vali == 0:
                self.logger.info(f'Validation @ Epoch {epoch}')
                val_metrics = self.validation(epoch)
                self._log_metrics(epoch, val_metrics, prefix='val')
                self._save_checkpoint(epoch, val_metrics, desc='best')

            if epoch == self.epochs - 1:
                self._save_checkpoint(epoch, val_metrics, desc='final')
            
    def validation(self, epoch: int):
        self.model.eval()
        with torch.no_grad():
            return self._run_epoch(epoch, is_train=False)

    def _run_epoch(self, epoch: int, is_train: bool):
        data_loader = self.train_loader if is_train else self.valid_loader
        loss_meter = AverageMeter()
        all_probs, all_preds, all_labels = [], [], []

        for batch in data_loader:
            *inputs, label = batch if is_train else (*batch[:2], batch[2])
            
            inputs = [t.to(self.device) for t in inputs]
            label = label.long().to(self.device)

            alpha, loss_dict = self.model(inputs, label, epoch)
            loss = loss_dict['total']

            if is_train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Metrics calculation
            with torch.no_grad():
                probs = torch.softmax(alpha['combined'], dim=1)
                preds = probs.argmax(dim=1)
                
                loss_meter.update(loss.item(), label.size(0))
                all_probs.append(probs[:, 1].cpu().numpy())
                all_preds.append(preds.cpu().numpy())       
                all_labels.append(label.cpu().numpy().astype(np.int64))

        # Calculate epoch metrics
        acc, auc = self._calculate_metrics(all_preds, all_probs, all_labels)
        return {'loss': loss_meter.avg, 'acc': acc, 'auc': auc}

    def _calculate_metrics(self, preds: list, probs: list, labels: list):
        preds = np.concatenate(preds)
        probs = np.concatenate(probs)
        labels = np.concatenate(labels)
        acc = (preds == labels).mean() * 100
        auc = roc_auc_score(labels, probs) * 100
        return acc, auc

    def _log_metrics(self, epoch: int, metrics, prefix: str) -> None:
        log_str = f"{prefix.capitalize()} Epoch:{epoch} "
        log_str += " ".join([f"{k}:{v:.4f}" for k, v in metrics.items()])
        log_str += f" lr: {self.lr_scheduler.get_last_lr()[0]}"
        self.logger.info(log_str)

        if self.writer:
            for k, v in metrics.items():
                self.writer.add_scalar(f'{prefix}_{k}', v, epoch)

    def _save_checkpoint(self, epoch: int, metrics, desc) -> None:
        
        checkpoint = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'acc': metrics['acc'],
            'auc': metrics['auc'],
            'optimizer': self.optimizer.state_dict(),
        }
        
        if metrics['acc'] > self.best_acc:
            self.best_acc = metrics['acc']
            self.logger.info(f"Saved best model @ Epoch {epoch} with Acc {self.best_acc:.2f}%")
        elif desc == 'final':
            self.logger.info(f"Saved final model @ Epoch {epoch} with Acc {self.best_acc:.2f}%")
        else:
            return

        save_path = os.path.join(self.checkpoint_dir, f'{desc}_model_{self.fold}.pth')
        torch.save(checkpoint, save_path)
            
import numpy as np
import torch
import os
from tqdm import tqdm
from utils import _save_image_grid
from sklearn.metrics import roc_auc_score

class Tester:
    def __init__(self, model, test_loader, device, logger, save_error_samples,
                 error_save_dir):
        self.model = model
        self.test_loader = test_loader
        self.logger = logger
        self.device = device
        
        self.save_error_samples = save_error_samples
        self.error_save_dir = error_save_dir
        if save_error_samples and error_save_dir:
            os.makedirs(error_save_dir, exist_ok=True)

    def test(self, global_step):
        self.model.eval()
        all_probs, all_preds, all_labels = [], [], []
        error_samples = []

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                inputs, labels = self._process_batch(batch)
                alpha, loss_dict = self.model(inputs, labels, global_step)
                
                probs = torch.softmax(alpha['combined'], dim=1)
                preds = probs.argmax(dim=1)

                all_probs.append(probs[:, 1].cpu().numpy())
                all_preds.append(preds.cpu().numpy())       
                all_labels.append(labels.cpu().numpy().astype(np.int64))
                
                if self.save_error_samples:
                    error_samples.extend(self._collect_errors(inputs, preds, labels, batch))

        metrics = self._calculate_metrics(all_preds, all_probs, all_labels)
        if error_samples:
            self._save_error_samples(error_samples)
            
        return metrics
    
    def _process_batch(self, batch):
        *inputs, labels, _ = batch  
        return [t.to(self.device) for t in inputs], labels.long().to(self.device)
    
    def _get_predictions(self, outputs):
        combined = outputs['combined']
        probs = torch.softmax(combined, dim=1)
        preds = probs.argmax(dim=1)
        return probs, preds

    def _calculate_metrics(self, preds: list, probs: list, labels: list):
        preds = np.concatenate(preds)
        probs = np.concatenate(probs)
        labels = np.concatenate(labels)
        acc = (preds == labels).mean() * 100
        auc = roc_auc_score(labels, probs) * 100

        ece = self._calculate_ece(torch.from_numpy(probs).to(self.device), torch.from_numpy(labels).to(self.device))
        
        return {
            'acc': acc,
            'auc': auc,
            'ece': ece
        }

    def _calculate_ece(self, probs, labels, n_bins: int = 15):
        if probs.dim() == 1: # binary
            probs = probs.unsqueeze(1)  # (N, 1)
            probs = torch.cat([1 - probs, probs], dim=1)  # (N, 2)

        bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=self.device)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        confidences, predictions = torch.max(probs, dim=1) 
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=self.device)
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.float().mean()

            if prop_in_bin > 0:
                acc_in_bin = accuracies[in_bin].float().mean()
                avg_conf_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_conf_in_bin - acc_in_bin) * prop_in_bin

        return ece.item()
    def _collect_errors(self, inputs, preds, labels, batch) -> list:
        error_mask = ~preds.eq(labels)
        return [{
            'images': inputs[0][i].cpu(),  # Local views
            'info': batch[-1][i],          # Original file info
            'pred': preds[i].item(),
            'true': labels[i].item()
        } for i in torch.where(error_mask)[0]]

    def _save_error_samples(self, error_samples: list) -> None:
        """Save misclassified samples for analysis"""
        for sample in error_samples:
            _save_image_grid(
                sample['images'], 
                filename=f"error_{sample['info']}_pred{sample['pred']}_true{sample['true']}.png",
                save_dir = self.error_save_dir,
            )
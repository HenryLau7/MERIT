# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from vit_pytorch.vit_for_small_dataset import ViT
import torchvision.models as models
from utils import to_torch

class MERIT(nn.Module):
    def __init__(self, num_classes, in_channel, num_local_views, pretrain, train_prior, test_prior, lambda_epochs, glb, combine, window_size):
        super(MERIT, self).__init__()
        self.num_local_views = num_local_views
        self.in_channel = in_channel
        self.num_classes = num_classes

        assert len(train_prior) == num_classes and len(test_prior) == num_classes, "Prior probabilities must be provided for all classes"

        self.train_beta = to_torch(train_prior) * num_classes
        self.test_beta = to_torch(test_prior) * num_classes
        self.glb = glb
        self.combine = combine
        self.lambda_epochs = lambda_epochs
        local_pretrain, global_pretrain = pretrain
        self.local_net = nn.ModuleDict()
        for i in range(num_local_views):
            self.local_net[f"local_{i}"] = ResNet34(in_channel=in_channel,num_classes=num_classes,pretrain=local_pretrain)

        if self.glb:
            self.vit = ViT(
                image_size = window_size,
                patch_size = 16,
                num_classes=num_classes,
                dim=1024,
                depth=6,
                heads=16,
                mlp_dim=2048,
                channels=self.in_channel,
                dropout=0.1,
                emb_dropout=0.1
            )
            self.global_net = nn.Sequential(self.vit, nn.Softplus())

            if global_pretrain:
                self.vit.load_from(weights = 'checkpoint/vit.pth')

        
    def to_device(self, device):
        if 'cpu' in device or 'cuda' in device:
            device = device
        else:
            device = f'cuda:{device}'

        self.train_beta = self.train_beta.to(device)
        self.test_beta = self.test_beta.to(device)
        self.local_net = self.local_net.to(device)
        if self.glb:
            self.global_net = self.global_net.to(device)

    def forward(self, data, label, global_step):
        local_data, global_data = data
        evidence = self._compute_evidence(local_data, global_data)
        alpha = self._compute_alpha(evidence)
        loss = self._calculate_loss(alpha, label, global_step)
        return alpha, loss

    def _compute_evidence(self, local_data, global_data):
        evidence = {
            f"local_{i}": self.local_net[f'local_{i}'](local_data[:, i:i+1]) for i in range(self.num_local_views)
        }
        if self.glb:
            evidence["global"] = self.global_net(global_data)
        return evidence
    
    def _compute_alpha(self, evidence):
        alpha = dict()
        self.beta = self.train_beta if self.training else self.test_beta
        for v_num in range(self.num_local_views):
            alpha[f"local_{v_num}"] = evidence[f"local_{v_num}"] + self.beta
        alpha['global'] = evidence['global'] + self.beta
        alpha['combined'] = self._combine_views(alpha)
        return alpha
    
    def _calculate_loss(self, alpha, label, global_step):
        annealing_coef = min(1, global_step / self.lambda_epochs)
        loss = {"ce": 0.0, "kl": 0.0, "total": 0.0}
        
        for key, a in alpha.items():
            ce, kl = self._compute_individual_loss(label, a)
            loss["ce"] += ce
            loss["kl"] += kl
            loss["total"] += ce + annealing_coef * kl
            
        return {k: torch.mean(v) for k, v in loss.items()}
    
    def _compute_individual_loss(self, label, alpha):
        beta = self.beta.expand(alpha.shape[0], -1)
        label = F.one_hot(label, num_classes=self.num_classes)

        # Cross entropy calculation
        S = alpha.sum(dim=1, keepdim=True)
        ce = label * ( torch.digamma(S) - torch.digamma(alpha)).sum(dim=1, keepdim=True) 
    
        # KL divergence calculation
        alp = label * beta + (1 - label) * alpha
        kl = self._kl_divergence(alp, beta)
        
        return ce, kl
    
    def _kl_divergence(self, alpha, beta):
        # NOTE: beta = alpha - E = base_rate x classes

        S_alpha = torch.sum(alpha, dim=1, keepdim=True)
        S_beta = torch.sum(beta, dim=1, keepdim=True)
        lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
        lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
        dg0 = torch.digamma(S_alpha)
        dg1 = torch.digamma(alpha)
        kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
        return kl
    
    def _fuse_opinions(self, alpha1, alpha2, combine_method):
        
        strength = [alpha1.sum(dim=1, keepdim=True), alpha2.sum(dim=1, keepdim=True)]
        evidence = [a - self.beta for a in [alpha1, alpha2]]
        belief = [e / s.expand_as(e) for e, s in zip(evidence, strength)]
        uncertainty = [self.num_classes / s for s in strength]

        if combine_method == 'constraint':
            # Calculate conflict measure
            belief_product = torch.bmm(
                belief[0].unsqueeze(2),  
                belief[1].unsqueeze(1)   
            ).sum(dim=(1,2)) - (belief[0] * belief[1]).sum(dim=1)
            conflict = 1 - belief_product.unsqueeze(1)
            fused_belief = (belief[0] * belief[1] + belief[0] * uncertainty[1] + belief[1] * uncertainty[0]) / conflict
            fused_uncertainty = (uncertainty[0] * uncertainty[1]) / conflict

        elif combine_method == 'cumulative':
            combined_uncertainty = uncertainty[0] + uncertainty[1] - uncertainty[0] * uncertainty[1]
            fused_belief = (belief[0] * uncertainty[1] + belief[1] * uncertainty[0]) / combined_uncertainty
            fused_uncertainty = (uncertainty[0] * uncertainty[1]) / combined_uncertainty
        else:
            raise ValueError(f"Invalid combination method: {combine_method}")
        
        fused_strength = self.num_classes / fused_uncertainty
        
        return fused_belief * fused_strength + self.beta

        
    def _combine_views(self, alphas):
        """Combine multiple views according to specified strategy"""
        # Validate combination method
        COMBINE_STRATEGIES = {
            'hybrid': ('cumulative', 'constraint'),
            'BCF': ('constraint', 'constraint'),
            'CBF': ('cumulative', 'cumulative')
        }
        if self.combine not in COMBINE_STRATEGIES:
            raise ValueError(f"Invalid combine method: {self.combine}")
        
        local_method, global_method = COMBINE_STRATEGIES[self.combine]

        # Combine local views
        combined_alpha = alphas["local_0"]
        for i in range(1, self.num_local_views):
            combined_alpha = self._fuse_opinions(
                combined_alpha, 
                alphas[f"local_{i}"], 
                local_method
            )

        # Combine with global view if enabled
        if self.glb and "global" in alphas:
            combined_alpha = self._fuse_opinions(
                combined_alpha,
                alphas["global"],
                global_method
            )

        return combined_alpha
        
class ResNet34(nn.Module):
    def __init__(self, in_channel, num_classes, pretrain):
        super().__init__()
        
        self.preprocess = nn.Sequential(
            nn.Upsample(size=(224, 224)),
            nn.Conv2d(in_channel, 3, kernel_size=1) if in_channel != 3 else nn.Identity()
        )
        
        self.resnet = models.resnet34(pretrained=pretrain)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        
        self.output = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(x)
        return self.output(self.resnet(x))
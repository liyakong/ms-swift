# Copyright (c) ModelScope Contributors. All rights reserved.
"""A DDP-safe cross-entropy loss used when a segment contains padding rows.

The default SFT path lets the model compute its own mean loss. For a micro-batch that
consists solely of padding rows (all labels ``-100``) that mean is ``0 / 0`` and becomes
NaN. This loss uses ``reduction='none'`` (ignored positions are exactly ``0``) and
clamps the denominator, so an all-padding rank contributes a finite zero loss.
"""
import torch

from swift.loss.base import BaseLoss


class ElasticCrossEntropyLoss(BaseLoss):

    def __call__(self, outputs, labels, *, num_items_in_batch=None, loss_scale=None, **kwargs):
        from swift.trainers import per_token_loss_func
        token_loss = per_token_loss_func(outputs, labels)
        if num_items_in_batch is None:
            num_items_in_batch = (labels[:, 1:] != -100).sum()
        if torch.is_tensor(num_items_in_batch):
            denom = num_items_in_batch.clamp(min=1)
        else:
            denom = max(int(num_items_in_batch), 1)
        return token_loss.sum() / denom

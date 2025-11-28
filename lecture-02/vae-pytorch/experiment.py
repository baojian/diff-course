import os
import math
from typing import Any, Dict

from models import BaseVAE

import torch
from torch import optim
from torch import Tensor
import pytorch_lightning as pl
import torchvision.utils as vutils


class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: Dict) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, input: Tensor, **kwargs) -> Any:
        return self.model(input, **kwargs)

    # ------------------------------------------------------------------
    # Training / validation steps (NO optimizer_idx anymore)
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx: int):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        train_loss = self.model.loss_function(
            *results,
            M_N=self.params['kld_weight'],  # real_img.shape[0] / self.num_train_imgs,
            batch_idx=batch_idx,
        )

        # keep distributed logging flag if you want; it's harmless on single device
        self.log_dict(
            {key: val.item() for key, val in train_loss.items()},
            sync_dist=True,
        )

        return train_loss['loss']

    def validation_step(self, batch, batch_idx: int):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        val_loss = self.model.loss_function(
            *results,
            M_N=1.0,  # real_img.shape[0] / self.num_val_imgs,
            batch_idx=batch_idx,
        )

        self.log_dict(
            {f"val_{key}": val.item() for key, val in val_loss.items()},
            sync_dist=True,
        )

    # ------------------------------------------------------------------
    # Validation end → sample images
    # ------------------------------------------------------------------
    # on_validation_end is deprecated; use on_validation_epoch_end in PL 2.x
    def on_validation_epoch_end(self) -> None:
        self.sample_images()

    def sample_images(self):
        # Get sample reconstruction image from the test dataloader in the datamodule
        test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)

        # Reconstructions
        recons = self.model.generate(test_input, labels=test_label)
        vutils.save_image(
            recons.data,
            os.path.join(
                self.logger.log_dir,
                "Reconstructions",
                f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png",
            ),
            normalize=True,
            nrow=12,
        )

        # Samples, if implemented
        try:
            samples = self.model.sample(
                144,
                self.curr_device,
                labels=test_label,
            )
            vutils.save_image(
                samples.cpu().data,
                os.path.join(
                    self.logger.log_dir,
                    "Samples",
                    f"{self.logger.name}_Epoch_{self.current_epoch}.png",
                ),
                normalize=True,
                nrow=12,
            )
        except Warning:
            # Some models may not implement `sample`
            pass

    # ------------------------------------------------------------------
    # Optimizers / schedulers
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        optims = []
        scheds = []

        # Main optimizer
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.params['LR'],
            weight_decay=self.params['weight_decay'],
        )
        optims.append(optimizer)

        # Optional second optimizer (e.g. adversarial submodel)
        try:
            if self.params.get('LR_2', None) is not None:
                optimizer2 = optim.Adam(
                    getattr(self.model, self.params['submodel']).parameters(),
                    lr=self.params['LR_2'],
                )
                optims.append(optimizer2)
        except Exception:
            pass

        # Scheduler(s)
        try:
            if self.params.get('scheduler_gamma', None) is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(
                    optims[0],
                    gamma=self.params['scheduler_gamma'],
                )
                scheds.append(scheduler)

                # Optional second scheduler
                try:
                    if self.params.get('scheduler_gamma_2', None) is not None and len(optims) > 1:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(
                            optims[1],
                            gamma=self.params['scheduler_gamma_2'],
                        )
                        scheds.append(scheduler2)
                except Exception:
                    pass

                # (optimizers, schedulers) form is still supported in PL 2.x
                return optims, scheds
        except Exception:
            # No scheduler configured
            return optims

        # If we reach here: only optimizers, no schedulers
        return optims
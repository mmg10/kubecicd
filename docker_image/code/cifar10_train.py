# !/usr/bin/env/python3

# pylint: disable=no-member,unused-argument,arguments-differ
"""Cifar10 training module."""
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
from torch import nn
from torchvision import models
from torchmetrics import MaxMetric, MeanMetric
import timm
from typing import Any


class CIFAR10Classifier(
    pl.LightningModule
):  # pylint: disable=too-many-ancestors,too-many-instance-attributes
    """Cifar10 model class."""

    def __init__(self, **kwargs):
        """Initializes the network, optimizer and scheduler."""
        super(
            CIFAR10Classifier, self
        ).__init__()  # pylint: disable=super-with-arguments
        num_classes = 10
        # self.model_conv.fc = nn.Linear(num_ftrs, num_classes)
        self.model = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=True,
            num_classes=10,
        )
        self.criterion = torch.nn.CrossEntropyLoss()

        self.scheduler = None
        self.optimizer = None
        self.args = kwargs

        self.train_acc = Accuracy(task="multiclass", num_classes=10)
        self.val_acc = Accuracy(task="multiclass", num_classes=10)
        self.test_acc = Accuracy(task="multiclass", num_classes=10)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

        self.preds = []
        self.target = []
        self.example_input_array = torch.rand((1, 3, 224, 224))

    def forward(self, x_var):
        """Forward function."""
        out = self.model(x_var)
        return out

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def model_step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        if batch_idx == 0:
            self.reference_image = (batch[0][0]).unsqueeze(
                0
            )  # pylint: disable=attribute-defined-outside-init
            # self.reference_image.resize((1,1,28,28))
            print("\n\nREFERENCE IMAGE!!!")
            print(self.reference_image.shape)

        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True
        )

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self):
        pass


    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log(
            "val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True
        )

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log(
            "test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.preds += preds.tolist()
        self.target += targets.tolist()

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = torch.optim.Adam(
            lr=1e-3, weight_decay=0.0, params=self.parameters()
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            mode="min", factor=0.1, patience=10, optimizer=optimizer
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def makegrid(self, output, numrows):  # pylint: disable=no-self-use
        """Makes grids.

        Args:
             output : Tensor output
             numrows : num of rows.
        Returns:
             c_array : gird array
        """
        outer = torch.Tensor.cpu(output).detach()
        plt.figure(figsize=(20, 5))
        b_array = np.array([]).reshape(0, outer.shape[2])
        c_array = np.array([]).reshape(numrows * outer.shape[2], 0)
        i = 0
        j = 0
        while i < outer.shape[1]:
            img = outer[0][i]
            b_array = np.concatenate((img, b_array), axis=0)
            j += 1
            if j == numrows:
                c_array = np.concatenate((c_array, b_array), axis=1)
                b_array = np.array([]).reshape(0, outer.shape[2])
                j = 0

            i += 1
        return c_array

    def show_activations(self, x_var):
        """Showns activation
        Args:
             x_var: x variable
        """

        # logging reference image
        self.logger.experiment.add_image(
            "input", torch.Tensor.cpu(x_var[0][0]), self.current_epoch, dataformats="HW"
        )

        # logging layer 1 activations
        #out = self.net.conv1(x_var)
        #c_grid = self.makegrid(out, 4)
        #self.logger.experiment.add_image(
        #    "layer 1", c_grid, self.current_epoch, dataformats="HW"
        #)

    def on_train_epoch_end(self):
        """Training epoch end.

        Args:
             outputs: outputs of train end
        """
        self.show_activations(self.reference_image)

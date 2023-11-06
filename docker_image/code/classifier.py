# !/usr/bin/env/python3


from torch import nn
import timm

class CIFAR10CLASSIFIER(nn.Module):  # pylint: disable=too-many-ancestors
    """
    model wrapper for cifar10 classification
    """

    def __init__(self, **kwargs):
        """
        Initializes the network, optimizer and scheduler
        """
        super().__init__()
        self.model = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=True,
            num_classes=10,
        )

    def forward(self, x):  # pylint: disable=arguments-differ
        out = self.model(x)
        return out

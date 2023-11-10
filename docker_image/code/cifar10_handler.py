# !/usr/bin/env/python3
""" Cifar10 Custom Handler."""

import base64
import io
import json
import logging
import os
from abc import ABC
from base64 import b64encode
from io import BytesIO

import numpy as np
import torch
from PIL import Image
from captum.attr import IntegratedGradients, Occlusion, LayerGradCam, LayerAttribution
from captum.attr import visualization as viz
from classifier import CIFAR10CLASSIFIER
from matplotlib.colors import LinearSegmentedColormap
from torchvision import transforms
from ts.torch_handler.image_classifier import ImageClassifier

logger = logging.getLogger(__name__)


class CIFAR10Classification(ImageClassifier, ABC):
    """
    Base class for all vision handlers
    """

    def initialize(self, ctx):  # pylint: disable=arguments-differ
        """In this initialize function, the CIFAR10 trained model is loaded and
        the Integrated Gradients,occlusion and layer_gradcam Algorithm for
        Captum Explanations is initialized here.
        Args:
            ctx (context): It is a JSON Object containing information
            pertaining to the model artifacts parameters.
        """
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        print("Model dir is {}".format(model_dir))
        serialized_file = self.manifest["model"]["serializedFile"]
        model_pt_path = os.path.join(model_dir, serialized_file)
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available()
            else "cpu"
        )
        logger.info(f"Torch Version: {torch.__version__}")
        logger.info(f"Using Device: {self.device}")

        self.model = CIFAR10CLASSIFIER()
        self.model.load_state_dict(torch.load(model_pt_path))
        self.model.to(self.device)
        self.model.eval()
        self.model.zero_grad()
        logger.info("CIFAR10 model from path %s loaded successfully", model_dir)

        # Read the mapping file, index to object name
        mapping_file_path = os.path.join(model_dir, "class_mapping.json")
        if os.path.isfile(mapping_file_path):
            print("Mapping file present")
            with open(mapping_file_path) as pointer:
                self.mapping = json.load(pointer)
        else:
            print("Mapping file missing")
            logger.warning("Missing the class_mapping.json file.")

        self.initialized = True
        self.image_processing = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]
                ),
            ]
        )

    def _get_img(self, row):
        """Compat layer: normally the envelope should just return the data
        directly, but older version of KFServing envelope and
        Torchserve in general didn't have things set up right
        """

        if isinstance(row, dict):
            image = row.get("data") or row.get("body")
        else:
            image = row

        if isinstance(image, str):
            # if the image is a string of bytesarray.
            image = base64.b64decode(image)

        return image

    def preprocess(self, data):
        """The preprocess function of cifar10 program
        converts the input data to a float tensor
        Args:
            data (List): Input data from the request is in the form of a Tensor
        Returns:
            list : The preprocess function returns
            the input image as a list of float tensors.
        """
        images = []

        for row in data:
            image = self._get_img(row)

            # If the image is sent as bytesarray
            if isinstance(image, (bytearray, bytes)):
                image = Image.open(io.BytesIO(image))
                image = self.image_processing(image)
            else:
                # if the image is a list
                image = torch.FloatTensor(image)

            images.append(image)

        return torch.stack(images).to(self.device)

    def attribute_image_features(self, algorithm, data, **kwargs):
        """Calculate tensor attributions"""
        self.model.zero_grad()
        tensor_attributions = algorithm.attribute(data, target=0, **kwargs)
        return tensor_attributions

    def output_bytes(self, fig):
        """Convert image to bytes"""
        fout = BytesIO()
        fig.savefig(fout, format="png")
        fout.seek(0)
        return fout.getvalue()

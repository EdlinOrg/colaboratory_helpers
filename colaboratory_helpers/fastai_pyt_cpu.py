from fastai.core import resnet34
from fastai.transforms import tfms_from_model
from fastai.dataset import open_image

import numpy as np

import torch
from torch.autograd import Variable

class FaiCPU:
    """
    Fastai 0.7.0
    """

    def __init__(self, modelfilename, verbose=False):
        self.model = torch.load(modelfilename, map_location='cpu').eval()
        _, self.val_tfms = tfms_from_model(resnet34, 224)

        self.verbose=verbose

    def predict(self, filename):
        image = Variable(torch.Tensor(self.val_tfms(open_image(filename))[None]))
        pred = self.model(image).data.numpy()
        return (np.argmax(pred, axis=-1)[0] == 1)

    def predictscore(self, filename):
        image = Variable(torch.Tensor(self.val_tfms(open_image(filename))[None]))
        pred = self.model(image).data.numpy()
        idx = np.argmax(pred, axis=-1)[0]
        probs = np.exp(pred)  # probabilities

        if self.verbose:
            print("pred")
            print(pred)
            print("argmax")
            print(np.argmax(pred, axis=-1))
            print(pred[0][idx])
            print(probs)

        return (idx == 1, probs[0][idx])

import ntpath

from fastai.basic_train import load_learner
from fastai.vision.image import open_image

class FaiGPU:
    """
    For Fastai v1
    """
    def __init__(self, fastaimodelpkl, fp16=False, verbose=False):
        """
        :param fastaimodelpkl - full path to the pkl file, e.g. "mystuff/fastaimodel.pkl"
        :param fp16 - set it to use half precision
        """
        dirname = ntpath.dirname(fastaimodelpkl)
        filename = ntpath.basename(fastaimodelpkl)

        if fp16:
            self.learn = load_learner(dirname, file=filename).to_fp16()
        else:
            self.learn = load_learner(dirname, file=filename)
            
        self.verbose=verbose

    def predict(self, filename):
        img = open_image(filename)
        pred_class, pred_idx, outputs = self.learn.predict(img)
        return pred_class

    def predictscore(self, filename):
        img = open_image(filename)
        pred_class, pred_idx, outputs = self.learn.predict(img)

        if self.verbose:
            print("pred")
            print(pred_class)
            print("pred_idx")
            print(pred_idx)
            print(outputs)

        return (pred_class.obj, outputs[pred_idx].item())

#fastai1_pyt_gpu.FaiGPU = FaiGPU

import os

from colaboratory_helpers import colaboratory_helpers
from colaboratory_helpers.fastai_pyt_gpu import FaiGPU

def evalVisual(modelfile, posDir, negDir, extension="jpg", dirToProcess=["test"]):
    """
    Uses fastai visual classifier and presents the result

    assumes posDir / negDir contains train/valid/test subdirs

    :param modelfile:
    :param posDir: what the directory with positive labels are called
    :return:
    """

    fc = FaiGPU(modelfile)

    stats = {
        'tp': 0,
        'fp': 0,
        'tn': 0,
        'fn': 0
    }

    goon=True

    expectedLabel=True
    basedir = posDir

    while(goon):
        for subDirName in dirToProcess:

            print("Processing {}/{}".format(basedir, subDirName))
            tmpDir=basedir + "/" + subDirName + "/"

            files = os.listdir(tmpDir)

            numfiles = len(files)

            cnt=1
            for f in files:
                if cnt % 100 == 0:
                    print("Cnt {}/{}".format(cnt, numfiles))
                cnt +=1

                if extension != '' and not f.endswith(extension):
                    continue

                filename = tmpDir + f
                (visualPrediction, __) = fc.predictscore(filename)

                if expectedLabel:
                    if visualPrediction == expectedLabel:
                        stats['tp'] += 1
                    else:
                        stats['fn'] += 1
                else:
                    if visualPrediction == expectedLabel:
                        stats['tn'] += 1
                    else:
                        stats['fp'] += 1

        if expectedLabel:
            expectedLabel=False
            basedir=negDir
        else:
            goon=False

    print(" - - - - - - - - - - - - -")
    colaboratory_helpers.stats_small(stats['tp'], stats['tn'], stats['fp'], stats['fn'])

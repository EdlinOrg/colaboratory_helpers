
import os

import pickle
import pandas as pd

import matplotlib.pyplot as plt

from IPython.display import HTML

from colaboratory_helpers import colaboratory_helpers
from colaboratory_helpers.fastai_pyt_cpu import FaiCPU
from colaboratory_helpers.fastai_pyt_gpu import FaiGPU
from colaboratory_helpers.ffsfastText import MyFastTexter

import colaboratory_helpers.uicheckbox as uicheckbox

class FaiAnalyze:

    def __init__(self, modelfile, dir, expected, verbose=False, csvcprocessed=None, cb=None, gpu=True, fastTextModel=None, pk2fastTextStrFile=None):
        if gpu:
            self.fc = FaiGPU(modelfile)
        else:
            self.fc = FaiCPU(modelfile)

        self.expected=expected
        self.dir = dir
        self.cb = cb

        if csvcprocessed is not None:
            self.processedset = colaboratory_helpers.createSetFromCVSs(csvcprocessed)
        else:
            self.processedset = {}


        self.fastText=False
        if fastTextModel is not None:
            self.fastTextClassifier = MyFastTexter()
            self.fastTextClassifier.loadModel(fastTextModel)
            self.pk2fastTextStr = pickle.load( open(pk2fastTextStrFile, "rb"))
            self.fastText = True

            # Save the data for an ensemble
            self.ensembleX = []
            self.ensembleY = []

    def analyze(self, start, num, treshold=0.7, upperlimit=-1, extension='.jpeg', plot=True, interactive=True, showWrongFor=1):
        """
        showWrongFor: 1= visual, 2=text, 3=combined, 4=visualy correct classified, but score < treshold
        """

        if showWrongFor==4:
            print("Will show items correctly classified, but score is lower than {}".format(treshold))


        if interactive:
            display(HTML(uicheckbox.cssMove() + uicheckbox.cssRemove()))
            uicheckbox.init()

        files = os.listdir(self.dir)

        stats ={
            'correct': 0,
            'incorrect': 0
        }

        ret=[]


        cnt = 0
        processed=0
        for f in files:
            if extension != '' and not f.endswith(extension):
                continue

            if cnt < start:
                cnt += 1
                continue

            if f in self.processedset:
                cnt += 1
                continue

            filename=self.dir + "/" + f
            (visualPrediction, score) = self.fc.predictscore(filename)


            expectedLabel = (visualPrediction == self.expected)

            if self.fastText:
                #We assume that the fileanme is numerical
                tt = f.split(".")
                pk = int(tt[0])
                # it expects a string, we have loaded  a dictionary itemid -> str

                #(fastTextLabel, fastTextScore) = self.fastTextClassifier.predictscore(self.pk2fastTextStr[pk])


                ftResult = self.fastTextClassifier.predictprobs(self.pk2fastTextStr[pk])

                if ftResult['pos'] > ftResult['neg']:
                    fastTextLabel = 'pos'
                    fastTextScore = ftResult['pos']
                else:
                    fastTextLabel = 'neg'
                    fastTextScore = ftResult['neg']

                #if not "pos", then the score will be something like 0.34, so we convert that to a confidence
                #score that is similar to fastText.
                if not visualPrediction:
                    faiConfidencePos = score
                    faiConfidenceNeg = 1.0 - score
                else:
                    faiConfidencePos = score
                    faiConfidenceNeg = 1.0 - score

#                if fastTextLabel != 'pos':
#                    ftConfidencePos = 1.0 - score
#                    ftConfidenceNeg = score
#                else:
#                    ftConfidencePos = score
#                    ftConfidenceNeg = 1.0 - score

                ftConfidencePos = ftResult['pos']
                ftConfidenceNeg = ftResult['neg']

                textPrediction= (ftConfidencePos >= ftConfidenceNeg)

                if (faiConfidencePos + ftConfidencePos) > (faiConfidenceNeg + ftConfidenceNeg):
                    combinedPrediction=True
                else:
                    combinedPrediction=False

                if showWrongFor==2:
                    expectedLabel = (textPrediction == self.expected)
                else:
                    expectedLabel = (combinedPrediction == self.expected)

            cnt += 1

            if cnt % 50 == 0:
                print("Cnt: {}".format(cnt))

            if showWrongFor==3:
                #probs visual, text
                if self.expected:
                    tmpFtscore = ftConfidencePos
                else:
                    tmpFtscore = 1 - ftConfidencePos

                self.ensembleX.append([score, tmpFtscore])
                #prediction
                self.ensembleY.append(self.expected)

            if( (showWrongFor==1 and visualPrediction != self.expected) or \
                (showWrongFor==2 and textPrediction != self.expected) or \
                (showWrongFor==3 and combinedPrediction != self.expected) or \
                (showWrongFor==4 and visualPrediction==self.expected and score < treshold)):
#            if (upperlimit < 0 and (not expectedLabel) and treshold < score) or (upperlimit > 0 and isit == self.expected and score < upperlimit):

                if plot:
                    plt.figure()
                    colaboratory_helpers.plt_show_img(filename)
                    plt.show()

                print("Cnt: {} - {} - {}".format(cnt, visualPrediction, score))
                if self.fastText:
                    print("FastText: {} - {}".format(fastTextLabel, fastTextScore))

                print("  {}           {}".format(filename, f))
                if self.cb is not None:
                    self.cb(f)

                if interactive:
                    #checkboxMove = ui.getCheckbox(str(f))
                    #checkboxRemove = ui.getCheckbox(str(f) + "remove")
                    #display(HTML(checkboxMove._repr_html_() + " Move" + checkboxRemove._repr_html_() + " Remove"))
                    display(HTML(uicheckbox.checkboxCodeMove(f) + uicheckbox.checkboxCodeRemove(f) ))

                processed +=1

                ret.append(f)
                if processed >= num:
                    break

                stats['incorrect'] += 1

                print("  Correct: {} Incorrect: {}".format(stats['correct'], stats['incorrect']))
                display(HTML("<hr noshade>"))
            else:
                stats['correct'] += 1


        return (ret, stats)

    def printTriggers(self):
        display(HTML(uicheckbox.triggerReportCodeMove() ))
        display(HTML(uicheckbox.triggerReportCodeRemove() ))

    def printSummary(self):
       print("Want to move {} items".format(len(uicheckbox.uiMove)))
       print("Want to remove {} items".format(len(uicheckbox.uiRemove)))

    def printItemsToMove(self):
        for f in uicheckbox.uiMove:
            print(f)

    def printItemsToRemove(self):
        for f in uicheckbox.uiRemove:
            print(f)

    def appendMoversToFile(self, filename):
        colaboratory_helpers.appendArrToFile(filename, uicheckbox.uiMove)

    def appendRemoversToFile(self, filename):
        colaboratory_helpers.appendArrToFile(filename, uicheckbox.uiRemove)

    # def printItemsToMove(self):
    #     for key, value in ui.uiCheckedItemids.items():
    #         if value and not key.endswith("remove"):
    #             print(key)
    #
    # def printItemsToRemove(self):
    #     for key, value in ui.uiCheckedItemids.items():
    #         if value and key.endswith("remove"):
    #             key = key[:-5]
    #             print(key)


    def saveEnsemble(self, filename):
        ensemble = {
            'input': self.ensembleX,
            'output': self.ensembleY
        }

        if len(self.ensembleY == 0):
            print("PROBLEM: Empty array for ensembleY")
        print(self.ensembleX)

        pickle.dump(ensemble, open( filename, "wb" ) )


"""
fastai_analyze.FaiAnalyze = FaiAnalyze
"""

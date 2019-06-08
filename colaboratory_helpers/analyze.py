
import os

from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import pandas as pd

from IPython.display import HTML

from colaboratory_helpers import colaboratory_helpers

#fastai v.0.7.0
#from colaboratory_helpers.fastai_pyt_cpu import FaiCPU
#from colaboratory_helpers.fastai_pyt_gpu import FaiGPU

#fastai v.1
from colaboratory_helpers.fastai1_pyt_cpu import FaiCPU
from colaboratory_helpers.fastai1_pyt_gpu import FaiGPU


from colaboratory_helpers.ffsfastText import MyFastTexter
#from ffsfastText import MyFastTexter

import colaboratory_helpers.uicheckboxmulticlass as uicheckboxmulti
#import uicheckboxmulticlass as uicheckboxmulti

class Analyze:

    def __init__(self, expected,
                 dir=None,
                 moveLabels={},
                 modelfile=None,
                 csvcprocessed=None,
                 cb=None,
                 gpu=True,
                 fastTextModel=None,
                 multi_label=None,
                 multi_metadata=None,
                 multi_label_moversdir=None,
                 pk2fastTextStrFile=None,
                 pk2label=None
                 ):
        """
        :param expected: a string of the expected value, for fastai we cast boolean to string, so 'True' or 'False' are expected values
        :param dir: (fastai) this is the dir if we loop over images in a dir
        :param moveLabels: a dict: label -> filename to append movers to
        :param modelfile: (fastai) full path to a fastai model file
        :param csvcprocessed: list of strings; files that contain pk, i.e. entries we already processed and will ignore
        :param cb: general callback function that takes pk as argument, can do whatever you want
        :param gpu: (fastai) if it shall be processed on cpu or gpu 
        :param fastTextModel: (fasttext) the full path to the fastText model file
        :param multi_metadata: the full path to the csv file that contains meta data for a multi label item. The labels are assumed to be in a column named "label"
        :param multi_label_moversdir: the full path to the directory where to load/save the pk of the items that shall be added to a label
        :param pk2fastTextStrFile: (fasttext) pk -> string in fastText format (without label)
        """
        self.useFastai=False

        self.multi_label = multi_label
        self.multi_label_moversdir = multi_label_moversdir

        if multi_metadata is not None:
            #load the csv, process and put all unique labels into two dicts: label -> list of pk, pk -> list of labels
            df = pd.read_csv(multi_metadata)
            (self.label2pks, self.pk2labels) = self.createdict(df)

        if modelfile is not None:
            #Load fastai model
            self.useFastai = True
            if gpu:
                self.fc = FaiGPU(modelfile)
            else:
                self.fc = FaiCPU(modelfile)

        #These are the labels where we could move items to
        self.moveLabels=moveLabels.copy()

        #Remove the label we are looking at from the move list (no point moving it to itself)
        #self.moveLabels.remove(expected) # This one for array
        #For multi labels however, we still want the checkbox so we can decide if the label is kept or not
        if expected in self.moveLabels and not self.multi_label:
            del self.moveLabels[expected]

        self.expected=expected # a string
        self.dir = dir 
        self.cb = cb

        if csvcprocessed is not None:
            self.processedset = colaboratory_helpers.createSetFromCVSs(csvcprocessed)
        else:
            self.processedset = {}

        self.fastText=False

        if fastTextModel is not None:
            self.fastTextClassifier = MyFastTexter(multi_label=multi_label)
            self.fastTextClassifier.loadModel(fastTextModel)
            self.pk2fastTextStr = pickle.load( open(pk2fastTextStrFile, "rb"))
            if pk2label is not None:
                self.pk2label = pickle.load( open(pk2label, "rb"))

            self.fastText = True

            # Save the data for an ensemble
            self.ensembleX = []
            self.ensembleY = []

    def analyze(self,
                   imgCb,
                   textCb=None,
                   limit=100,
                   treshold=0.7,
                   showWrongFor=2,
                   grepText=None,
                   grepTextNot=None,
                   plot=True,
                   interactive=True,
                   extension='.jpg'
                   ):
        """
        Loop over entries in a pkl and process each
        :param imgCb callback function that returns path to image based on pk
        :param textCb callback function that takes the text string and returns a string (with html)
        :param limit number of "wrong" result that shall be displayed
        :param treshold score needs to be higher than this to be considered correctly labeled (if showWrongFor==11)
        :param showWrongFor:
            1 = visual, display all that are not getting classied as 'expected'
            11 = visually correct classified, but score < treshold
            2 = text, display all that are not getting classied as 'expected'
            20 = text, display all that gets classified as "expected", but are not suppose to be that
            3 = combined NOT IMPLEMENTED
        :param grepText a list of string that shall trigger false positive. If this is defined, the fastText prediction is disabled
        :param grepTextNot a list of strings that should NOT be present to trigger false positive, only used combined with grepText
        :param extension (fastai) the extension of the image files
        :return: (ret, stats) where ret is a list of processed PK and stats is some statistics
        """

        if showWrongFor==1:
            print("Fastai: Will show items that were not predicted as {}, although that is what we expected".format(self.expected))
        if showWrongFor==11:
            print("Fastai: Will show items correctly classified, but score is lower than {}".format(treshold))

        if showWrongFor == 2:
            print("FastText: Processing pkl, fastText prediction only, display all that are not getting classied as 'expected'")
        elif showWrongFor == 20:
            print("FastText: Processing pkl, fastText prediction, display all that gets classified as 'expected' but are not supposed to be that")

        grepTextUsed=False

        if grepText is not None:
            grepTextUsed = True
            print("DISABLED fastText and only grep for text in the training data")
            print(grepText)

        if interactive:
            #Init the uicheckboxes
            if self.multi_label:
                print("Init checkboxes multi label")
                #we need to make a list, so we can sort it
                tmplabellist = list(self.label2pks.keys())
                tmplabellist.sort()
                self.uicheckboxes = uicheckboxmulti.Uicheckboxmulti(tmplabellist)
            else:
                print("Init checkboxes normal")
                self.uicheckboxes = uicheckboxmulti.Uicheckboxmulti(self.moveLabels)

            display(HTML(self.uicheckboxes.initialCssCode()))

        stats ={
            'correct': 0,
            'incorrect': 0
        }

        ret=[]
        cnt = 0

        if self.useFastai:

            numberprocessed = 0

            #Loop over the files in a directory
            if self.dir is None:
                print("ERROR: No directory <dir> is defined")
                return

            print("Processing dir {}".format(self.dir))

            files = os.listdir(self.dir)
            print("Found {} files".format(len(files)))

            for f in files:
                if extension != '' and not f.endswith(extension):
                    continue

                if f in self.processedset:
                    continue

                numberprocessed += 1

                if numberprocessed % 1000 == 0:
                    print("Items processed {}".format(numberprocessed))

                filename = self.dir + "/" + f

                #take before ., and convert to int
                tmpf = f.split(".")
                #it might be an underscore in there as well
                tmpf = tmpf[0].split("_")

                pk = int(tmpf[0])


                (visualPrediction, score) = self.fc.predictscore(filename)
                #Cast the boolean to a string
                visualPrediction = str(visualPrediction)

                if not self.processOne(imgCb, pk, cnt, None, grepTextUsed, grepText, grepTextNot, showWrongFor, treshold, plot, interactive, visualPrediction=visualPrediction, visualScore=score):
                    cnt += 1
                    stats['incorrect'] += 1

                    print("  Correct: {} Incorrect: {}".format(stats['correct'], stats['incorrect']))
                    display(HTML("<hr noshade>"))

                    ret.append(pk)

                    if cnt >= limit:
                        break
                else:
                    stats['correct'] += 1

        else:
            if not self.multi_label:
                for pk, fastTextStr in self.pk2fastTextStr.items():
                    if not self.processOne(imgCb, pk, cnt, fastTextStr, grepTextUsed, grepText, grepTextNot, showWrongFor, treshold, plot, interactive, textCb):
                        cnt += 1
                        stats['incorrect'] += 1

                        print("  Correct: {} Incorrect: {}".format(stats['correct'], stats['incorrect']))
                        display(HTML("<hr noshade>"))

                        ret.append(pk)

                        if cnt >= limit:
                            break
                    else:
                        stats['correct'] += 1
            else:
                # MULTI LABEL, loop over the label
                for pk in self.label2pks[self.expected]:
                    fastTextStr = self.pk2fastTextStr[pk]
                    if not self.processOne(imgCb, pk, cnt, fastTextStr, grepTextUsed, grepText, grepTextNot, showWrongFor, treshold, plot, interactive, textCb):
                        cnt += 1
                        stats['incorrect'] += 1

                        print("  Correct: {} Incorrect: {}".format(stats['correct'], stats['incorrect']))
                        display(HTML("<hr noshade>"))

                        ret.append(pk)

                        if cnt >= limit:
                            break
                    else:
                        stats['correct'] += 1

        return (ret, stats)

    def processOne(self,
                    imgCb,
                    pk,
                    cnt,
                    fastTextStr,
                    grepTextUsed,
                    grepText,
                    grepTextNot,
                    showWrongFor,
                    treshold,
                    plot,
                    interactive,
                    textCb=None,
                    visualPrediction=None,
                    visualScore=None
                    ):
        """
        :param visualPrediction (fastai) the predicted label 
        :param visualScore (fastai) the predicted score
        """
        combinedPrediction='TODO'

        if grepText is not None:
            fastTextLabel = self.expected
            fastTextScore = -1
            for strToSearchFor in grepText:
                if strToSearchFor in fastTextStr:
                    triggerAlert=True
                    if grepTextNot is not None:
                        for strToSearchForNot in grepTextNot:
                            if strToSearchForNot in fastTextStr:
                                triggerAlert = False
                                break

                    if triggerAlert:
                        fastTextLabel = strToSearchFor
                        break
        else:
            if not self.useFastai:
                #multi label, returns all labels that got a score higher than the default treshold
                (fastTextLabel, fastTextScore) = self.fastTextClassifier.predictscore(fastTextStr)


        if ((showWrongFor == 1 and visualPrediction != self.expected) or \
                (showWrongFor == 2 and not self.multi_label and fastTextLabel != self.expected) or \
                (showWrongFor == 2 and self.multi_label and fastTextLabel.count(self.expected) == 0) or \
                (showWrongFor == 20 and fastTextLabel == self.expected) and self.pk2label[pk] != self.expected or \
                (showWrongFor == 3 and combinedPrediction != self.expected) or \
                (showWrongFor == 11 and visualPrediction == self.expected and visualScore < treshold)
            ):

            if str(pk) in self.processedset:
                return True

            if plot:
                plt.figure()
                colaboratory_helpers.plt_show_img(imgCb(pk))
                plt.axis('off')
                plt.show()

            if fastTextStr is not None:
                if textCb is not None:
                    display(HTML("<blockquote>" + textCb(fastTextStr) + "</blockquote"))
                else:
                    display(HTML("<blockquote>" + fastTextStr + "</blockquote"))

            print("Cnt: {}  Pk: {}".format(cnt, pk))
            if self.useFastai:
                print("Fastai: {} - {}".format(visualPrediction, visualScore))

            if self.fastText:
                if grepTextUsed:
                    print("grepText: Expected: {} text matched: {}".format(self.expected, fastTextLabel))
                else:
                    if showWrongFor == 20:
                        print("FastText: true label: '{}' - prediction: '{}' - score: {}".format(self.pk2label[pk], fastTextLabel, fastTextScore))
                    else:
                        print("FastText: expected label: '{}' - prediction: '{}' - score: {}".format(self.expected, fastTextLabel, fastTextScore))
                        if self.multi_label:
                            tmplabels = self.pk2label[pk].split(",")
                            tmplabels.sort()
                            print("FastText: true labels we defined: {}".format(tmplabels))
            if self.cb is not None:
                self.cb(pk)

            if interactive:

                if self.multi_label:
                    #pre-select the ones we have labelled now
                    display(HTML(self.uicheckboxes.allCheckboxes(pk, tmplabels)))

                    #If we want to check the one ft selected
                    #display(HTML(self.uicheckboxes.allCheckboxes(pk, fastTextLabel)))
                else:
                    display(HTML(self.uicheckboxes.allCheckboxes(pk, removeboxes=False)))

            return False

        return True

        """
    def analyze(self, start, num, treshold=0.7, upperlimit=-1, extension='.jpeg', plot=True, interactive=True, showWrongFor=1):

        
        Loop over a directory
        showWrongFor: 1= visual, 2=text, 3=combined, 4=visualy correct classified, but score < treshold
        

        if showWrongFor==4:
            print("Will show items correctly classified, but score is lower than {}".format(treshold))


        if interactive:
            display(HTML(uicheckboxmulti.init(self.moveLabels)))

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

            if self.useFastai:
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

                print("Cnt: {}".format(cnt))
                if self.useFastai:
                    print("Fastai: {} - {}".format(visualPrediction, score))
                if self.fastText:
                    print("FastText: {} - {}".format(fastTextLabel, fastTextScore))

                print("  {}           {}".format(filename, f))
                if self.cb is not None:
                    self.cb(f)

                if interactive:
                    display(HTML(uicheckboxmulti.allCheckboxes(f)))

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
        """

    def printTriggers(self):
        display(HTML(self.uicheckboxes.triggerHTMLCode() ))

    def printSummary(self):
        print(self.uicheckboxes.selectedCheckboxes)

    def appendMoversToTheirFilesMultiLabel(self):
        """
        Append the arrays to generated filenames for the selected labels
        """
        for lbl in self.uicheckboxes.uiMoveLables:
            arr = self.uicheckboxes.getMoverAsArray(lbl)

            #create filename
            filename = self.multi_label_moversdir + lbl + ".csv"

            print("Checking {} {} ".format(lbl, filename))
            if len(arr) == 0:
                continue
            if not Path(filename).is_file():
                print("File does not exist, will create it: {}".format(filename))
                colaboratory_helpers.dumpArrToFile(filename, arr)
            else:
                colaboratory_helpers.appendArrToFile(filename, arr)


    def appendRemoversToTheirFilesMultiLabel(self):
        """
        Append the arrays to generated filenames for the selected labels which should have a label REMOVED
        """
        for lbl in self.uicheckboxes.uiMoveLables:
            arr = self.uicheckboxes.getRemoveLabelAsArray(lbl)

            #create filename
            filename = self.multi_label_moversdir + lbl + ".remove.csv"

            print("Checking {} {} ".format(lbl, filename))
            if len(arr) == 0:
                continue
            if not Path(filename).is_file():
                print("File does not exist, will create it: {}".format(filename))
                colaboratory_helpers.dumpArrToFile(filename, arr)
            else:
                colaboratory_helpers.appendArrToFile(filename, arr)


    def appendMoversToTheirFiles(self):
        """
        Append the arrays to the filenames (but NOT including remove), not multi label
        """
        for lbl, filename in self.uicheckboxes.uiMoveLables.items():
            arr = self.uicheckboxes.getMoverAsArray(lbl)
            print("Checking {} {} ".format(lbl, filename))
            if len(arr) == 0:
                continue
            if not Path(filename).is_file():
                print("File does not exist, will create it: {}".format(filename))
                colaboratory_helpers.dumpArrToFile(filename, arr)
            else:
                colaboratory_helpers.appendArrToFile(filename, arr)

    def appendRemoveToFile(self, filename):
        arr = self.uicheckboxes.getRemoveAsArray()
        if len(arr) == 0:
            return

        if not Path(filename).is_file():
            print("File does not exist, will create it: {}".format(filename))
            colaboratory_helpers.dumpArrToFile(filename, arr)
        else:
            colaboratory_helpers.appendArrToFile(filename, arr)

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

    def appendAllProcessed(self, arr, filename):
        if not Path(filename).is_file():
            print("File does not exist, will create it: {}".format(filename))
            colaboratory_helpers.dumpArrToFile(filename, arr)
        else:
            colaboratory_helpers.appendArrToFile(filename, arr)

    def saveEnsemble(self, filename):
        ensemble = {
            'input': self.ensembleX,
            'output': self.ensembleY
        }

        if len(self.ensembleY == 0):
            print("PROBLEM: Empty array for ensembleY")
        print(self.ensembleX)

        pickle.dump(ensemble, open( filename, "wb" ) )

    def createdict(self, df):
        """
        return a dict of item -> list of labels
        """
        tmp = {}
        pk2labels = {}
        df['labellist'] = df['label'].str.split(',')
        for __, row in df.iterrows():
            for lbl in row['labellist']:
                if lbl not in tmp:
                    tmp[lbl] = []
                tmp[lbl].append(row['objectid'])

            pk2labels[row['objectid']] = row['labellist']
            
        return (tmp, pk2labels)

#analyze.Analyze = Analyze

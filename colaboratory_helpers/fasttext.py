import re

import pandas as pd

import fasttext

class MyFastTexter():
    """
    This is for the one install via
    pip install fasttext
    """

    def __init__(self, trainfile):
        self.trainfile = trainfile

    def trainer(self, modelfile):
        self.classifier = fasttext.supervised(self.trainfile, modelfile, epoch=25, word_ngrams=1)

    def loadModel(self, modelfile):
        #Note: we have to specify label__prefix, it doesnt default to __label__
        self.classifier = fasttext.load_model( modelfile + '.bin', encoding='utf-8',  label_prefix='__label__')

    def predictProb(self, mystr):
        labels = self.classifier.predict_proba([mystr])
        return labels

    def predict(self, mystr):
        labels = self.predictProb(mystr)
        return labels[0][0][0]

    def evaluate(self):

        self.createDictFromTrainfile()

        poslabels = self.classifier.predict(self.prepdict['pos'])

        tp = poslabels.count(["pos"])
        fn = poslabels.count(["neg"])

        neglabels = self.classifier.predict(self.prepdict['neg'])
        tn = neglabels.count(["neg"])
        fp = neglabels.count(["pos"])

        self.printStats(tp,fn, tn, fp)


    def printStats(self, tp, fn, tn, fp):
        result = self.classifier.test(self.trainfile)
        print('P@1:', result.precision)
        print('R@1:', result.recall)
        print('Number of examples in training file:', result.nexamples)

        print("- - - - ")
        print('Number of examples in our data after parsing training file: {}'.format((tp+fn+tn+fp)))

        print("TP: {}".format(tp))
        print("FN: {}".format(fn))
        print("TN: {}".format(tn))
        print("FP: {}".format(fp))

        acc = (tp + tn) / (tp + fn + tn + fp)
        prec = tp / (tp + fp)
        recall = tp / (tp + fn)

        print("Accuracy {}".format(acc))
        print("Precision {}".format(prec))
        print("Recall {}".format(recall))



    def createDictFromTrainfile(self):

        print("Load training file to generate dict {}".format(self.trainfile))
        fh = open(self.trainfile, "r")

        self.prepdict={}
        for line in fh:
            tt = line.split(' ', 1)
            if(len(tt) != 2):
                Exception("wrong split")

            tt2 = tt[0].split("__label__")
            lbl = tt2[1]
            if lbl not in self.prepdict:
                self.prepdict[lbl] = []

            self.prepdict[lbl].append(tt[1])



        print("Labels found in training file {}".format(self.prepdict.keys()))



def createTrainingFile(inputdict, headersToUse, outfile, cleanupCSVLabels=[]):
    """
    Loads the cvs file (with first line being header),
    only uses the headers specified in headersToUse

    :param inputdict: filename -> label
    :param headersToUse: array of strings
    :return:
    """

    fh = open(outfile, "w")


    prepdict={}

    for cvsfilenameWitHeader, label in inputdict.items():

        prepdict[label] = []

        print("Loading data to prepare from file %s\n" % cvsfilenameWitHeader)
        df = pd.read_csv(cvsfilenameWitHeader)

        df = df[headersToUse]

        df.fillna('', inplace=True)

        df.replace({'\n': ' '}, regex=True, inplace=True)

        for csvlabel in cleanupCSVLabels:
            df[csvlabel] = df[csvlabel].apply(lambda x: fixCommas(x))

        cnt = 1000000
        for index, row in df.iterrows():
            mystr = ''
            for lbl in headersToUse:
                if mystr != '':
                    mystr += '. '
                mystr += row[lbl]


            mystr = '__label__' + label + ' ' + cleanStr(mystr)

            prepdict[label].append(mystr)

            fh.write(mystr + "\n")
            #print(mystr)
            cnt -= 1

            if cnt <0:
                break

    fh.close()

    return prepdict



def fixCommas(mystr):
    mystr = re.sub('\s+', ',', mystr)
    mystr = re.sub(',+', ', ', mystr).strip(',')
    return mystr


def cleanStr(mystr):
    mystr = ' '.join(mystr.split())

    mystr = re.sub('\.+', '.', mystr).strip('.')
    return mystr

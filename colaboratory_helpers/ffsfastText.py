import os

import fastText

class MyFastTexter():
    """
    This one is for the version installed via
    https://github.com/facebookresearch/fastText
    """

    def setDataFile(self, trainfile):
        self.trainfile = trainfile

    def trainer(self, modelfile, epochs=25, settings=None):
        if settings is not None:
            lr = settings['lr']
            ngram = settings['ngram']
            mn = settings['mn']
            minCount = settings['minCount']
            self.classifier = fastText.train_supervised(input=self.trainfile,
                                                        epoch=epochs,
                                                        lr=lr,
                                                        wordNgrams=ngram,
                                                        minn=mn,
                                                        minCount=minCount)
        else:
            self.classifier = fastText.train_supervised(input=self.trainfile, epoch=epochs, wordNgrams=3)
        self.saveModel(modelfile)

    def saveModel(self, modelfile):
        filename = modelfile + ".bin"
        print("Saving model {}".format(filename))
        self.classifier.save_model(filename)

    def loadModel(self, modelfile, addExt=True):
        print("Loading model file {}".format(modelfile))
        if addExt:
            self.classifier = fastText.load_model( modelfile + '.bin')
        else:
            self.classifier = fastText.load_model( modelfile)

    def predictprobs(self, mystr):
        label = self.classifier.predict(mystr.strip(), k=2)

        res =  {}
        one = label[0][0].strip("__label__")
        res[one] = label[1][0]

        two = label[0][1].strip("__label__")
        res[two] = label[1][1]
        return res

    def predict(self, mystr):
        (lbl, __) = self.predictscore(mystr)
        return lbl

    def predictscore(self, mystr):
        label = self.classifier.predict(mystr.strip())

        #the bloody label contains __label__ for some reason, so we strip it
        res = label[0][0].strip("__label__")
        return (res, label[1][0])

    def predictArray(self, myarr):
        res=[]
        for entry in myarr:
            res.append(self.predict(entry))
        return res

    def trainerSetBest(self, acc, prec, recall):
        self.stat_acc_best = acc
        self.stat_prec_best = prec
        self.stat_recall_best = recall

    def trainerLoop(self, modelfile, testsetfile=None, resetBestResult=True, startFrom=None, epochs=15):

        # 15 epochs + first test split, Best so far [0.5, 3, 0], modelfile dummy_lr0.5_ngram3_minn0

        if testsetfile is not None:
            print("Using testsetfile {}".format(testsetfile))
            self.createDictFromFile(testsetfile)
        else:
            self.createDictFromTrainfile()

        #lrs = [3.0,2.5,2.0, 1.5, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        lrs = [1.3, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        #ngrams = [1, 2, 3]
        ngrams = [2, 3, 1]
        minn = [0, 1, 2, 3]
        minCounts = [4, 3, 2, 1] #default is 1

        if startFrom is not None:
            ignoreUntilStart=True
        else:
            ignoreUntilStart=False


        bestsofar = []
        bestmodelfile = ""

        if resetBestResult:
            print("Resetting best result values")
            self.trainerSetBest(-1, -1, -1)
        else:
            print("Using old best result values: {} {} {}".format(self.stat_acc_best, self.stat_prec_best, self.stat_recall_best))

        iteration=0

        totaliterations = len(lrs) * len(ngrams) * len(minn) * len(minCounts)

        print("Using {} epochs, total num of iterations {}".format(epochs, totaliterations))
        for lr in lrs:
            for ngram in ngrams:
                for mn in minn:
                    for minCount in minCounts:
                        iteration += 1

                        if ignoreUntilStart:
                            if startFrom['lr'] == lr and \
                                startFrom['ngram'] == ngram and \
                                startFrom['minn'] == mn and \
                                startFrom['minCount'] == minCount:
                                ignoreUntilStart=False
                            else:
                                #this assumes the arrays are kept in the same order!
                                continue

                        print(" - - -  - - -  - - -  - - -  - - -  - - -  - - -  - - - ")
                        print("Iteration {}/{} Trying lr={} ngram={} mn={} minCount={}".format(iteration, totaliterations, lr, ngram, mn, minCount))
                        self.classifier = fastText.train_supervised(input=self.trainfile, epoch=epochs, lr=lr, wordNgrams=ngram, minn=mn, minCount=minCount)
                        self.evaluateOnly()
                        if self.compareTrainResults():
                            bestsofar=[lr, ngram,mn]
                            print("******* Best so far {} ************************ * * * * * *".format(bestsofar))
                            self.trainerSetBest(self.stat_acc, self.stat_prec, self.stat_recall)
                            bestmodelfile=modelfile + "_iter{}_lr{}_ngram{}_minn{}_minCount{}".format(iteration, lr, ngram, mn, minCount)
                            self.saveModel(bestmodelfile)

        print("End result: Best so far {}, modelfile {}".format(bestsofar, bestmodelfile))

    def compareTrainResults(self):
        if self.stat_acc >= self.stat_acc_best:
            if self.stat_prec >= self.stat_prec_best:
                if self.stat_recall >= self.stat_recall_best:
                    return True
        return False

    def evaluate(self):
        self.createDictFromTrainfile()
        self.evaluateOnly()

    def evaluateOnly(self):

        poslabels = self.predictArray(self.prepdict['pos'])
        #print(poslabels)
        self.tp = poslabels.count("pos")
        self.fn = poslabels.count("neg")

        neglabels = self.predictArray(self.prepdict['neg'])
        self.tn = neglabels.count("neg")
        self.fp = neglabels.count("pos")

        self.printStats(self.tp, self.fn, self.tn, self.fp)


    def printStats(self, tp, fn, tn, fp):
        result = self.classifier.test(self.trainfile)
        print(result)
#        print('P@1:', result.precision)
#        print('R@1:', result.recall)
#        print('Number of examples in training file:', result.nexamples)

        print("- - - - ")
        print('Number of examples in our data after parsing training file: {}'.format((tp+fn+tn+fp)))

        print("TP: {}".format(tp))
        print("FN: {}".format(fn))
        print("TN: {}".format(tn))
        print("FP: {}".format(fp))

        self.stat_acc = (tp + tn) / (tp + fn + tn + fp)
        self.stat_prec = tp / (tp + fp)
        self.stat_recall = tp / (tp + fn)

        print("Accuracy {}".format(self.stat_acc))
        print("Precision {}".format(self.stat_prec))
        print("Recall {}".format(self.stat_recall))


    def createDictFromTrainfile(self):
        self.createDictFromFile(self.trainfile)

    def createDictFromFile(self, filename):

        print("Load training file to generate dict {}".format(filename))
        fh = open(filename, "r")

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

        print("Num of entries in dict {}".format(len(self.prepdict.values())))
        print("Labels found in training file {}".format(self.prepdict.keys()))



def compareModels(trainTestFile, mydir):
    """
    Loop over models in a directory and compare which one got the best stats
    :param trainTestFile: the file with data to evaluate upon
    :param mydir: dir to process
    """

    best_acc=-1
    best_prec=-1
    best_recall=-1
    bestfile=""

    files = os.listdir(mydir)
    for f in files:
        print(" ************************************************************************************************ ")
        modelfile = os.path.splitext(f)[0]
        mft = MyFastTexter()
        mft.setDataFile(trainTestFile)
        mft.loadModel( mydir + "/" + modelfile)
        mft.evaluate()

        if mft.stat_acc >= best_acc:
            if mft.stat_prec >= best_prec:
                if mft.stat_recall >= best_recall:
                    print("---- Best so far! -----")
                    best_acc=mft.stat_acc
                    best_prec=mft.stat_prec
                    best_recall=mft.stat_recall
                    bestfile=modelfile

    print(" ************************************************************************************************ ")
    print("Best one: {}".format(bestfile))
    print("Accuracy {}".format(best_acc))
    print("Precision {}".format(best_prec))
    print("Recall {}".format(best_recall))

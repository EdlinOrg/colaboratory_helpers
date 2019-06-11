import os

import fastText

class MyFastTexter():
    """
    This one is for the version installed via
    https://github.com/facebookresearch/fastText

    Re multi-label
    https://github.com/facebookresearch/fastText/issues/478
    """

    def __init__(self, multi_label=False, multi_treshold=0.05):
        """
        :param multilabel if we are dealing with a multilabel or not
        :param multi_treshold everything above is considered to be of that label
        """
        self.multi_label=multi_label
        self.multi_treshold=multi_treshold

        #Used when we want items to only to be classified as positive when its over a certain treshold
        self.nolabel = "**no_label**"

    def setDataFile(self, trainfile):
        self.trainfile = trainfile

    def trainer(self, modelfile, epochs=25, settings=None):

        if self.multi_label:
            loss="ova"
        else:
            loss="softmax"

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
                                                        loss=loss,
                                                        minCount=minCount)
        else:
            self.classifier = fastText.train_supervised(
                input=self.trainfile,
                epoch=epochs,
                wordNgrams=3,
                loss=loss
                )
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
        # DEPRECATED
        label = self.classifier.predict(mystr.strip(), k=2)

        #TODO: this one just gives back two, probably change to be the same as number of labels, and in order
        res =  {}
        one = label[0][0].strip("__label__")
        res[one] = label[1][0]

        two = label[0][1].strip("__label__")
        res[two] = label[1][1]
        return res

    def predict(self, mystr, treshold=None):
        (lbl, thescore) = self.predictscore(mystr)

        if treshold is not None:
            #check if its a dict
            if isinstance(treshold, dict):
                if lbl in treshold and thescore < treshold[lbl]:
                    return self.nolabel
            else:
                if thescore < treshold:
                    return self.nolabel
        return lbl

    def predictscore(self, mystr):
        """
        :return 
            normal: tuple of label, score
            for multi_label: tuple of list of labels, list of scores
                it only returns the labels where the score is higher or equal to self.multi_treshold
        """
        if self.multi_label:
            k=-1
        else:
            k=1

        label = self.classifier.predict(mystr.strip(), k=k)

        if self.multi_label:
            #print("Label {}".format(label[0]))
            #print("Label {}".format(label[1]))
            #print("Label 0 {}".format(label[0][0]))
            #print("Label 1 {}".format(label[1][0]))
            keeplbls=[]
            keepscore=[]

            if(len(label) != 1):
                Exception("wrong length of multi label return: {}".format(len(label)))

            idx=0
            numentries = len(label[0])
            while idx < numentries:
                lbl = label[0][idx]
                lbl = lbl.replace("__label__", '')
                score = label[1][idx]
                #print("lbl {}".format(lbl))
                #print("score {}".format(score))
                if score >= self.multi_treshold:
                    keeplbls.append(lbl)
                    keepscore.append(score)
                else:
                    break
                idx += 1
            return (keeplbls, keepscore)            
        else:
            #the bloody label contains __label__ for some reason, so we remove it
            res = label[0][0].replace("__label__", '')
            return (res, label[1][0])

    def predictArray(self, myarr, treshold=None):
        res=[]
        for entry in myarr:
            res.append(self.predict(entry, treshold=treshold))
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

        lrs = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
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

        if self.multi_label:
            loss="ova"
        else:
            loss="softmax"

        print("Using {} epochs, total num of iterations {}".format(epochs, totaliterations))
        for lr in lrs:
            for ngram in ngrams:
                for mn in minn:
                    for minCount in minCounts:
                        iteration += 1

                        if ignoreUntilStart:
                            if startFrom['lr'] == lr and \
                                startFrom['ngram'] == ngram and \
                                startFrom['mn'] == mn and \
                                startFrom['minCount'] == minCount:
                                ignoreUntilStart=False
                            else:
                                #this assumes the arrays are kept in the same order!
                                continue

                        print(" - - -  - - -  - - -  - - -  - - -  - - -  - - -  - - - ")
                        print("Iteration {}/{} Trying lr={} ngram={} mn={} minCount={}".format(iteration, totaliterations, lr, ngram, mn, minCount))
                        self.classifier = fastText.train_supervised(
                            input=self.trainfile,
                            epoch=epochs,
                            lr=lr,
                            wordNgrams=ngram,
                            minn=mn,
                            minCount=minCount,
                            loss=loss
                            )
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

    def evaluate(self, treshold=None):
        self.createDictFromTrainfile()
        self.evaluateOnly(treshold=treshold)

    def evaluateOnlyOld(self):

        poslabels = self.predictArray(self.prepdict['pos'])
        #print(poslabels)
        self.tp = poslabels.count("pos")
        self.fn = poslabels.count("neg")

        neglabels = self.predictArray(self.prepdict['neg'])
        self.tn = neglabels.count("neg")
        self.fp = neglabels.count("pos")

        self.printStats(self.tp, self.fn, self.tn, self.fp)

    def evaluateOnly(self, treshold=None):
        """
        :param treshold - for none multi label only: float or dict label -> float (to have different treshold for different labels)
            if defined, only the labels that got higher than the treshold are considered positive, otherwise negative,
            i.e. we give them the label self.nolabel
        """

        print("evaluateOnly")

        if treshold is not None:
            print("Using treshold {}".format(treshold))

        self.tp = 0
        self.fn = 0
        self.fp = 0
        self.tn = 0

        for label in self.prepdict.keys():

            tp=0
            fn=0
            fp=0
            tn=0

            poslabels = self.predictArray(self.prepdict[label], treshold=treshold)

            #print("Poslabels:")
            #print(poslabels)

            if self.multi_label:
                #for multilabel, poslabels will be a list of lists

                #For each list, if the label is contained, it is a TP
                for l in poslabels:
                    inlist = l.count(label) #this really can only be max 1
                    if inlist > 1:
                        Exception("more than one match")

                    tp += inlist

                    #The label was not in the list, so we say FN
                    if 0 == inlist:
                        fn += 1


                #for all other items that hasnt got this label, check for FP, TN

                #loop over all items, the items that does not have label as ground truth are the ones we process
                for itemstr, labels in self.prepdictreverse.items(): 
                    if labels.count(label) > 0:
                        #already processed this one above
                        continue

                    otherpredarr = self.predictArray([itemstr])
                    
                    #we got a list of lists back (with just one entry)
                    if len(otherpredarr) != 1:
                        Exception("Got back wrong length {}".format(otherpredarr))

                    if otherpredarr[0].count(label) > 0:
                        #we got prediction for this label, even though we shouldnt: FP
                        fp += 1
                    else:
                        #otherwise we are all good, its a TN
                        tn += 1




                """


                #for all other labels, check for false positives
                #if the other label is one of the other true labels for an item, just ignore it, we will calculate that in the loop above

                #TODO: we should check for other labels we have for this item, those are ok so should not be FP
                for otherlabel in self.prepdict.keys():
                    if otherlabel == label:
                        continue

                    olbl = self.predictArray(self.prepdict[otherlabel])
                    #olbl is a list of lists

                    tmpothers=0
                    tmpfp=0
                    idx=0
                    for l in olbl:
                        #l is a list of labels that were predicted for entry idx

                        truelabels = self.prepdictreverse[self.prepdict[label][idx]]

                        if l.count(label) > 0 and truelabels.count(label > 0):
                            # this one had the label predicted, and also has label in ground truth
                            # just ignore, since we should have calculated that as TP already
                            continue

                        if l.count(label) > 0:
                            #predicted this label, but shouldnt be, hence FP
                            tmpfp += 1
                        else:
                            tn += 1

                        idx += 1

                    fp += tmpfp
                    #we dont include the others in "tn"
                    tn += len(olbl) - tmpfp - tmpothers
                    """
            else:
                tp += poslabels.count(label)
                fn += len(poslabels) - poslabels.count(label)

                #for all other labels, check for false positives
                for otherlabel in self.prepdict.keys():
                    if otherlabel == label:
                        continue

                    olbl = self.predictArray(self.prepdict[otherlabel], treshold=treshold)
                    fp += olbl.count(label)
                    tn += len(olbl) - olbl.count(label)

            print("= " * 80)
            print("Stats for label {}".format(label))
            self.printStats(tp, fn, tn, fp)

            self.tp += tp
            self.fn += fn
            self.fp += fp
            self.tn += tn

        self.printStats(self.tp, self.fn, self.tn, self.fp)

    def printStats(self, tp, fn, tn, fp):
        print("printStats")
        result = self.classifier.test(self.trainfile)
        print(result)
        #print('P@1:', result.precision)
        #print('R@1:', result.recall)
        #print('Number of examples in training file:', result.nexamples)

        print("- - - - ")
        print('Number of examples in our data after parsing training file: {}'.format((tp+fn+tn+fp)))

        print("TP: {}".format(tp))
        print("FN: {}".format(fn))
        print("TN: {}".format(tn))
        print("FP: {}".format(fp))

        self.stat_acc = (tp + tn) / (tp + fn + tn + fp)
        if tp == 0:
            self.stat_prec = 0
            self.stat_recall = 0
        else:
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

        # label -> list of strings which have that label as true prediction
        self.prepdict={}

        # string -> list of labels as true prediction for that string
        self.prepdictreverse={}

        for line in fh:

            if self.multi_label:
                #print("parsing as multi label") #should work for single label as well
                labels = line.split("__label__")

                #The last entry should now be the the start of a label + content, the rest labels
                last = labels.pop()
                tt = last.split(' ', 1)

                tmplabels=[]

                for tmpl in labels:
                    tmpl = tmpl.strip().strip(',').strip()
                    if tmpl != '':
                        tmplabels.append(tmpl)
                        
                #add the last one we extracted above
                tmplabels.append(tt[0])

                for lbl in tmplabels:
                    if lbl not in self.prepdict:
                        self.prepdict[lbl] = []
                    self.prepdict[lbl].append(tt[1])

                #reverse string -> list of labels
                self.prepdictreverse[tt[1]] = tmplabels
            else:
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

def compareModels(trainTestFile, mydir, treshold=None):
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
        mft.evaluate(treshold=treshold)

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


#ffsfastText.MyFastTexter = MyFastTexter
#ffsfastText.compareModels = compareModels
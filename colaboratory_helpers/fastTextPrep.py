import json
import re
import pickle

import pandas as pd

def createTrainingFile(inputdict,
                       headersToUse,
                       outfile,
                       filterset=None,
                       filterReverse=False,
                       cleanupCSVLabels=[],
                       pkfield=None,
                       pkoutfile=None,
                       pkoutfilelabel=None,
                       modifyCSVentries=True,
                       duplicators=None,
                       beBrutal=False,
                       additionalPklinfo=None,
                       stopwordsfile=None):
    """
    Loads the cvs file (with first line being header),
    only uses the headers specified in headersToUse

    :param inputdict: filename -> label
    :param headersToUse: array of strings
    :param filterset: only process these keys
    :param pkoutfile: File to save pk -> string info
    :param pkoutfilelabel: File to save pk -> label
    :param additionalPklinfo: dict:
                                filename -> str|array of str
                                wantedLabels -> None|array of str
                                strategy -> 1=RAW|2=DIGIT2LETTERS  TODO: is this one necessary?
    :param stopwordsfile: path to file which contain stop words (newline separated)
    :return:
    """

    if filterset is not None and pkfield is None:
        print("Error: you must define pkfield to be the header with the pk")
        return

    if filterset is not None:
        print("Number of items in filterset {}".format(len(filterset)))
        print("filterReverse: {}".format(filterReverse))

    itemsTaken={}

    if pkfield is not None:
        print("pkfield is defined, will ignore any duplicate ids")


    if beBrutal:
        print("Will make lowercase and only keep a-z0-9")

    addInfoDict=None
    if additionalPklinfo is not None:

        if isinstance(additionalPklinfo['filename'], list):
            addInfoDict = {}
            for addfile in additionalPklinfo['filename']:
                print("Loop loading additional pkl info from file {}".format(addfile))
                addInfoDictTmp = pickle.load(open(addfile, "rb"))
                addInfoDict = {**addInfoDict, **addInfoDictTmp}
                print("Size of dict {}".format(len(addInfoDict)))
        else:
            print("Loading additional pkl info from file {}".format(additionalPklinfo['filename']))
            addInfoDict=pickle.load(open(additionalPklinfo['filename'], "rb"))

        print("Size of dict {}".format(len(addInfoDict)))
#        print("Keys in dict {}".format(addInfoDict.keys()))

        addInfoEntries= additionalPklinfo['wantedLabels']

    stopwordsdict=None
    if stopwordsfile is not None:
        stopwordsdict = {}
        with open(stopwordsfile) as f:
            for line in f:
                stopwordsdict[line.strip()] = True


    fh = open(outfile, "w")

    # pkey -> string
    pkdict = {}

    pk2label = {}

    prepdict={}

    for cvsfilenameWitHeader, label in inputdict.items():

        prepdict[label] = []

        print("Loading data to prepare from file %s\n" % cvsfilenameWitHeader)
        df = pd.read_csv(cvsfilenameWitHeader)

        #df = df[headersToUse]

        df.fillna('', inplace=True)

        df.replace({'\n': ' '}, regex=True, inplace=True)

        if modifyCSVentries:
            for csvlabel in cleanupCSVLabels:
                df[csvlabel] = df[csvlabel].apply(lambda x: fixCommas(x))

        cnt = 1000000
        for index, row in df.iterrows():

            if pkfield is not None:
                if row[pkfield] in itemsTaken:
                    if str(label) != str(row[pkfield]):
                        print("Already taken {}, ignoring. Conflicting labels! taken={}, this one={}".format(row[pkfield], itemsTaken[row[pkfield]], label))
                    continue
                itemsTaken[row[pkfield]]=label

            if filterset is not None:
                if filterReverse:
                    if row[pkfield] in filterset:
                        continue
                else:
                    if not row[pkfield] in filterset:
                        continue

            mystr = ''
            for lbl in headersToUse:
                if mystr != '':
                    mystr += '. '

                if duplicators is not None and lbl in duplicators:
                    tmpCnt = duplicators[lbl]
                    while tmpCnt > 0:
                        mystr += " " + row[lbl]
                        tmpCnt -= 1
                else:
                    mystr += row[lbl]

            if additionalPklinfo is not None:
                if addInfoDict is not None:
                    if 'jsonHeaders' in additionalPklinfo:
                        for jsonlabel in additionalPklinfo['jsonHeaders']:
                            tmpjson = parseJson(row[jsonlabel], addInfoEntries)
#                            if tmpjson != '':
#                                print("oldexif {}".format(row[pkfield]))
                            mystr += tmpjson

                    tmpjson = addAdditionalInfo(row[pkfield], addInfoDict, addInfoEntries)
#                    if tmpjson != '':
#                        print("nexeif {}".format(row[pkfield]))
                    mystr += tmpjson

            if beBrutal:
                mystr = brutal(mystr, stopwordsdict=stopwordsdict)
            else:
                mystr = cleanStr(mystr)

            if pkfield is not None:
                pkdict[row[pkfield]] = mystr
                pk2label[row[pkfield]] = label

            mystr = '__label__' + label + ' ' + mystr

            prepdict[label].append(mystr)

            fh.write(mystr + "\n")
            #print(mystr)
            cnt -= 1

            if cnt <0:
                break

    fh.close()

    if pkoutfile is not None:
        print("Saving to pkl file {}".format(pkoutfile))
        pickle.dump(pkdict, open( pkoutfile, "wb" ))
        fh.close()

    if pkoutfilelabel is not None:
        print("Saving to pk2label file {}".format(pkoutfilelabel))
        pickle.dump(pk2label, open(pkoutfilelabel, "wb"))
        fh.close()

    return prepdict

def parseJson(jsonTxt, wantedEntries):
    if jsonTxt == '':
        return ''
#    print("Check old exif\n")
#    print(jsonTxt)
    j = json.loads(jsonTxt)
    return parseDict(j, wantedEntries)

def addAdditionalInfo(pkid, addInfoDict, wantedEntries):
    #print("check new exif {}".format(pkid))
    strpkid=str(pkid)
    if not strpkid in addInfoDict:
        return ''

    return parseDict(addInfoDict[strpkid], wantedEntries)

def parseDict(myDict, wantedEntries):

    tmp = ''
    if wantedEntries is not None:
        #only take these entries, without keys
        for entry in wantedEntries:
            if entry in myDict:
                if tmp != '':
                    tmp += " "
                tmp += str(myDict[entry])
    else:
        #take all, including keys
        for entry in myDict:
            if tmp != '':
                tmp += " "

            #XXX: Debug shit
            #Exif
#            print(myDict[entry])
#            print("\n")

            tmp += entry + " " + str(myDict[entry])

    if tmp != '':
        tmp = ". " + tmp

    return tmp


def brutal(mystr, withNumbers=True, stopwordsdict=None):
    """
    Lowercase, remove any none a-z chars
    :param mystr:
    :return:
    """

    if withNumbers:
        regExpr = '[^a-z0-9]'
    else:
        regExpr = '[^a-z]'

    mystr = re.sub(regExpr, ' ', mystr.lower())
    mystr = re.sub(' +', ' ', mystr).strip()


    myl = mystr.split()

    mystr=''
    for word in myl:
        #ignore any one charater words
        if len(word) > 1:

            #ignore if just digits
            if word.isdigit():
                continue

            #ignore any words in the stop words dict
            if stopwordsdict is not None and word in stopwordsdict:
                continue

            if mystr != '':
                mystr += ' '
            mystr += word

    return mystr


def fixCommas(mystr):
    mystr = re.sub('\s+', ',', mystr)
    mystr = re.sub(',+', ', ', mystr).strip(',')
    return mystr


def cleanStr(mystr):
    mystr = ' '.join(mystr.split())

    mystr = re.sub('\.+', '.', mystr).strip('.')
    return mystr


"""
fastTextPrep.createTrainingFile = createTrainingFile
fastTextPrep.parseJson = parseJson
fastTextPrep.addAdditionalInfo = addAdditionalInfo
fastTextPrep.parseDict = parseDict
fastTextPrep.brutal = brutal
fastTextPrep.fixCommas = fixCommas
fastTextPrep.cleanStr = cleanStr
"""

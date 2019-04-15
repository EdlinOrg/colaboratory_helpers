# -*- coding: utf-8 -*-

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

import hashlib
import os
import random
import time
import urllib
import shutil
#import zipfile

import numpy as np
import pandas as pd

from pathlib import Path

import matplotlib.pyplot as plt

import cv2

from sklearn.metrics import confusion_matrix, recall_score, f1_score, precision_score, accuracy_score

import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import classification_report

def drive_getfileifneeded(filename, filenamedriveid, force=False):
    drive_load(filename, filenamedriveid, force)

def drive_load(filename, filenamedriveid, force=False):

  if force or not os.path.isfile(filename):
    print("Fetching ID {} - {}".format(filenamedriveid, filename))

    # 1. Authenticate and create the PyDrive client.
    auth.authenticate_user()
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    drive = GoogleDrive(gauth)

    downloaded = drive.CreateFile({'id': filenamedriveid})

    downloaded.GetContentFile(filename)


def drive_save(filename):
  # 1. Authenticate and create the PyDrive client.
  auth.authenticate_user()
  gauth = GoogleAuth()
  gauth.credentials = GoogleCredentials.get_application_default()
  drive = GoogleDrive(gauth)

  filenameonly = os.path.basename(filename)
  uploaded = drive.CreateFile({'title': filenameonly})
  uploaded.SetContentFile(filename)
  uploaded.Upload()
  print('Uploaded file {} with ID {}'.format(filename, uploaded.get('id')))

def drive_formatSharedLink(id):
  print("https://drive.google.com/file/d/{}/view?usp=sharing".format(id))

# ########### FILE UTILS ############


def dir_stats(mydir, prefix=" - - \n"):
    print(prefix + "{} {} files".format(mydir, len(os.listdir(mydir))))

def moveFiles(source, dest1, limit=-1, extension='', wipeAndMakeDest=False):

    if wipeAndMakeDest:
        shutil.rmtree(dest1, ignore_errors=True)
        os.mkdir(dest1)

    files = os.listdir(source)

    print(" * *  * *  * *  * *  * *  * *  * * ")

    print("Before:")
    dir_stats(source, "Source: ")
    dir_stats(dest1, "Destination: ")

    if limit > 0:
        random.shuffle(files)

    cnt = 0
    for f in files:
        if extension != '' and not f.endswith(extension):
            continue

        os.rename(source + "/" + f, dest1 + "/" + f)
        cnt += 1
        if limit > 0 and cnt >= limit:
            break


    print("After: Moved {} files".format(cnt))

    dir_stats(source, "Source: ")
    dir_stats(dest1, "Destination: ")


def moveFilesFromCSV(filename, source, dest1, ignoreErrors=False):
    dfToMove = pd.read_csv(filename, header=None, names=['Filename'])

    print(" * *  * *  * *  * *  * *  * *  * * ")

    print("Before:")
    dir_stats(source, "Source: ")
    dir_stats(dest1, "Destination: ")

    cnt = 0
    for index, row in dfToMove.iterrows():
        f = row['Filename'].strip()
        # print(f)

        if ignoreErrors:
            try:
                os.rename(source + "/" + f, dest1 + "/" + f)
                cnt += 1
            except FileNotFoundError:
                print("Failed to find {}, ignoring that one".format(f))

        else:
            os.rename(source + "/" + f, dest1 + "/" + f)
            cnt += 1

    print("After: Moved {} files".format(cnt))

    dir_stats(source, "Source: ")
    dir_stats(dest1, "Destination: ")



def removeFiles(processdir, myset):
  cnt = 0
  removed = 0

  dir_stats(processdir)
  print(" - - ")

  for fn in os.listdir(processdir):
    filename = processdir + '/' + fn
    if os.path.isfile(filename) and fn != '.DS_Store':

      if fn in myset:
        os.remove(filename)
        removed += 1

    cnt += 1

    if cnt % 3000 == 0:
      print("Cnt {}, Removed {}".format(cnt, removed))

  dir_stats(processdir)

def removeFilesBasedOnCSVs(csvs, mydir):
    myset = createSetFromCVSs(csvs)
    removeFiles(mydir, myset)

def createSetFromCVSs(csvs):
    dfFiles=[]
    for filename in csvs:
        print("Loading {}".format(filename))
        dfFiles.append(pd.read_csv(filename, header=None, names=['Filename']))

    df = pd.concat(dfFiles)

    df.reset_index(drop=True, inplace=True)
    df.drop_duplicates(subset='Filename', keep="first", inplace=True)
    return set(df.Filename.astype(str).str.strip().unique())

def createFilesIfNotExist(filenames):
    """
    Will create a empty file if the file doesn't exist
    :param filenames: list of filenames
    """
    for filename in filenames:
        if not os.path.isfile(filename):
            print("File didn't exist, will create empty one {}".format(filename))
            open(filename, 'a').close()

def numFilesInDir(source):
  return len(os.listdir(source))

def createCSVwithFilesInDir(indir, filename):
    files = os.listdir(indir)
    dumpArrToFile(filename, files)

def fetchWithCache(url, cachedir="webcache"):
  if not os.path.isdir(cachedir):
    os.makedirs(cachedir)

  hash = hashlib.md5(url.encode('utf-8')).hexdigest()
  __, file_extension = os.path.splitext(url)

  filename = cachedir + "/" + hash + "." + file_extension

  if os.path.isfile(filename):
    return filename

  with urllib.request.urlopen(url) as response, open(filename, 'wb') as out_file:
    shutil.copyfileobj(response, out_file)
  return filename

def fetch(url, filename):
  with urllib.request.urlopen(url) as response, open(filename, 'wb') as out_file:
    shutil.copyfileobj(response, out_file)


# ######### DATA MODIFICATION ########

def modifyCsv(inputfile,
              outputfile,
              pkfield,
              removefiles=None,
              moveinfofile=None,
              filestomovefrom=None,
              killwords=None,
              multi_label=False,
              multi_label_correctiondir=None,
              multi_label_field=None
              ):
    """
    Loads the CSV file, and modify it and save as outputfile
    :param inputfile: CSV in (with header)
    :param outputfile: CSV out
    :param pkfield: the header that is the primary key
    :param removefiles: a list of files with primary keys on each line to remove (no header)
    :param moveinfofile: file with primary keys we want to move here (no header)
    :param filestomovefrom: list of filenames, each file has the same format as the inputfile,
    this is the meta data that we will copy over using the entries in moveinfofile
    if the same name as inputfile is in the list, it will be ignored
    :param killwords a list of strings. Any rows where any column partially match any of these strings,
    will be removed. Note: this removal takes place before the moveinfofile is processed.
    :param multi_label: boolean if its multi label or not
    :param multi_label_correctiondir: full path to directory with files of the name <label>.csv and <label>.remove.csv containing only PK
        these files are used to determine labels for each pk.
    :param multi_label_field: the name of the field in inputfile that contains the labels (assumed the rows will contain a comma separated string of labels)
    """

    print("Loading inputfile {}".format(inputfile))
    df = pd.read_csv(inputfile)

    print(">>>Real size of inputfile before modifying {}".format(df.shape[0]))

    #we need to remove items that are in this file, but also are in files to be moved to other files
    #so we need to change removefile to be an array, and pass them in there

    if removefiles is not None:
        for removefile in removefiles:
            print("Loading fields to remove from file {}".format(removefile))
            if removefile == moveinfofile:
                print("Ignoring this file since its the same as the moveinfofile")
                continue

            dfRemove = pd.read_csv(removefile, header=None, names=['PK'])

            for index, row in dfRemove.iterrows():
                #print("Removing {} (if it is present)".format(row['PK']))

                df = df[df[pkfield] != row['PK']]

    if killwords is not None:
        print("Will use killwords")
        print(killwords)
        print("Size before killwords {}".format(df.shape[0]))
        for killword in killwords:
            for col in df.loc[:, df.dtypes == object].columns:
                df = df[~df[col].str.contains(killword, na=False)]
        print("Size after killwords {}".format(df.shape[0]))


    if moveinfofile is not None:
        print("Loading fields to move, these are the PK that shall be moved from other files to this one {}".format(moveinfofile))

        dfMove = pd.read_csv(moveinfofile, header=None, names=['PK'])

        for movefile in filestomovefrom:
            print("Loading raw data for other files we want to move from {}".format(movefile))

            if movefile == inputfile:
                print("Ignoring this file since its the same as the inputfile")
                continue

            cntAdded=0
            try:
                dfRaw = pd.read_csv(movefile)
            except pd.errors.EmptyDataError:
                print("File was empty")
                continue

            for index, row in dfMove.iterrows():
                #print("Will try adding {}".format(row['PK']))
                tmpdf = dfRaw[ dfRaw[pkfield] == row['PK']]
                if not tmpdf.empty:
                    cntAdded += 1
                    #print("adding {}".format(row['PK']))
                    df = df.append(tmpdf, ignore_index=True)

            print("Added {} entries from {}".format(cntAdded, movefile))

    if multi_label:
        print("Processing multi label")
        #for each remove file, also load corresponding add file.
        #if the same PK is in both files, it is assumed the PK should be removed.

        processedfiles={}

        for fn in os.listdir(multi_label_correctiondir):
            filename = multi_label_correctiondir + '/' + fn
            if os.path.isfile(filename) and fn.endswith('.remove.csv'):
                processedfiles[fn] = True
                #extract label from filename
                label = fn.split(".remove.csv")[0]

                print("Loading remove file {}, label {}".format(fn, label))
                dfRemove = pd.read_csv(filename, header=None, names=['PK'])

                for __, row in dfRemove.iterrows():
                    #remove the label from the label field multi_label_field
                    print("Attempt to remove {}".format(row['PK']))
                    tmpdf = df[ df[pkfield] == row['PK']]
                    if tmpdf.empty:
                        print("Remove failed: Could not find an entry with PK {}".format(row['PK']))
                    else:
                        labels = tmpdf[multi_label_field].get_values()[0].split(',')
                        if label in labels:
                            labels.remove(label)
                            df.loc[ df[pkfield] == row['PK'], multi_label_field] = ",".join(labels)

                #LOAD ADD FILE AND DONT PROCESS ANY FROM REMOVE PK IN that one
                fn = label + ".csv"
                filename = multi_label_correctiondir + '/' + fn
                if os.path.isfile(filename):
                    processedfiles[fn] = True
                    print("Loading corresponding add file {}, label {}".format(fn, label))
                    dfAdd = pd.read_csv(filename, header=None, names=['PK'])
                    for __, row in dfAdd.iterrows():
                        tmpdf = dfRemove[ dfRemove['PK'] == row['PK']]

                        if tmpdf.empty:
                            #Not found in remove file, that means we can add it, otherwise we ignore it
                            print("Attempt to add {}".format(row['PK']))
                            tmpdf = df[ df[pkfield] == row['PK']]

                            if tmpdf.empty:
                                print("Add failed: Could not find an entry with PK {}".format(row['PK']))
                            else:
                                labels = tmpdf[multi_label_field].get_values()[0].split(',')
                                if label not in labels:
                                    labels.append(label)
                                    df.loc[ df[pkfield] == row['PK'], multi_label_field] = ",".join(labels)

        # Now process any add files that didnt have remove files
        for fn in os.listdir(multi_label_correctiondir):
            filename = multi_label_correctiondir + '/' + fn
            if os.path.isfile(filename) and fn.endswith('.csv'):
                label = fn.split(".csv")[0]

                if fn not in processedfiles:
                    print("Loading add file {}".format(fn))
                    dfAdd = pd.read_csv(filename, header=None, names=['PK'])
                    for __, row in dfAdd.iterrows():
                        print("Attempt to add {}".format(row['PK']))
                        tmpdf = df[ df[pkfield] == row['PK']]
                        if tmpdf.empty:
                            print("Add failed: Could not find an entry with PK {}".format(row['PK']))
                        else:
                            labels = tmpdf[multi_label_field].get_values()[0].split(',')
                            if label not in labels:
                                labels.append(label)
                                df.loc[ df[pkfield] == row['PK'], multi_label_field] = ",".join(labels)


    print("<<<Real size of inputfile after modifying {}".format(df.shape[0]))

    df.to_csv(outputfile, index=False)

def splitCSV(inputfile, outputdir, trainRatio=0.85):
    """
    Split a CSV file into two files, one for training, one for testing
    :param inputfile:
    :param outputdir: will be saved with the same name, with postfix .train/.test
    :param trainRatio:
    :return:
    """
    print("Loading inputfile {}".format(inputfile))
    df = pd.read_csv(inputfile)

    print("Real size of input: {}".format(df.shape[0]))

    trainSize = int(len(df.index) * trainRatio)

    #randomize the df
    df = df.sample(frac=1).reset_index(drop=True)

    df1 = df.iloc[:trainSize,]
    df2 = df.iloc[trainSize:]

    print("Real size of training: {}".format(df1.shape[0]))
    print("Real size of test: {}".format(df2.shape[0]))

    filename=outputdir + "/" + inputfile + ".train"
    print("Saving {}".format(filename))
    df1.to_csv(filename, index=False)

    filename=outputdir + "/" + inputfile + ".test"
    print("Saving {}".format(filename))
    df2.to_csv(filename, index=False)

# ######### DATA SPLITTING #########


def splitDataset(inputdir, setDir, extension="", trainRatio=0.75, validRatio=0.15):

    if (trainRatio + validRatio) > 1.0:
        print("Error: trainRatio + validRatio = {}".format(trainRatio + validRatio))
        return

    wantTestSet = (trainRatio + validRatio) < 1.0

    numberOfFiles=numFilesInDir(inputdir)
    trainNumber = int(numberOfFiles * trainRatio)

    if not wantTestSet:
        print("No test set will be created")
        validNumber=-1
    else:
        validNumber = int(numberOfFiles * validRatio)

    print("Train set {} - {}".format(trainRatio, trainNumber))
    print("Valid set {} - {}".format(validRatio, validNumber))
    if wantTestSet:
        print("Test set {} - {}".format(1.0 - trainRatio - validRatio, numberOfFiles - trainNumber - validNumber))

    shutil.rmtree(setDir, ignore_errors=True)
    os.mkdir(setDir)

    moveFiles(inputdir, setDir + "/train", trainNumber, extension=extension, wipeAndMakeDest=True)
    moveFiles(inputdir, setDir + "/valid", validNumber, extension=extension, wipeAndMakeDest=True)

    if wantTestSet:
        moveFiles(inputdir, setDir + "/test", extension=extension, wipeAndMakeDest=True)


# ############### FILE MODIFICATION #########

def dumpArrToFile(filename, arr, mode='w'):
    with open(filename, mode) as myfile:
        for r in arr:
            myfile.write("{}\n".format(r))

def appendArrToFile(filename, arrToAdd):

    sizeToAdd=len(arrToAdd)

    if 0 == sizeToAdd:
        print("Array is empty, not adding anything to {}".format(filename))
        return

    print("Will append {} entries to {}".format(sizeToAdd, filename))

    oldsize = os.path.getsize(filename)
    print("Old size {}".format(oldsize))

    #make bkp copy
    postfix= time.strftime("%Y%m%d_%H%M")
    bkpfilename=filename + '.' + postfix
    shutil.copyfile(filename, bkpfilename)

    dumpArrToFile(filename, arrToAdd, "a")

    newsize = os.path.getsize(filename)
    print("New size {}".format(newsize))
    if newsize == oldsize:
        print("PROBLEM? The filesize didnt change")

    if newsize < oldsize:
        print("****** ERROR: The filesize decreased!")

# *************** STATS *****************




#https://stackoverflow.com/questions/44054534/confusion-matrix-error-when-array-dimensions-are-of-size-3/44193420#44193420
def pretty_print_conf_matrix(y_true, y_pred,
                             classes,
                             normalize=False,
                             title='Confusion matrix',
                             cmap=plt.cm.Blues):
    """
    Mostly stolen from: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py

    Normalization changed, classification_report stats added below plot
    """

    cm = confusion_matrix(y_true, y_pred)

    # Configure Confusion Matrix Plot Aesthetics (no text yet)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=14)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.ylabel('True label', fontsize=12)
    plt.xlabel('Predicted label', fontsize=12)

    # Calculate normalized values (so all cells sum to 1) if desired
    if normalize:
        cm = np.round(cm.astype('float') / cm.sum(), 2)  # (axis=1)[:, np.newaxis]

    # Place Numbers as Text on Confusion Matrix Plot
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=12)

    # Add Precision, Recall, F-1 Score as Captions Below Plot
    rpt = classification_report(y_true, y_pred)
    rpt = rpt.replace('avg / total', '      avg')
    rpt = rpt.replace('support', 'N Obs')

    plt.annotate(rpt,
                 xy=(0, 0),
                 xytext=(-50, -140),
                 xycoords='axes fraction', textcoords='offset points',
                 fontsize=12, ha='left')

    # Plot
    plt.tight_layout()

def stats_dump(y_true, y_pred, labels):
  plt.style.use('classic')
  plt.figure(figsize=(3, 3))
  pretty_print_conf_matrix(y_true, y_pred,
                             classes=labels,
#                             classes=['F', 'M'],
                             normalize=True,
                             title='Confusion Matrix')


  cm = confusion_matrix(y_true, y_pred, labels=labels)
  print(labels)
  print(cm)

  recall = recall_score(y_true, y_pred)
  precision = precision_score(y_true, y_pred)
  f1 = f1_score(y_true, y_pred)
  acc = accuracy_score(y_true, y_pred)
  print("Recall {}".format(recall))
  print("Precision {}".format(precision))
  print("F1 {}".format(f1))
  print("Accuracy {}".format(acc))


def stats_small(tp,tn,fp,fn):
    acc = (tp+tn) / (tp+tn+fp+fn)
    pr = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = (2 * pr * rec) / (pr + rec)

    print("tp={}".format(tp))
    print("tn={}".format(tn))
    print("fp={}".format(fp))
    print("fn={}".format(fn))

    print("Accuracy {}".format(acc))
    print("F1 {}".format(f1))
    print("Precision {}".format(pr))
    print("Recall {}".format(rec))

# *************** PLOTTING ***************

def plt_show_img(filename):
    img = plt.imread(filename)
    plt.imshow(img)
    plt.axis('off')

def plt_show_imgs_from_set(basedir, myset):
    errs = []
    cnt = 1
    tot=len(myset)
    for f in myset:
        filename = basedir + "/train/" + f

        if not Path(filename).is_file():
            filename = basedir + "/valid/" + f
            if not Path(filename).is_file():
                print("ERR: file doesnt exist {}".format(f))
                errs.append(f)
                continue

        plt.figure()
        plt_show_img(filename)
        plt.show()
        print("{}/{} - {}".format(cnt, tot, f))
        cnt +=1

    return errs

def plt_show_imgs(path_root, img_paths, dictkey=False, cols=4, rows=3):
    img_num = cols * rows

    if len(img_paths) < img_num:
        img_num = len(img_paths)
#        if len(img_paths) < cols:
#            cols = img_paths
#            rows = 1

    img_ids = np.random.choice(len(img_paths), img_num, replace=False)

    for i, img_id in enumerate(img_ids):

        a = plt.subplot(rows, cols, i + 1)
        if dictkey:
            filename = img_paths[img_id][dictkey]
            #a.set_title("{}".formatimg_paths[img_id]())
            a.set_title("{}".format(filename + "|" + str(img_paths[img_id]['age'])))
        else:
            filename = img_paths[img_id]

        img = cv2.imread(path_root + str(filename))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')

    plt.show()

def plt_show_imgs_from_dir(path, cols=4, rows=3):
    files = os.listdir(path)
    random.shuffle(files)

    #just limit to cols*rows, so its easier to figure out filenames
    files = [files[i] for i in sorted(random.sample(range(len(files)), cols*rows))]
    print(files)

#    print(files[0])
    #colaboratory_helpers.plt_show_imgs(path + "/", files, dictkey=False, cols=cols, rows=rows)
    plt_show_imgs(path + "/", files, dictkey=False, cols=cols, rows=rows)

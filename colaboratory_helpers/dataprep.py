import pandas as pd

def prepareMetaData(label2filenames, metadatafiles, outdir, split=None):
    """
    Takes lists of ids, fetches the metadata for corresponding ids from metadatafiles,
    and split the data according to the ratio we want in train/valid/test sets and saves

    :param label2filenames: string -> list of files with pks. E.g. "pos" -> ["positemid.csv"]
    :param metadatainfo: all files with meta data, we assume the first column is the pk that will match the ones from our label2filenaes
    :param outdir: where to save the result
    :param split: dict if we want to split ratio -> postfix for filename, E.g. { 0.7: "train", 0.2: 'valid', 0.1: 'test'}
    :return:
    """

    #Validate that ratios adds up
    if split is not None:
        summ=sum(split.keys())
        if 1.0 != summ:
            print("ERROR: Ratio doesnt add up to 1.0 != {}".format(summ))
            print(split)
            return

    #load metadatainfo into a dataframe, assuming header and the same format
    dfMetadata = concatIntoDf(metadatafiles, header=True)

    pkfield=dfMetadata.columns[0]

    # for each label, load the files with itemids, take the union, extract that info from metadata files
    for label, filenames in label2filenames.items():
        dfLabel = concatIntoDf(filenames, headerNames=[pkfield])

        dfMerged = pd.merge(dfLabel, dfMetadata, on=pkfield, how="left")

        if dfMerged.isnull().values.any():
            print("ERROR: entries without metadata")
            failed = dfMerged[ dfMerged.isnull().values ][pkfield].copy()
            failed.drop_duplicates(inplace=True)
            print(failed)
            print("XXXXXXXXXXXXXXXXXXXXX")

            #Since we carry on regardless if we dont find all entries, we drop the NaN entries
            dfMerged.dropna(inplace=True)

        if split is not None:

            #randomize it
            dfMerged = dfMerged.sample(frac=1).reset_index(drop=True)

            print("€€€€€€€€€€€€€€")
            print(dfMerged)
            print(len(dfMerged.index))
            print("€€€€€€€€€€€€€€")

            numrows = dfMerged.shape[0]

            numratios = len(split.keys())
            idx=0
            i=0

            for ratio, postfix in split.items():
                print("ratio {} postfix {}".format(ratio, postfix))


                if(i == numratios -1):
                    break


                chunk = int(numrows * ratio)
                print("idx {} chunk {}".format(idx, chunk))
                df1 = dfMerged.iloc[idx:idx + chunk]


                print("=" * 50)
                print(df1)

                outfile = outdir + "/" + postfix + ".csv"
                print("Saving {}".format(outfile))
                df1.to_csv(outfile, index=False)

                idx=idx + chunk
                i += 1


            df1 = dfMerged.iloc[idx:]
            outfile = outdir + "/" + postfix + ".csv"
            print("Saving {}".format(outfile))
            df1.to_csv(outfile, index=False)
        else:
            outfile = outdir + "/" + label + ".csv"
            print("Saving {}".format(outfile))
            dfMerged.to_csv(outfile, index=False)


    return dfMerged


def concatIntoDf(filelist, header=False, headerNames=None):
    """
    Load all files into a single dataframe, remove duplicates (assuming first column is the pk)
    :param filelist: list of csv filenames
    :return: dataframe
    """

    dfFiles = []
    for filename in filelist:
        print("Loading {}".format(filename))
        if header:
            dfFiles.append(pd.read_csv(filename))
        else:
            dfFiles.append(pd.read_csv(filename, header=None, names=headerNames))

    df = pd.concat(dfFiles, ignore_index=True)

    df.reset_index(drop=True, inplace=True)
    df.drop_duplicates(subset=df.columns[0], keep="last", inplace=True)
    return df


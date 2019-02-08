import pickle

def setTrainableFrom(model, layername):
    """
    Set the layers from <layername> and after to trainable
    :param model: keras model
    :param layername: string
    """

    model.trainable=True
    setTrainable = False

    for layer in model.layers:
        if layer.name == layername:
            setTrainable=True
        layer.trainable = setTrainable

def showLayerInfo(model):
    """
    Display which layers are trainable
    """

    for layer in model.layers:
        print("{} Trainable: {}".format(layer.name, layer.trainable))

def saveHistAndWeights(history, model, prefix):
    """
    Save the history and model weights into GDrive
    :param history:
    :param model:
    :param prefix:
    :return:
    """
    filename = prefix + "_history.pkl"
    with open(filename, 'wb') as fh:
        pickle.dump(history.history, fh)
    colaboratory_helpers.drive_save(filename)

    filename = prefix + "_model.h5"
    model.save_weights(filename)
    colaboratory_helpers.drive_save(filename)

def loadHistAndWeights(gdriveIDhistory, gdriveIDmodel, model, prefix):
    """
    Load history and model weights from GDrive
    :param gdriveIDhistory:
    :param gdriveIDmodel:
    :param model:
    :param prefix:
    :return: (history, model)
    """
    filename = prefix + "_history.pkl"
    print("Fetching and loading history {}".format(filename))
    colaboratory_helpers.drive_load(filename, gdriveIDhistory)

    history = pickle.load(open(filename, "rb"))

    tmphist = lambda: None
    setattr(tmphist, 'history', history)

    filename = prefix + "_model.h5"
    print("Fetchning and loading weights {}".format(filename))
    colaboratory_helpers.drive_load(filename, gdriveIDmodel)

    model.load_weights(filename)

    return (tmphist, model)

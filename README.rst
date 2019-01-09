====================
Colaboratory Helpers
====================

Helper functions to use within Colaboratory

If you want to use it, it's probably best to fork it since I just hack what I currently need and
I don't make sure it's backwards compatible (unless I need it), so functions might change without notice.

On the plus side:

* Free software: MIT license

Features
--------

* Helper functions I use, e.g. save/load from GDrive, move files around when preparing data, plotting etc.
* View the source code in https://github.com/EdlinOrg/colaboratory_helpers/tree/master/colaboratory_helpers
  to see what is there

Examples
--------

Note: if you work on the same drive in one notebook its easier to just mount it

::

    from google.colab import drive
    drive.mount('/content/gdrive')

**Save/load files to/from GDrive using PyDrive:**

To get gdrive-file-id:

::

    Click "Share" -> "Advanced" in GDrive, the "Link to share" looks like this
    https://drive.google.com/file/d/GDRIVE-ID/view?usp=sharing
    and there you have the id

::

    colaboratory_helpers.drive_load("filename.ext", "gdrive-file-id", force=True)
    colaboratory_helpers.drive_save("filename.ext")

**Predict using fastai model (resnet34)**

Save the model you trained in fastai

::

    torch.save(learn.model, modelfilept)

Load and use that model for prediction elsewhere

::

    from colaboratory_helpers import fastai_pyt_gpu
    clss=fastai_pyt_gpu.FaiGPU(modelfilept)
    clss.predictscore("somefile.jpg")

Install stuff
-------------
Install dlib:

::

  !pip install -q cmake
  !pip install -q dlib

Install fastai 0.7.0:

::

  !pip install -q torchtext==0.2.3
  !pip install -q fastai==0.7.0

Install fastText

::

  !pip install -q cmake
  !pip install -q wrap-pybind11
  !git clone https://github.com/facebookresearch/fastText.git
  !cd fastText; pip install -q .

Install fasttext

::

  !pip install -q cython
  !pip install -q fasttext

  
Imagemagick

::

   !apt-get install imagemagick

   
Random Notes
------------

Add at the top of notebooks

::

    %load_ext autoreload
    %autoreload 2

::

    colabDev=False

    if not colabDev:
      #!pip uninstall colaboratory_helpers -y
      !pip --no-cache-dir  -q install --upgrade --force-reinstall --ignore-installed git+https://github.com/EdlinOrg/colaboratory_helpers/
      from colaboratory_helpers import colaboratory_helpers
      #from colaboratory_helpers import colaboratory_helpers, fastai_analyze
      #import colaboratory_helpers.ffsfastText as ffsfastText
      #import colaboratory_helpers.fastTextPrep as fastTextPrep
    else:
      !pip install -q PyDrive
      !rm -f colaboratory_helpers.py
      !wget "https://raw.githubusercontent.com/EdlinOrg/colaboratory_helpers/master/colaboratory_helpers/colaboratory_helpers.py"
      import colaboratory_helpers

      #!rm -f fastai_analyze.py
      #!wget "https://raw.githubusercontent.com/EdlinOrg/colaboratory_helpers/master/colaboratory_helpers/fastai_analyze.py"
      #import fastai_analyze
    %matplotlib inline
    %pylab inline
    pylab.rcParams['figure.figsize'] = (16, 6)
    !date

    gdrivepathmodules="/content/gdrive/My Drive/colab/"

Add this at end of the notebook to get a notification that it is finished

::

  !date
  !test ! -e done.wav && wget https://github.com/EdlinOrg/colaboratory_helpers/blob/master/assets/benhill.wav?raw=true -O done.wav
  import IPython
  IPython.display.Audio("done.wav",autoplay=True)


Imagemagick
-----------

Resize to 50%
!convert  -resize 50% a.jpg b.jpg

Convert to grayscale
!convert <img_in> -set colorspace Gray -separate -average <img_out>

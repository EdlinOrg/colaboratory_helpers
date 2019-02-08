# https://gist.github.com/korakot/c1dd8575ea9de6c218011d7bf996ba0f
from ctypes import cast, py_object
from google.colab import output

"""
This one is using the label of the checkbox instead of idx.

Note: we use the label exactly as it is, no modifying with removing spaces/special characters etc
so make sure the label is ONE WORD ONLY and no funny chars
"""

moveClass="UIBOXMOVE"
removeClass="UIBOXREMOVE"
uiMoveLables={}

def css(className, selectedColor):
    return """
<style>    
.%s {
    -webkit-appearance:none;
    width:30px;
    height:30px;
    background:white;
    border-radius:5px;
    border:2px solid #555;
}
.%s:checked {
    background: %s;
}    
</style>
    """ % (className, className, selectedColor)

def checkboxCode(pk, className, label):
    return """<br>
    <label><input type='checkbox' class="%s" value="%s"> %s</label>
    """ % (className, pk, label)

def triggerItCode(cbName, label):

    jsstr="""
    <script>
    function sumUpEntries(myClassName) {
        var apa="";       

        var cusid_ele = document.getElementsByClassName(myClassName);
        for (var i = 0; i < cusid_ele.length; ++i) {
            var item = cusid_ele[i];  
            if(item.checked){
                if(apa != "") {
                  apa += ";"
                }
                apa += item.value;
            }
        }

        return apa;
    }
    
    function triggerAction() {
      var allResult={};
    """

    idx=0
    for moveLabel, __ in uiMoveLables.items():
        jsstr += """
allResult.%s = sumUpEntries('%s');
""" % (moveClass + moveLabel, moveClass + moveLabel)
        idx +=1

    jsstr += """
allResult.%s = sumUpEntries('%s');
""" % (removeClass, removeClass)

    jsstr += """
      google.colab.kernel.invokeFunction(
             '%s', [ allResult ],
             {});
      
}    
</script>
        <label><input type='checkbox'
         onchange="triggerAction()"
        > %s</label>

    """ % (cbName, label)

    return jsstr

def checkboxCodeRemove(filename):
    return checkboxCode(filename, removeClass, " &#9003; REMOVE")

def triggerHTMLCode():
    return triggerItCode('callback_letsdoit', "TRIGGER IT!")

def allCheckboxes(pk):
    myhtml=""
    idx=0

    for lbl, __ in uiMoveLables.items():
        myhtml += checkboxCode(pk, moveClass + lbl, " &#9654; " + lbl)
        idx +=1

    myhtml += checkboxCodeRemove(pk)

    return myhtml

def init(moveLabels={}):
    """
    :param moveLabels: make sure it is a collections.OrderedDict()
    :return:
    """
    print("Init checkboxmulti")

    output.register_callback('callback_letsdoit', cbTriggered)

    global uiMoveLables
    uiMoveLables=moveLabels

    cssCode = css(removeClass, '#F00')

    for lbl, __ in uiMoveLables.items():
        cssCode += css(moveClass + lbl, "#090")

    return cssCode

def cbTriggered(value):
    """
    This one is called from the invokeFunction with the result as parameter
    #Note: a pk can appear in several lists if the user ticked two boxes
    :param value: an object with our data
    """
    global selectedCheckboxes
    selectedCheckboxes=value


def getMoverAsArray(labelName):
    """
    Return all the id:s of all checkboxes for this labelName that has been ticked
    :param labelName:
    :return: list of ids
    """
    strarr = selectedCheckboxes[moveClass + labelName]
    arr = strarr.split(';')
    if len(arr) == 1 and arr[0] == '':
        return []
    return arr

def getRemoveAsArray():
    """
    Return all the id:s of all "remove" checkboxes that has been ticked
    :return: list of ids
    """
    strarr = selectedCheckboxes[removeClass]
    arr = strarr.split(';')
    if len(arr) == 1 and arr[0] == '':
        return []
    return arr

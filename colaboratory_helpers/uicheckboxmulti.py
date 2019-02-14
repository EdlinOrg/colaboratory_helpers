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


def anyTextToColor(mystr):
    """
    Converts s string to a hex color
    :param mystr: any string
    :return: for example "a0c73a"
    """

    if len(mystr) < 3:
        # pad up with zeros
        while len(mystr) % 3 != 0:
            mystr += "0"

    i = 0
    sum1 = 0
    sum2 = 0
    sum3 = 0
    for c in mystr:
        if i % 3 == 0:
            sum1 += int( str(ord(c)) + str(i)[::-1])
        if i % 3 == 1:
            sum2 += int(str(ord(c)) + str(i)[::-1])
        if i % 3 == 2:
            sum3 += int(str(ord(c)) + str(i)[::-1])
        i += 1

    x1 = sum1 % 255
    x2 = sum2 % 255
    x3 = sum3 % 255

    #actually, lets go for shades of green
    x2 = 255

    outstr = "%x%x%x" % (x1, x2, x3)

    while len(outstr) < 6:
        outstr += "a"

    return outstr

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
        #cssCode += css(moveClass + lbl, "#090")
        mecolor = anyTextToColor(lbl)
        cssCode += css(moveClass + lbl, "#" + mecolor)

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

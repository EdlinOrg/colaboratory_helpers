# https://gist.github.com/korakot/c1dd8575ea9de6c218011d7bf996ba0f
from ctypes import cast, py_object
from google.colab import output

"""
This one is using the label of the checkbox instead of idx.

Note: we use the label exactly as it is, no modifying with removing spaces/special characters etc
so make sure the label is ONE WORD ONLY and no funny chars
"""

class Uicheckboxmulti():

    def __init__(self, moveLabels, line_breaks=False):
        """
        :param moveLabels: make sure it is a collections.OrderedDict()
        """
        self.moveClass="UIBOXMOVE"
        self.removeClass="UIBOXREMOVE"
        self.uiMoveLables=moveLabels
        self.line_breaks = line_breaks

        output.register_callback('callback_letsdoit', self.cbTriggered)

    def css(self, className, selectedColor):
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

    def checkboxCode(self, pk, className, label, selected=False):
        if self.line_breaks:
            mystr = '<br>'
        else:
            mystr = ''

        selhtml=''
        if selected:
            selhtml='checked'

        mystr += """
<label><input type='checkbox' %s class="%s" value="%s"> %s</label>
""" % (selhtml, className, pk, label)

        return mystr

    def triggerItCode(self, cbName, label):

        jsstr = """
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

        for moveLabel in self.uiMoveLables:
            jsstr += """
    allResult.%s = sumUpEntries('%s');
""" % (self.moveClass + moveLabel, self.moveClass + moveLabel)

            # the remove label ones
            jsstr += """
    allResult.%s = sumUpEntries('%s');
""" % (self.removeClass + moveLabel, self.removeClass + moveLabel)

        # The item that shall be removed completely
        jsstr += """
    allResult.%s = sumUpEntries('%s');
""" % (self.removeClass, self.removeClass)

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

    def checkboxCodeRemove(self, filename):
        return self.checkboxCode(filename, self.removeClass, " &#9003; REMOVE")

    def triggerHTMLCode(self):
        return self.triggerItCode('callback_letsdoit', "TRIGGER IT!")

    def allCheckboxes(self, pk, preselected=None):
        """
        :param preselected: a list of labels that should be marked as selected
        """
        myhtml=""

        for lbl in self.uiMoveLables:
            selected=False
            if preselected is not None:
                selected = lbl in preselected
            myhtml += self.checkboxCode(pk, self.moveClass + lbl, " &#9654; " + lbl, selected=selected)

            #remove buttons for each label
            myhtml += self.checkboxCode(pk, self.removeClass + lbl, " &#9003; Remove " + lbl)

            if not self.line_breaks:
                #if we dont have line breaks, we add one so the checkboxes will appear in pairs: add / remove
                myhtml += "<br>"

        if not self.line_breaks:
            #if we dont have line breaks, we add one so the remove checkbox comes on a new line
            myhtml += "<br>"

        myhtml += self.checkboxCodeRemove(pk)

        return myhtml

    def anyTextToColor(self, mystr, r=None):
        """
        Converts s string to a hex color
        :param mystr: any string
        :param r: hard code the red value to this
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

        if r is not None:
            x1 = r

        # if we wants to force a shade of green
        # x2 = 255

        outstr = "%x%x%x" % (x1, x2, x3)

        while len(outstr) < 6:
            outstr += "a"

        return outstr

    def initialCssCode(self):
        """
        :return: string with the base css needed
        """

        cssCode = self.css(self.removeClass, '#F00')

        for lbl in self.uiMoveLables:
            mecolor = self.anyTextToColor(lbl)
            cssCode += self.css(self.moveClass + lbl, "#" + mecolor)

            #remove label css
            #mecolor = self.anyTextToColor(lbl, r=255)
            mecolor = self.anyTextToColor(lbl)
            cssCode += self.css(self.removeClass + lbl, "#" + mecolor)

        return cssCode

    def cbTriggered(self, value):
        """
        This one is called from the invokeFunction with the result as parameter
        #Note: a pk can appear in several lists if the user ticked two boxes
        :param value: an object with our data
        """
        global selectedCheckboxes
        selectedCheckboxes=value

    def getMoverAsArray(self, labelName):
        """
        Return all the id:s of all checkboxes for this labelName that has been ticked
        :param labelName:
        :return: list of ids
        """
        return self.getItemsAsArray(self.moveClass + labelName)

    def getRemoveLabelAsArray(self, labelName):
        """
        Return all the id:s of all checkboxes for this labelName that has been ticked to be removed
        :param labelName:
        :return: list of ids
        """
        return self.getItemsAsArray(self.removeClass + labelName)

    def getRemoveAsArray(self):
        """
        Return all the id:s of all "remove" checkboxes that has been ticked (to remove the item completely)
        :return: list of ids
        """
        return self.getItemsAsArray(self.removeClass)

    def getItemsAsArray(self, name):
        """
        Return all the id:s of all the checkboxes that has been ticked with that name in the css class
        :param name: the class to look for
        :return: list of ids
        """
        strarr = selectedCheckboxes[name]
        arr = strarr.split(';')
        if len(arr) == 1 and arr[0] == '':
            return []
        return arr

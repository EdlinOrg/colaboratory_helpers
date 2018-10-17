# https://gist.github.com/korakot/c1dd8575ea9de6c218011d7bf996ba0f
from ctypes import cast, py_object
from google.colab import output

moveClass="UIBOXMOVE"
removeClass="UIBOXREMOVE"
uiMove = []
uiRemove = []


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



def cssMove():
    return css(moveClass, "#090")

def cssRemove():
    return css(removeClass, "#F00")

def checkboxCode(filename, className, label):
    return """<br>
    <label><input type='checkbox' class="%s" value="%s"> %s</label>
    """ % (className, filename, label)


def triggerReportCode(cbName, className, label):
    return """
    <script>
    function sumUpEntries%s() {
        var apa="";       

        var cusid_ele = document.getElementsByClassName('%s');
        for (var i = 0; i < cusid_ele.length; ++i) {
            var item = cusid_ele[i];  
            if(item.checked){
                if(apa != "") {
                  apa += ";"
                }
                apa += item.value;
            }
        }

        google.colab.kernel.invokeFunction(
             '%s', [ apa ],
             {});
    }
    </script>
        <label><input type='checkbox'
         onchange="sumUpEntries%s()"
        > %s</label>
        """ % (className, className, cbName, className, label)


def checkboxCodeMove(filename):
    return checkboxCode(filename, moveClass, " &#9654; MOVE")

def checkboxCodeRemove(filename):
    return checkboxCode(filename, removeClass, " &#9003; REMOVE")

def triggerReportCodeMove():
    return triggerReportCode('calc_report_move', moveClass, "TRIGGER MOVE")

def triggerReportCodeRemove():
    return triggerReportCode('calc_report_remove', removeClass, "TRIGGER REMOVE")

def init():
    print("Init simpleui")
    calcReportMoveCb("")
    calcReportRemoveCb("")

    activateCallback()



def calcReport(value):
    if('' == value):
        return []
    return value.split(";")


def calcReportMoveCb(value):
    global uiMove
    uiMove = calcReport(value)


def calcReportRemoveCb(value):
    global uiRemove
    uiRemove = calcReport(value)


def activateCallback():
    output.register_callback('calc_report_move', calcReportMoveCb)
    output.register_callback('calc_report_remove', calcReportRemoveCb)

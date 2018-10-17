# https://gist.github.com/korakot/c1dd8575ea9de6c218011d7bf996ba0f
from ctypes import cast, py_object
from google.colab import output

uiVid2itemid = {}
uiCheckedItemids = {}

def init():
    print("Init ui")
    global uiVid2itemid
    global uiCheckedItemids
    uiVid2itemid = {}
    uiCheckedItemids = {}

    activateCallback()

class Checkbox:
    def __init__(self, itemid, value=False):
        self.itemid = itemid
        self.value = value

    def _repr_html_(self):
        checked = 'checked' if self.value else ""
        return """<br>
    <input type='checkbox' %s 
     onchange="
       google.colab.kernel.invokeFunction(
         'set_var', [ %d, this.checked ],
         {})
     "
    >
    """ % (checked, id(self))


def setVarCb(v_id, value):
    obj = cast(v_id, py_object).value
    obj.value = value
    uiCheckedItemids[uiVid2itemid[v_id]] = value

def getCheckbox(itemid):
    cb = Checkbox(itemid)
    uiVid2itemid[id(cb)] = itemid
    return cb

def activateCallback():
    output.register_callback('set_var', setVarCb)

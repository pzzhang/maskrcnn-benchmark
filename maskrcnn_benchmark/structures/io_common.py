import os
import numpy as np
import base64
import cv2
import yaml


def img_from_base64(imagestring):
    jpgbytestring = base64.b64decode(imagestring)
    nparr = np.frombuffer(jpgbytestring, np.uint8)
    try:
        r = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return r
    except:
        return None


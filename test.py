from test_detector import test_detector
from PIL import Image
import sys

try:

    im = Image.open('./testfolder/QRDC1025_00022.jpg')

    res = test_detector.test(im, './testfolder/','','')
    print(res)

except:
    print("Unexpected error:", sys.exc_info()[0])




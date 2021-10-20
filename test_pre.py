from funcOD import *
from PIL import Image
import sys

try:

    resultData = mainPreLoadModel('./testfolder/')
    #print(resultData['result'])
    #mainPredict('','./testfolder/', '' ,xx['result'], '', '')
    im = Image.open('./testfolder/QRDC1025_00022.jpg')
   
    mainPredict(im, './testfolder/' , '', resultData['result'], '', '')
    mainPredict(im, './testfolder/' , '', resultData['result'], '', '')
    

    #res = test_detector.test(im, './testfolder/','','')
    #print(res)

except:
    print("Unexpected error:", sys.exc_info()[0])


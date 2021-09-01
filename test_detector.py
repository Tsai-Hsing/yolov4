# -*- coding: utf-8 -*

import os
import string
from PIL import Image
import json
import sys
import tempfile

class test_detector:
    @classmethod
    def readcmd(cls, cmd):
        try:
            ftmp = tempfile.NamedTemporaryFile(suffix='.out', prefix='tmp', delete=False)
            fpath = ftmp.name
            if os.name=="nt":
                fpath = fpath.replace("/","\\") # forwin
            ftmp.close()
            os.system(cmd + " > " + fpath)
            data = ""
            with open(fpath, 'r') as file:
                data = file.read()
                file.close()
            os.remove(fpath)
            return data
        except:
            print(sys.exc_info()[0])

    def __init__(self):
        pass


    @classmethod
    def test(cls, img, path, graph, session):
        try:
            #path = os.path.abspath(os.getcwd())
            #print(path)
            img.save(os.path.abspath(os.getcwd()) + '/original.' + img.format.lower())
            modelname = ''
            for filename in os.listdir(path):
                if filename.endswith('.weights'):
                    modelname = os.path.splitext(filename)[0]
            fin = open(path + 'obj.data', 'rt')
            data = fin.read()
            data = data.replace('/mnt/2c67bd82-3031-40f8-8f53-58564ba23509/Graphics/yolo/test/', path).replace('cfg/' + modelname,'')
            fin.close()
            fin = open(path + 'obj.data', 'wt')
            fin.write(data)
            fin.close()
            resultstr = cls.readcmd('./darknet detector test ' + path + '"obj.data" ' + path + '"yolov4.cfg" ' + path + modelname + '".weights" "original.' + img.format.lower() + '" -gpus 0 -ext_output  2>&1 | tee -a ./temp.log')
            #print('\n\n\n\n\n\n\n')
            #print(resultstr.split('\n'))
            resultarr = resultstr.split('\n')
            returnarr = []
            for str in resultarr:
                if str.startswith('[[') and str.endswith(']]'):
                    #print(str + '\n')
                    tmparr = str.replace('[[','').replace(']]','').split(' ')
                    tmpval = {
                        'class':tmparr[4],
                        'classResult':[{'score': tmparr[5],'boundingBox':[tmparr[0], tmparr[1], tmparr[2], tmparr[3]]}]
                    }
                    for obj in returnarr:
                        if obj['class'] == tmparr[4]:
                            obj['classResult'].append([{'score':tmparr[5],'boundingBox':[tmparr[0], tmparr[1], tmparr[2], tmparr[3]]}])
                        else:
                            returnarr.append(tmpval)
                    if len(returnarr) == 0:
                        returnarr.append(tmpval)
            returnval = {
                    'isSuccess':'true',
                    'ErrorMsg':'',
                    'result':returnarr
            }
            return json.dumps(returnval)
        except:
            returnval = {
                    'isSuccess': 'false',
                    'ErrorMsg': sys.exc_info()[0],
            }
            return json.dumps(returnval)


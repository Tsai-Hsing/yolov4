import argparse
import os
import glob
import random
#import darknet
import time
import cv2
import numpy as np
#import darknet
import json

from ctypes import *
import math
#import random
#import os


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int),
                ("uc", POINTER(c_float)),
                ("points", c_int),
                ("embeddings", POINTER(c_float)),
                ("embedding_size", c_int),
                ("sim", c_float),
                ("track_id", c_int)]

class DETNUMPAIR(Structure):
    _fields_ = [("num", c_int),
                ("dets", POINTER(DETECTION))]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


def network_width(lib, net):
    return lib.network_width(net)


def network_height(lib, net):
    return lib.network_height(net)


def bbox2points(bbox):
    """
    From bounding box yolo format
    to corner points cv2 rectangle
    """
    x, y, w, h = bbox
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def class_colors(names):
    """
    Create a dict with one random BGR color for each
    class name
    """
    return {name: (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255)) for name in names}


def load_network(lib, config_file, data_file, weights, batch_size=1):
    """
    load model description and weights from config files
    args:
        config_file (str): path to .cfg model file
        data_file (str): path to .data model file
        weights (str): path to weights
    returns:
        network: trained model
        class_names
        class_colors
    """
    load_net_custom = lib.load_network_custom
    load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
    load_net_custom.restype = c_void_p

    load_meta = lib.get_metadata
    lib.get_metadata.argtypes = [c_char_p]
    lib.get_metadata.restype = METADATA

    network = load_net_custom(
        config_file.encode("ascii"),
        weights.encode("ascii"), 0, batch_size)
    metadata = load_meta(data_file.encode("ascii"))
    class_names = [metadata.names[i].decode("ascii") for i in range(metadata.classes)]
    colors = class_colors(class_names)
    return network, class_names, colors


def print_detections(detections, coordinates=False):
    print("\nObjects:")
    for label, confidence, bbox in detections:
        x, y, w, h = bbox
        if coordinates:
            print("{}: {}%    (left_x: {:.0f}   top_y:  {:.0f}   width:   {:.0f}   height:  {:.0f})".format(label, confidence, x, y, w, h))
        else:
            print("{}: {}%".format(label, confidence))


def draw_boxes(detections, image, colors):
    import cv2
    for label, confidence, bbox in detections:
        left, top, right, bottom = bbox2points(bbox)
        cv2.rectangle(image, (left, top), (right, bottom), colors[label], 1)
        cv2.putText(image, "{} [{:.2f}]".format(label, float(confidence)),
                    (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    colors[label], 2)
    return image


def decode_detection(detections):
    decoded = []
    for label, confidence, bbox in detections:
        confidence = str(round(confidence * 100, 2))
        decoded.append((str(label), confidence, bbox))
    return decoded


def remove_negatives(detections, class_names, num):
    """
    Remove all classes with 0% confidence within the detection
    """
    predictions = []
    for j in range(num):
        for idx, name in enumerate(class_names):
            if detections[j].prob[idx] > 0:
                bbox = detections[j].bbox
                bbox = (bbox.x, bbox.y, bbox.w, bbox.h)
                predictions.append((name, detections[j].prob[idx], (bbox)))
    return predictions


def detect_image(lib, network, class_names, image, thresh=.5, hier_thresh=.5, nms=.45):
    predict_image = lib.network_predict_image
    predict_image.argtypes = [c_void_p, IMAGE]
    predict_image.restype = POINTER(c_float)

    get_network_boxes = lib.get_network_boxes
    get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int), c_int]
    get_network_boxes.restype = POINTER(DETECTION)

    do_nms_sort = lib.do_nms_sort
    do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

    free_detections = lib.free_detections
    free_detections.argtypes = [POINTER(DETECTION), c_int]
    
    """
        Returns a list with highest confidence class and their bbox
    """
    pnum = pointer(c_int(0))
    predict_image(network, image)
    detections = get_network_boxes(network, image.w, image.h,
                                   thresh, hier_thresh, None, 0, pnum, 0)
    num = pnum[0]
    if nms:
        do_nms_sort(detections, num, len(class_names), nms)
    predictions = remove_negatives(detections, class_names, num)
    predictions = decode_detection(predictions)
    free_detections(detections, num)
    return sorted(predictions, key=lambda x: x[1])


##  lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
##  lib = CDLL("libdarknet.so", RTLD_GLOBAL)
#hasGPU = True
#if os.name == "nt":
#    cwd = os.path.dirname(__file__)
#    os.environ['PATH'] = cwd + ';' + os.environ['PATH']
#    winGPUdll = os.path.join(cwd, "yolo_cpp_dll.dll")
#    winNoGPUdll = os.path.join(cwd, "yolo_cpp_dll_nogpu.dll")
#    envKeys = list()
#    for k, v in os.environ.items():
#        envKeys.append(k)
#    try:
#        try:
#            tmp = os.environ["FORCE_CPU"].lower()
#            if tmp in ["1", "true", "yes", "on"]:
#                raise ValueError("ForceCPU")
#            else:
#                print("Flag value {} not forcing CPU mode".format(tmp))
#        except KeyError:
#            # We never set the flag
#            if 'CUDA_VISIBLE_DEVICES' in envKeys:
#                if int(os.environ['CUDA_VISIBLE_DEVICES']) < 0:
#                    raise ValueError("ForceCPU")
#            try:
#                global DARKNET_FORCE_CPU
#                if DARKNET_FORCE_CPU:
#                    raise ValueError("ForceCPU")
#            except NameError as cpu_error:
#                print(cpu_error)
#        if not os.path.exists(winGPUdll):
#            raise ValueError("NoDLL")
#        lib = CDLL(winGPUdll, RTLD_GLOBAL)
#    except (KeyError, ValueError):
#        hasGPU = False
#        if os.path.exists(winNoGPUdll):
#            lib = CDLL(winNoGPUdll, RTLD_GLOBAL)
#            print("Notice: CPU-only mode")
#        else:
#            # Try the other way, in case no_gpu was compile but not renamed
#            lib = CDLL(winGPUdll, RTLD_GLOBAL)
#            print("Environment variables indicated a CPU run, but we didn't find {}. Trying a GPU run anyway.".format(winNoGPUdll))
#else:
#    lib = CDLL(os.path.join(
#        os.environ.get('DARKNET_PATH', './'),
#        "libdarknet.so"), RTLD_GLOBAL)
#lib.network_width.argtypes = [c_void_p]
#lib.network_width.restype = c_int
#lib.network_height.argtypes = [c_void_p]
#lib.network_height.restype = c_int

#copy_image_from_bytes = lib.copy_image_from_bytes
#copy_image_from_bytes.argtypes = [IMAGE,c_char_p]

#predict = lib.network_predict_ptr
#predict.argtypes = [c_void_p, POINTER(c_float)]
#predict.restype = POINTER(c_float)

#if hasGPU:
#    set_gpu = lib.cuda_set_device
#    set_gpu.argtypes = [c_int]

#init_cpu = lib.init_cpu

#make_image = lib.make_image
#make_image.argtypes = [c_int, c_int, c_int]
#make_image.restype = IMAGE

#get_network_boxes = lib.get_network_boxes
#get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int), c_int]
#get_network_boxes.restype = POINTER(DETECTION)

#make_network_boxes = lib.make_network_boxes
#make_network_boxes.argtypes = [c_void_p]
#make_network_boxes.restype = POINTER(DETECTION)

#free_detections = lib.free_detections
#free_detections.argtypes = [POINTER(DETECTION), c_int]

#free_batch_detections = lib.free_batch_detections
#free_batch_detections.argtypes = [POINTER(DETNUMPAIR), c_int]

#free_ptrs = lib.free_ptrs
#free_ptrs.argtypes = [POINTER(c_void_p), c_int]

#network_predict = lib.network_predict_ptr
#network_predict.argtypes = [c_void_p, POINTER(c_float)]

#reset_rnn = lib.reset_rnn
#reset_rnn.argtypes = [c_void_p]

#load_net = lib.load_network
#load_net.argtypes = [c_char_p, c_char_p, c_int]
#load_net.restype = c_void_p

#load_net_custom = lib.load_network_custom
#load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
#load_net_custom.restype = c_void_p

#free_network_ptr = lib.free_network_ptr
#free_network_ptr.argtypes = [c_void_p]
#free_network_ptr.restype = c_void_p

#do_nms_obj = lib.do_nms_obj
#do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

#do_nms_sort = lib.do_nms_sort
#do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

#free_image = lib.free_image
#free_image.argtypes = [IMAGE]

#letterbox_image = lib.letterbox_image
#letterbox_image.argtypes = [IMAGE, c_int, c_int]
#letterbox_image.restype = IMAGE

#load_meta = lib.get_metadata
#lib.get_metadata.argtypes = [c_char_p]
#lib.get_metadata.restype = METADATA

#load_image = lib.load_image_color
#load_image.argtypes = [c_char_p, c_int, c_int]
#load_image.restype = IMAGE

#rgbgr_image = lib.rgbgr_image
#rgbgr_image.argtypes = [IMAGE]

#predict_image = lib.network_predict_image
#predict_image.argtypes = [c_void_p, IMAGE]
#predict_image.restype = POINTER(c_float)

#predict_image_letterbox = lib.network_predict_image_letterbox
#predict_image_letterbox.argtypes = [c_void_p, IMAGE]
#predict_image_letterbox.restype = POINTER(c_float)

#network_predict_batch = lib.network_predict_batch
#network_predict_batch.argtypes = [c_void_p, IMAGE, c_int, c_int, c_int,
#                                   c_float, c_float, POINTER(c_int), c_int, c_int]
#network_predict_batch.restype = POINTER(DETNUMPAIR)



def load_images(images_path):
    """
    If image path is given, return it directly
    For txt file, read it and return each line as image path
    In other case, it's a folder, return a list with names of each
    jpg, jpeg and png file
    """
    input_path_extension = images_path.split('.')[-1]
    if input_path_extension in ['jpg', 'jpeg', 'png']:
        return [images_path]
    elif input_path_extension == "txt":
        with open(images_path, "r") as f:
            return f.read().splitlines()
    else:
        return glob.glob(
            os.path.join(images_path, "*.jpg")) + \
            glob.glob(os.path.join(images_path, "*.png")) + \
            glob.glob(os.path.join(images_path, "*.jpeg"))


def prepare_batch(lib, images, network, channels=3):
    #width = darknet.network_width(network)
    #height = darknet.network_height(network)
    width = network_width(lib, network)
    height = network_height(lib, network)


    darknet_images = []
    for image in images:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (width, height),
                                   interpolation=cv2.INTER_LINEAR)
        custom_image = image_resized.transpose(2, 0, 1)
        darknet_images.append(custom_image)

    batch_array = np.concatenate(darknet_images, axis=0)
    batch_array = np.ascontiguousarray(batch_array.flat, dtype=np.float32)/255.0
    #darknet_images = batch_array.ctypes.data_as(darknet.POINTER(darknet.c_float))
    darknet_images = batch_array.ctypes.data_as(POINTER(c_float))
    #return darknet.IMAGE(width, height, channels, darknet_images)
    return IMAGE(width, height, channels, darknet_images)



def image_detection(lib,image_path, network, class_names, class_colors, thresh):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect

    make_image = lib.make_image
    make_image.argtypes = [c_int, c_int, c_int]
    make_image.restype = IMAGE

    copy_image_from_bytes = lib.copy_image_from_bytes
    copy_image_from_bytes.argtypes = [IMAGE,c_char_p]
    
    free_image = lib.free_image
    free_image.argtypes = [IMAGE]

    #width = darknet.network_width(network)
    #height = darknet.network_height(network)
    width = network_width(lib, network)
    height = network_height(lib, network)
    image = cv2.imread(image_path)
    #print(image.shape[0])
    height = image.shape[0]
    width = image.shape[1]
    #print(width)
    #print(height)
    #darknet_image = darknet.make_image(width, height, 3)
    darknet_image = make_image(width, height, 3)


    image = cv2.imread(image_path)
    #print(image.shape)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    #darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    #detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    #darknet.free_image(darknet_image)
    #image = darknet.draw_boxes(detections, image_resized, class_colors)
    copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = detect_image(lib, network, class_names, darknet_image, thresh=thresh)
    free_image(darknet_image)
    image = draw_boxes(detections, image_resized, class_colors)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections


def batch_detection(network, images, class_names, class_colors,
                    thresh=0.25, hier_thresh=.5, nms=.45, batch_size=4):
    image_height, image_width, _ = check_batch_shape(images, batch_size)
    darknet_images = prepare_batch(images, network)
    #batch_detections = darknet.network_predict_batch(network, darknet_images, batch_size, image_width,
    #                                                 image_height, thresh, hier_thresh, None, 0, 0)
    batch_detections = network_predict_batch(network, darknet_images, batch_size, image_width,
                                                     image_height, thresh, hier_thresh, None, 0, 0)
    batch_predictions = []
    for idx in range(batch_size):
        num = batch_detections[idx].num
        detections = batch_detections[idx].dets
        if nms:
            #darknet.do_nms_obj(detections, num, len(class_names), nms)
            do_nms_obj(detections, num, len(class_names), nms)
        #predictions = darknet.remove_negatives(detections, class_names, num)
        #images[idx] = darknet.draw_boxes(predictions, images[idx], class_colors)
        predictions = remove_negatives(detections, class_names, num)
        images[idx] = draw_boxes(predictions, images[idx], class_colors)
        batch_predictions.append(predictions)
    #darknet.free_batch_detections(batch_detections, batch_size)
    free_batch_detections(batch_detections, batch_size)
    return images, batch_predictions


def image_classification(lib, image, network, class_names):
    #width = darknet.network_width(network)
    #height = darknet.network_height(network)
    width = network_width(lib, network)
    height = network_height(lib, network)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                                interpolation=cv2.INTER_LINEAR)
    #darknet_image = darknet.make_image(width, height, 3)
    #darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    #detections = darknet.predict_image(network, darknet_image)
    darknet_image = make_image(width, height, 3)
    copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = predict_image(network, darknet_image)
    predictions = [(name, detections[idx]) for idx, name in enumerate(class_names)]
    #darknet.free_image(darknet_image)
    free_image(darknet_image)
    return sorted(predictions, key=lambda x: -x[1])


def convert2relative(image, bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h = bbox
    height, width, _ = image.shape
    return x/width, y/height, w/width, h/height


def save_annotations(name, image, detections, class_names):
    """
    Files saved with image_name.txt and relative coordinates
    """
    file_name = name.split(".")[:-1][0] + ".txt"
    with open(file_name, "w") as f:
        for label, confidence, bbox in detections:
            x, y, w, h = convert2relative(image, bbox)
            label = class_names.index(label)
            f.write("{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(label, x, y, w, h, float(confidence)))


def mainPredict(image, path, modelName, userDict , graph, sess):
    data = {}
    data['isSuccess'] = 'true'
    data['ErrorMsg'] = ''
    data['result'] = []
    try:
        path = bytes(path, 'ascii')
        network = userDict['net']
        class_names = userDict['meta']
        class_colors = userDict['class_colors']
        lib = userDict['lib']
        imagepath = path.decode() + '/original.' + userDict["FileExtension"]#image.format.lower()
        #imagepath = os.path.abspath(os.getcwd()) + '/original.' + image.format.lower()
        image.save(imagepath)
        image, r = image_detection(lib, imagepath, network, class_names, class_colors, .25)
        #print(r)
        #r = detect(net, meta, bytes(imagepath, 'ascii'))   
        for cls, score, bbox in r:
            x1 = round((bbox[0] * 2 - bbox[2]) / 2)
            y1 = round((bbox[1] * 2 - bbox[3]) / 2)
            x2 = round((bbox[0] * 2 + bbox[2]) / 2)
            y2 = round((bbox[1] * 2 + bbox[3]) / 2)  
            obj = {
                        'class':cls,
                        'classResult':[{'score': str(score),'BoundingBox':[str(x1), str(y1), str(x2), str(y2)],'Value':[str(x1), str(y1), str(x2), str(y2)]}]
                    }
            if len(data['result']) == 0:
                data['result'].append(obj)
            else:
                for obj in data['result']:
                    if obj['class'] == cls:
                        obj['classResult'].append({'score': str(score),'BoundingBox':[str(x1), str(y1), str(x2), str(y2)],'Value':[str(x1), str(y1), str(x2), str(y2)]})
                    else:
                        data['result'].append(obj)
        print(json.dumps(data))
    except Exception as e:
        print(e)
        error_class = e.__class__.__name__  # 取得錯誤類型
        detail = e.args[0]  # 取得詳細內容
        cl, exc, tb = sys.exc_info()  # 取得Call Stack
        lastCallStack = traceback.extract_tb(tb)[-1]  # 取得Call Stack的最後一筆資料
        fileName = lastCallStack[0] #取得發生的檔案名稱
        lineNum = lastCallStack[1]  # 取得發生的行號
        funcName = lastCallStack[2]  # 取得發生的函數名稱
        errorMsg = "File \"{}\", line {}, in {}: [{}] {}".format(fileName, lineNum, funcName, error_class, detail)
        data['isSuccess']= 'false'
        data['ErrorMsg']= str(errorMsg)
    finally:
        return json.dumps(data)

def mainPreLoadModel(path):
    hasGPU = True
    lib = CDLL(path + "/libdarknet.so", RTLD_GLOBAL)
    lib.network_width.argtypes = [c_void_p]
    lib.network_width.restype = c_int
    lib.network_height.argtypes = [c_void_p]
    lib.network_height.restype = c_int

    predict = lib.network_predict_ptr
    predict.argtypes = [c_void_p, POINTER(c_float)]
    predict.restype = POINTER(c_float)

    if hasGPU:
        set_gpu = lib.cuda_set_device
        set_gpu.argtypes = [c_int]

    init_cpu = lib.init_cpu

    make_network_boxes = lib.make_network_boxes
    make_network_boxes.argtypes = [c_void_p]
    make_network_boxes.restype = POINTER(DETECTION)

    free_batch_detections = lib.free_batch_detections
    free_batch_detections.argtypes = [POINTER(DETNUMPAIR), c_int]

    free_ptrs = lib.free_ptrs
    free_ptrs.argtypes = [POINTER(c_void_p), c_int]

    network_predict = lib.network_predict_ptr
    network_predict.argtypes = [c_void_p, POINTER(c_float)]

    reset_rnn = lib.reset_rnn
    reset_rnn.argtypes = [c_void_p]

    load_net = lib.load_network
    load_net.argtypes = [c_char_p, c_char_p, c_int]
    load_net.restype = c_void_p

    free_network_ptr = lib.free_network_ptr
    free_network_ptr.argtypes = [c_void_p]
    free_network_ptr.restype = c_void_p

    do_nms_obj = lib.do_nms_obj
    do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

    letterbox_image = lib.letterbox_image
    letterbox_image.argtypes = [IMAGE, c_int, c_int]
    letterbox_image.restype = IMAGE

    load_image = lib.load_image_color
    load_image.argtypes = [c_char_p, c_int, c_int]
    load_image.restype = IMAGE

    rgbgr_image = lib.rgbgr_image
    rgbgr_image.argtypes = [IMAGE]

    predict_image_letterbox = lib.network_predict_image_letterbox
    predict_image_letterbox.argtypes = [c_void_p, IMAGE]
    predict_image_letterbox.restype = POINTER(c_float)

    network_predict_batch = lib.network_predict_batch
    network_predict_batch.argtypes = [c_void_p, IMAGE, c_int, c_int, c_int,
                                       c_float, c_float, POINTER(c_int), c_int, c_int]
    network_predict_batch.restype = POINTER(DETNUMPAIR)

    resultData = {}
    resultData['isSuccess'] = 'true'
    resultData['result'] = {}
    resultData['ErrorMsg'] = ""
    modelDict = {}
    try:
        #path = '/content/darknet_test/darknet/testfolder/'
        #path = bytes(path, 'ascii')
        modelname = ''
        for filename in os.listdir(path):
            if filename.lower().endswith('.weights'):
                modelname = os.path.splitext(filename)[0]
        #network, class_names, class_colors = darknet.load_network(
        #    path + "/yolov4.cfg",
        #    path + "/obj.data",
        #    path + "/" + modelname + ".weights",
        #    batch_size=1
        #)
        fin = open(path + 'obj.data', 'rt')
        data = fin.read()
        data = data.replace('/mnt/2c67bd82-3031-40f8-8f53-58564ba23509/Graphics/yolo/test/', path).replace('cfg/' + modelname,'')
        fin.close()
        fin = open(path + 'obj.data', 'wt')
        fin.write(data)
        fin.close()
        network, class_names, class_colors = load_network(
            lib,
            path + "/yolov4.cfg",
            path + "/obj.data",
            path + "/" + modelname + ".weights",
            batch_size=1
        )
        #net = load_net(path + b"/yolov3.cfg", path + b"/" + modelname + b".weights", 0)
        #meta = load_meta(path + b"/obj.data")
        modelDict['net'] = network
        modelDict['meta'] = class_names
        modelDict['class_colors'] = class_colors
        modelDict['lib'] = lib
        resultData['result'] = modelDict
    except Exception as e:
        print(e)
        error_class = e.__class__.__name__  # 取得錯誤類型
        detail = e.args[0]  # 取得詳細內容
        cl, exc, tb = sys.exc_info()  # 取得Call Stack
        lastCallStack = traceback.extract_tb(tb)[-1]  # 取得Call Stack的最後一筆資料
        fileName = lastCallStack[0]  # 取得發生的檔案名稱
        lineNum = lastCallStack[1]  # 取得發生的行號
        funcName = lastCallStack[2]  # 取得發生的函數名稱
        errorMsg = "File \"{}\", line {}, in {}: [{}] {}".format(fileName, lineNum, funcName, error_class, detail)
        resultData['isSuccess'] = 'false'
        resultData['ErrorMsg'] = str(errorMsg)
    return resultData

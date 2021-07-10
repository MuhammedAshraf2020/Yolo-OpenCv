import os
import argparse

ap = argparse.ArgumentParser()

def load_weights(v):
    current_path = os.getcwd()
    dictroy_path = os.path.join(current_path , "model_data")
    os.makedirs(dictroy_path)
    os.system("cd {dir}".format(dir = dictroy_path))
    os.system("wget http://pjreddie.com/media/files/yolov{x}.weights".format(x = v))
    os.system("wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov{x}.cfg".format(x = v))
    print("weights have just loaded succsesfuly!")


ap.add_argument('--version' , required = True , help = 'Version of yolo')

args = ap.parse_args()

load_weights(args.version)
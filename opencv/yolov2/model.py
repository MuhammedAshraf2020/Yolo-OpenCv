import cv2
import argparse
import numpy as np
import os

currentWorkDirectory = os.getcwd() + "/" + "model_data/"

parser = argparse.ArgumentParser()
parser.add_argument("--input" , help = "Image Path")

args = parser.parse_args()

# Read labels file
labels_path = currentWorkDirectory + "labels.txt"

# How I trust of this anchor is a box
_confidence =  0.20

model  = currentWorkDirectory + "yolov2.weights"
config = currentWorkDirectory + "yolov2.cfg"

# Read Classes
classes = None
with open(labels_path , "r") as file:
	classes = file.read().rstrip("\n").split("\n")

print("Classes is Done")

net = cv2.dnn.readNetFromDarknet(config , model)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

winName = "YOLO V2"
cv2.namedWindow(winName , cv2.WINDOW_NORMAL)

frame = cv2.imread(args.input)
height , width , _  = frame.shape
axis = (height , width)
blob  = cv2.dnn.blobFromImage(frame , 1.0/255 , (416 , 416) , True , crop = False)

net.setInput(blob)
predictions = net.forward()
probability_index = 5

def rescale_preds(prediction , axis):
  h , w = axis
  axis  = np.array([w , h , w , h])
  axis  = prediction * axis
  xys   = axis[:2].flatten()
  whs   = axis[2:].flatten()
  xy1   = xys - whs * 0.5
  xy2   = xys + whs * 0.5
  return np.array(xy1 , dtype = np.int32) , np.array(xy2 , dtype = np.int32)


for i in range(predictions.shape[0]):
	prop_arr = predictions[i][probability_index:]
	class_index = prop_arr.argmax(axis = 0)
	confidence  = prop_arr[class_index]
	if confidence > _confidence:
		(x1 , y1) , (x2 , y2) = rescale_preds(predictions[i][:4] , axis)
		cv2.rectangle(frame , (x1,y1),(x2,y2) , (25,215,55) , 2)
		cv2.putText(frame , classes[class_index] + " " +"{0:.1f}".format(confidence),(x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2,cv2.LINE_AA)

cv2.imshow(winName , frame)
if (cv2.waitKey() >= 0):
    cv2.destroyAllWindows()


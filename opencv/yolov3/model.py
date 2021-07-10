import cv2
import argparse
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument("--input" , help = "Input the path of the image")
args = parser.parse_args()

CurrentPath = os.getcwd() + "/" + "model_data" + "/"

labels = CurrentPath + "labels.txt"

model   = CurrentPath  + "yolov3.weights"
config  = CurrentPath  + "yolov3.cfg"

net = cv2.dnn.readNet(model , config)

classes = None

with open(labels , "r") as f:
	classes = [line.strip() for line in f.readlines()]

LayersNames   = net.getLayerNames()
output_layers = [LayersNames[YoloOut[0] - 1] for YoloOut in net.getUnconnectedOutLayers()]

colors = np.random.uniform(0 , 255 , size = (len(classes) , 3))

img = cv2.imread(args.input)
h , w , ch = img.shape

blob = cv2.dnn.blobFromImage(img , 1.0/255.0 , (416 , 416) , (0,0,0) , True , crop = False)

net.setInput(blob)

outs = net.forward(output_layers)

classes_ids = []
confidences = []
boxes = []
conf_threshold = 0.3
nms_threshold = 0.6
for out in outs:
	for detection in out:
		scores = detection[5:]
		class_id = np.argmax(scores)
		confidence = scores[class_id]
		if confidence > 0.3:
			center_x = int(detection[0] * w)
			center_y = int(detection[1] * h)
			w_box = int(detection[2] * w)
			h_box = int(detection[3] * h)
			x_min = (center_x - w_box / 2)
			y_min = (center_y - h_box / 2)
			boxes.append([x_min , y_min , w_box , h_box])
			confidences.append(float(confidence))
			classes_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes , confidences , conf_threshold , nms_threshold )

font = cv2.FONT_HERSHEY_SIMPLEX

for i in range(len(boxes)):
	if i in indexes:
		x , y , w , h = boxes[i]
		x , y , w , h = np.array([x , y , w , h] , dtype = np.int32 )
		label = str(classes[classes_ids[i]])
		color = colors[classes_ids[i]]
		cv2.rectangle(img , (x , y ) , ( x + w , y + h ) , color , 2)
		cv2.putText(img , label , (x - 10 , y - 10) , font ,  0.8 , color , 1)

cv2.imwrite("Image{}.jpg".format(np.random.randint(0 , 100 , 1)) , img )
cv2.waitKey(0)
cv2.destroyAllWindows()


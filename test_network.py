# USAGE
# python test_network.py --model santa_not_santa.model --image images/examples/santa_01.png

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2


label_lists = {"pomfret-cropped": 0, "thunnus": 1, "YellowCroaker": 2, "notOurFishCropped": 3}
num_labels = len(label_lists)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

# load the image
image = cv2.imread(args["image"])
orig = image.copy()

# pre-process the image for classification
image = cv2.resize(image, (28, 28))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# load the trained convolutional neural network
print("[INFO] loading network...")
model = load_model(args["model"])

# classify the input image

output = imutils.resize(orig, width=400)
#(notSanta, santa) = model.predict(image)[0]
probs = model.predict(image)[0]

for specie, index in label_lists.items():
	label = "{}: {:.2f}%".format(specie, probs[index] * 100)
	cv2.putText(output, label, (10, 25+20*index),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
# build the label
#label = "Santa" if santa > notSanta else "Not Santa"
#proba = santa if santa > notSanta else notSanta
#label = "{}: {:.2f}%".format(label, proba * 100)



# draw the label on the image
#output = imutils.resize(orig, width=400)
#cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
#	0.7, (0, 255, 0), 2)

# show the output image
cv2.imshow("Output", output)
cv2.waitKey(0)
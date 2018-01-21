from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import numpy as np
import imutils
import cv2
import os

def flatten_image(img):
	return img.flatten()

print("Nuskaitom nuotraukas")
imagePaths = list(paths.list_images("fnt"))

features = []
labels = []

for (i, imagePath) in enumerate(imagePaths):
	image = cv2.imread(imagePath)
	label = imagePath.split(os.path.sep)[-1].split(".")[0]
	hist = flatten_image(image)
	features.append(hist)
	labels.append(label)

le = LabelEncoder()
labels = le.fit_transform(labels)	

print "Padalinam nuskaitytas nuotraukas, kad 75% sudarytu apmokymui, o 25% testavimui"
trainData, testData, trainLabels, testLabels = train_test_split(features, labels, test_size=0.25, random_state=42)

for k in range(1, 11):
	print "K = %d" % (k)
	model = KNeighborsClassifier(n_neighbors=k)
	model.fit(trainData, trainLabels)
	predictions = model.predict(testData)
	print(classification_report(testLabels, predictions, target_names=le.classes_))
	
for p in range(1, 5):
	print "Polinomo laipsnis = %d" % (p)
	model = SVC(kernel='poly', degree=p, gamma=1)
	model.fit(trainData, trainLabels)
	predictions = model.predict(testData)
	print(classification_report(testLabels, predictions, target_names=le.classes_))
	
for l in range(1, 11):	
	print "Sluoksniu skaicius = %d" % (l*100)
	model = MLPClassifier(hidden_layer_sizes=(l*100, ))
	model.fit(trainData, trainLabels)
	predictions = model.predict(testData)
	print(classification_report(testLabels, predictions, target_names=le.classes_))
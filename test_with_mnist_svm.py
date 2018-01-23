from sklearn.svm import SVC
import pandas as pd
import numpy as np

print("Nuskaitom pradinius duomenis")

train  = pd.read_csv('train.csv', dtype='uint8')
test  = pd.read_csv('test.csv', dtype='uint8')
label = train["label"]

train = train.drop("label",1)

print "Klasifikuojam"
knn = SVC(kernel='poly', degree=1, gamma=1)
knn.fit(train, label)
print "Spejam"
predictions = knn.predict(test)

print "Saugom"
np.savetxt('resultsSVM.csv', 
           np.c_[range(1,len(test)+1),predictions], 
           delimiter=',', 
           header = 'ImageId,Label', 
           comments = '', 
           fmt='%d')
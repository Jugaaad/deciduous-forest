import pandas as pd 
import numpy as np
from random import randint,sample,seed,shuffle
from sklearn.preprocessing import normalize

from RandomForest import RandomForest

seed(1)

df = pd.read_excel('~/Desktop/t1.xlsx')
df1 = [df['Age'] , df['Week in labour'],df['MCHb'],df['trimester'],df['Estradiol'],df['HCG level'],df['PAPP-A'],df['Glucose level'],df['diastolicBP'],df['Iron'],df['Folates'],df['Calcium']]
df1 = pd.DataFrame(df1)
df1 = df1.loc[:,:118].T
X = np.array(df1)
X = normalize(X)


df2 = [df['label']]
df2 = pd.DataFrame(df2)
label = df2.loc[:,:118].T
label = np.array(label)
Y = normalize(label)

train_X = X[:90]
train_Y = Y[:90]

test_X = X[90:]
test_Y = Y[90:]

classifcation_dicationary = {0:'N/A',8:'Premature birth',12:'Gestational Diabetes',2:'Diabetes',3:'Heart Problem', 4:'Anemic', 5:'Aasthma',1:'HIV',6:'Hypertensive disorder',11:'Second pregnancy',10:'Genetic disorder',9:'Obese',7:'Abortion'}

clf = RandomForest(featurecount=12, forestsize=600, batchsize=2)
clf.fit(train_X , train_Y)
acc = 0.0
tcc = 0.0

for i in range(len(test_X)):
	predicted = clf.prediction(test_X[i])
	#print(classifcation_dicationary[predicted])
	if test_Y[i] == predicted:
		acc+=1
	tcc+=1

print(acc/tcc)



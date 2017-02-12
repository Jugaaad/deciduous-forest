import pandas as pd 
import numpy as np
from random import randint,sample,seed,shuffle
from sklearn.preprocessing import normalize

from RandomForest import RandomForest

seed(1)

df = pd.read_excel('~/Desktop/t1.xlsx')
df1 = [df['Age'] , df['Week in labour'],df['MCHb'],df['trimester'],df['Estradiol'],df['PAPP-A'],df['Glucose level'],df['diastolicBP'],df['Iron'],df['Folates'],df['Calcium']]
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

df1 = [df['HCG level']]
df1 = pd.DataFrame(df1)
df1 = df1.loc[:,:90].T
train_X_feature = np.array(df1)[:90]
train_X_feature = train_X_feature / np.max(train_X_feature)
train_X_feature = [i[0] for i in train_X_feature]
train_X_feature = np.array(train_X_feature)

df1 = [df['Age'] , df['Week in labour'],df['MCHb'],df['trimester'],df['Estradiol'],df['HCG level'],df['PAPP-A'],df['Glucose level'],df['diastolicBP'],df['Iron'],df['Folates'],df['Calcium']]
df1 = pd.DataFrame(df1)
df1 = df1.loc[:,:118].T
X = np.array(df1)
X = normalize(X)

test_X = X[90:]
test_Y = Y[90:]

clf = RandomForest(featurecount=11, forestsize=100, batchsize=5)
clf.fit(train_X , train_Y)

acc = 0.0
tcc = 0.0

for i in range(len(test_X)):
	predicted = clf.prediction(test_X[i])
	if test_Y[i] == predicted:
		acc+=1
	tcc+=1

print(acc/tcc)

print("Adding feature and performing sacrifice")

clf.sacrifice_features(train_X_feature, sacrifice_tree=50,subtractive=False)

acc = 0.0
tcc = 0.0

for i in range(len(test_X)):
	predicted = clf.prediction(test_X[i])
	if test_Y[i] == predicted:
		acc+=1
	tcc+=1

print(acc/tcc)
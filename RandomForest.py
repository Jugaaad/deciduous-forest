from sklearn.tree import DecisionTreeClassifier,export_graphviz
from random import randint,sample,seed,shuffle
import numpy as np
from sklearn.preprocessing import normalize

class RandomForest:

	def __init__(self,featurecount=1,forestsize=20,batchsize=50,dtcrandomstate=42):
		#List of tupples
		#Each tupple consits of tree object and features which the tupple takes
		self.forest = []
		self.forest_size = forestsize
		self.batch_size = batchsize
		self.dtc_random_state = dtcrandomstate
		self.feature_count = featurecount

		self.default_count = 0
		self.X = []
		self.Y = []


	#Train the function assuming we get list of featueres we want to train on
	#List of features is always in sorted order
	def train(self,X,y,features):
		x = []
		for i in range(len(X)):
			x.append([ X[i][j] for j in features ])

		tree = DecisionTreeClassifier(random_state=self.dtc_random_state)
		tree.fit(x,y)
		self.forest.append((tree,features))

	#for single predictions
	def prediction(self,X):
		predictions = []
		for tree in self.forest:

			x = [X[i] for i in tree[1]]

			prediction = tree[0].predict(np.array(x).reshape(1,-1)) #converting list to numpy array
			predictions.append(prediction)
		predictions = [j.tolist()[0] for j in predictions]
		return max(set(predictions), key=predictions.count) #Return the label with the maximum count in the array

	#Can pass in a list to the prediction function
	def predict(self,X):
		answer = []
		for x in X:
			answer.append(self.prediction(x))
		return answer

	def fit(self,X,Y):
		self.X = X.tolist()
		self.Y = Y.tolist()

		self.default_count = len(self.X[0])
		feature_list = range(len(self.X[0]))
		Z = zip(self.X,self.Y)

		for i in range( self.forest_size ):

			#handelling feature space
			feature_space = sample(feature_list, self.feature_count)
			feature_space.sort()
			#handelling batches
			z = sample(Z,self.batch_size)
			unzipped = zip(*z)
			x,y = unzipped[0],unzipped[1]

			#training
			self.train(x,y,feature_space)

	def sacrifice_features(self,feature_vector,sacrifice_tree=10,subtractive=True):

		if subtractive: 
			sampled = sample(self.forest,sacrifice_tree)
			for i in sampled:
				del i

		new_feature_count = 1
		feature_vector=feature_vector.tolist()
		#Append to feature vector
		for i in range(len(feature_vector)):
			if type(feature_vector[i]) == type(1.1) or type(feature_vector[i]) == type(1.1): #if single feature vector
				self.X[i].append(feature_vector[i])
			else:
				for j in feature_vector[i]:
					self.X[i].append(j)
				new_feature_count = len(feature_vector[i])
		#Add new trees
		feature_list = range(len(self.X[0]))
		Z = zip(self.X,self.Y)

		for i in range( sacrifice_tree ):

			#handelling feature space
			feature_space = sample(feature_list, self.feature_count-new_feature_count)
			for nfc in range(new_feature_count):
				feature_space.append(nfc+self.default_count)
			feature_space.sort()
			#handelling batches
			z = sample(Z,self.batch_size)
			unzipped = zip(*z)
			x,y = unzipped[0],unzipped[1]

			#training
			self.train(x,y,feature_space)

	def sacrifice_data(self,data,labels,sacrifice_tree=10,subtractive=True):

		if subtractive: 
			sampled = sample(self.forest,sacrifice_tree)
			for i in sampled:
				del i

		feature_list = range(len(data[0]))
		Z = zip(data,labels)

		for i in range( sacrifice_tree ):

			#handelling feature space
			feature_space = sample(feature_list, self.feature_count) 
			feature_space.sort()
			#handelling batches
			z = sample(Z,self.batch_size)
			unzipped = zip(*z)
			x,y = unzipped[0],unzipped[1]

			#training
			self.train(x,y,feature_space)

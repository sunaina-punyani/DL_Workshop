import os
import pandas as pd
import numpy as np

# hyperparameters

ALPHA = 1.2 # learning rate
EPOCHS = 10000 # number of iterations
N_H = 5 # hidden layer size


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

def load_data(file_name):
	df = pd.read_csv(file_name)
	#assigning numbers to classes
	mapping = {'virginica':0, 'versicolor':1, 'setosa':2 }
	df = df.replace({'species':mapping})
	df = df[df['species']!=2]
	X = df[['petal_length', 'petal_width', 'sepal_length', 'sepal_width']].values.T
	X = normalized(X) 
	Y = df[['species']].values.T
	Y = Y.astype('uint8')
	return X, Y


#################################################################################################
#################################### write the functions here ###################################
#################################################################################################

def set_layer_sizes(X, Y):
	#steps
	#initialize size of input layer using input array X
	#initialize size of output layer using output array Y
	#return both size of input and output layer
	print(X.shape, Y.shape)

	return (X.shape[0], Y.shape[0])

def initialize_parameters(n_x, n_h, n_y):
	np.random.seed(2) #do not change	
	#steps
	# n_x : size of input layer
	# n_h : size of hidden layer
	# n_y : size of output layer
	# initialize W1, W2 weight arrays and b1, b2 biases
	# create a dictionary of the parameters in the form {"W1" : W1, "b1": b1, "W2": W2, "b2": b2} and return it
	W1 = np.random.randn(n_h, n_x) * 0.01
	W2 = np.random.randn(n_y, n_h) * 0.01
	
	b1 = np.zeros((n_h, 1))
	b2 = np.zeros((n_y, 1))


	return {"W1" : W1, "b1": b1, "W2": W2, "b2": b2}
def sigmoid(Z):
	#apply the formula on input array Z and return the output
	return 1 / (1 + np.exp(-Z))


def forward_propagation(X, parameters):
	#steps
	#use the weights and biases from the parameters dictionary to calculate Z1, A1, Z2, A2
	#return a dictionary of the form {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
	Z1 = np.dot(parameters['W1'], X)
	A1 = np.tanh(Z1)

	Z2 = np.dot(parameters['W2'], A1)
	A2 = sigmoid(Z2)

	return {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}


def compute_cost(A2, Y, parameters):
	#apply the cost function formula and return the cost
	return (-1 / Y.shape[1]) * ( np.sum(Y * np.log(A2)) + np.sum((1 - Y) * np.log(1 - A2)) )

def backward_propagation(parameters, cache, X, Y):
	#steps
	#cache contains the dictionary obtained from forward propogation containing A1, Z1, A2, Z2
	#return the dictionary containing the gradients dZ1, dW1, db1, dZ2, dW2, db2
	dZ2 = cache['A2'] - Y
	dW2 = (1/Y.shape[1]) * (np.dot(dZ2, cache['A1'].T))
	db2 = (1/Y.shape[1]) * np.sum(dZ2, axis = 1, keepdims = True)
	dZ1 = np.dot(parameters['W2'].T, dZ2) * (1 - np.tanh(cache['Z1']) ** 2)
	dW1 = (1/Y.shape[1]) * (np.dot(dZ1, X.T))
	db1 = (1/Y.shape[1]) * np.sum(dZ1, axis = 1, keepdims = True)

	return {'db1': db1, 'dW1': dW1,'dZ1': dZ1, 'db2': db2, 'dW2': dW2,'dZ2': dZ2 }


def update_parameters(parameters, grads, alpha):
	#steps	
	#update the parameters using the gradients obtained from back propogation and the learning rate
	#return the dictionary of updated parameters
	W1 = parameters['W1'] - alpha * grads['dW1']
	W2 = parameters['W2'] - alpha * grads['dW2']

	b1 = parameters['b1'] - alpha * grads['db1']
	b2 = parameters['b2'] - alpha * grads['db2']

	return {'W1': W1, 'W2': W2, 'b1': b1, 'b2': b2}


def predict(X, parameters):
	# use forward propogation to calculate the probability at output layer
	#return thus calculated y_predicted

	return forward_propagation(X, parameters)['A2']


#################################################################################################
#################################### You are ready to train! ####################################
#################################################################################################

def train(X, Y, n_h, epochs, alpha):
    
	n_x = set_layer_sizes(X, Y)[0]
	n_y = set_layer_sizes(X, Y)[1]

	parameters = initialize_parameters(n_x, n_h, n_y)
	W1 = parameters["W1"]
	b1 = parameters["b1"]
	W2 = parameters["W2"]
	b2 = parameters["b2"]

    # gradient descent
	for i in range(0, epochs+1):
		cache = forward_propagation(X, parameters)  # forward propogation
		cost = compute_cost(cache["A2"], Y, parameters)  # cost function
		grads = backward_propagation(parameters, cache, X, Y)  # backward propogation
		parameters = update_parameters(parameters, grads, alpha)

		if i % 500 == 0:
			print("Iteration: %i  Cost: %f" % (i, cost))

	return parameters


def calculate_accuracy(Y, Y_predicted):
	Y_predicted = np.squeeze(Y_predicted)
	Y = np.squeeze(Y)
	valid = 0
	total = 0
	for idx, val in enumerate(Y_predicted):
		if(val > 0.5):
			if(Y[idx] == 1):
				valid+=1
		else:
			if(Y[idx] == 0):
				valid+=1
		total+=1

	return valid/total


def main():
	X, Y = load_data('./iris.csv')
	parameters = train(X, Y, N_H, EPOCHS, ALPHA)
	Y_predicted = predict(X, parameters)
	accuracy = calculate_accuracy(Y, Y_predicted)
	print("Accuracy: "+ str(accuracy)) 

if __name__ == '__main__':
	main()


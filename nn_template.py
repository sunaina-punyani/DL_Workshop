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
	pass #remove this line once you finish writing

def initialize_parameters(n_x, n_h, n_y):
	np.random.seed(2) #do not change	
	#steps
	# n_x : size of input layer
	# n_h : size of hidden layer
	# n_y : size of output layer
	# initialize W1, W2 weight arrays and b1, b2 biases
	# create a dictionary of the parameters in the form {"W1" : W1, "b1": b1, "W2": W2, "b2": b2} and return it
	pass #remove this line once you finish writing

def sigmoid(Z):
	#apply the formula on input array Z and return the output
	pass #remove this line once you finish writing


def forward_propagation(X, parameters):
	#steps
	#use the weights and biases from the parameters dictionary to calculate Z1, A1, Z2, A2
	#return a dictionary of the form {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
	pass #remove this line once you finish writing

def compute_cost(A2, Y, parameters):
	#apply the cost function formula and return the cost
	pass #remove this line once you finish writing

def backward_propagation(parameters, cache, X, Y):
	#steps
	#cache contains the dictionary obtained from forward propogation containing A1, Z1, A2, Z2
	#return the dictionary containing the gradients dZ1, dW1, db1, dZ2, dW2, db2
	pass #remove this line once you finish writing

def update_parameters(parameters, grads, alpha):
	#steps	
	#update the parameters using the gradients obtained from back propogation and the learning rate
	#return the dictionary of updated parameters
	pass #remove this line once you finish writing

def predict(X, parameters):
	# use forward propogation to calculate the probability at output layer
	#return thus calculated y_predicted
	pass #remove this line once you finish writing



#################################################################################################
#################################### You are ready to train! ####################################
#################################################################################################

'''
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
		cache = forward_propogation(X, parameters)  # forward propogation
		cost = compute_cost(cache["A2"], Y, parameters)  # cost function
		grads = backward_propogation(parameters, cache, X, Y)  # backward propogation
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


'''





import  numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def feature_scaling  (data_frame):
    my_data = (data_frame - data_frame.mean()) / data_frame.std()
    return my_data


def add_bias_term(x):
    x0 = np.ones([x.shape[0],1])
    x = np.concatenate((x0,x),axis=1)
    return x

def sigmoid(z):
    g_z = (1/(1+np.exp(-z)))
    return g_z



def get_data(datafile):
    my_data = pd.read_csv(datafile, names=["exam1", "exam2", "degree"])
    my_data = feature_scaling(my_data)
    x = my_data.iloc[:,:-1].values
    x = add_bias_term(x)
    y = my_data.iloc[:,-1].values
    m = y.shape[0]
    return [x,y,m]



def initial_weights(x):
    weights = np.zeros([1,x.shape[1]])
    return weights


def initial_hyper_parameters():
    number_of_iterations  = 1000
    learning_rate = 0.01
    return [number_of_iterations,learning_rate]


def predict_value_of  (feature_vector,weights):
    z = np.dot(weights,feature_vector)
    guess = sigmoid(z)
    return guess


def compute_cost(y_guessed, y_real,training_set_length):
    first = y_real*np.log(y_guessed)
    second = (1-y_real)*np.log(1-y_guessed)
    cost = -(1/training_set_length)* np.sum( first +second )
    return cost



def calculate_hypothesis(x,weights,training_set_length):
    hypothesis = np.zeros(training_set_length)
    for i in range (training_set_length):
        hypothesis[i] = predict_value_of(x[i],weights)
    return hypothesis


def batch_gradient_decent (x,y,weights,learning_rate,training_set_length):
    h = calculate_hypothesis(x,weights,training_set_length)
    weights  = weights - learning_rate * (1/training_set_length) * np.sum(x * (h - y))
    return weights

def stochastic_gradient_decent(x,y,weights,learning_rate,training_set_length):
    hypothesis = np.zeros(training_set_length)
    for i in range (training_set_length):
        hypothesis[i] = predict_value_of(x[i],weights)
        weights = weights - learning_rate * (1/training_set_length) * np.sum(x[i] * (hypothesis[i] - y[i]))
    return [weights,hypothesis]



def gradient_decent_runner (x,y,weights,number_of_iterations,learning_rate,training_set_length):
    costs = np.zeros(number_of_iterations)
    for i in range (number_of_iterations):
        weights ,y_guessed  = stochastic_gradient_decent(x,y,weights,learning_rate,training_set_length)
        costs[i] = compute_cost(y_guessed,y,training_set_length)
    return [weights,costs]


def check_gradient_decent (cost,itrs):
    plt.figure(figsize=(10,6))
    plt.plot(range(itrs),cost,'rx')
    plt.xlabel("number of iterations ")
    plt.ylabel("cost function ")
    plt.show()

def run_algorthim(datafile):
    X ,Y , M = get_data(datafile)
    theta = initial_weights(X)
    itrs , alpha = initial_hyper_parameters()
    theta , j = gradient_decent_runner(X,Y,theta,itrs,alpha,M)
    check_gradient_decent(j,itrs)

if __name__ == '__main__':
    run_algorthim('../data/ex2data1.txt')
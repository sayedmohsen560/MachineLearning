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


def get_data(datafile):
    my_data = pd.read_csv(datafile, names=["size", "bedroom", "price"])
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

def calculate_hypothesis (feature_vector,weights):
    y_guessed = np.dot(weights,feature_vector)
    return y_guessed



def compute_cost(y_guessed, y_real,training_set_length):
    cost = (1/2*training_set_length) * np.sum(np.square(y_guessed-y_real))
    return cost


def gradient_decent(x,y,weights,learning_rate,training_set_length):
    y_guessed = np.zeros(training_set_length)
    for i in range (training_set_length):
        y_guessed[i] = calculate_hypothesis(x[i], weights)
        weights = weights - learning_rate * np.sum(x[i] * (y_guessed[i] - y[i])) *(1/training_set_length)
    return [y_guessed,weights]

def gradient_decent_runner (x,y,weights,number_of_iterations,learning_rate,training_set_length):
    cost = np.zeros(number_of_iterations)
    for i in range(number_of_iterations):
        y_guessed ,weights = gradient_decent(x,y,weights,learning_rate,training_set_length)
        cost[i] = compute_cost(y_guessed, y, training_set_length)
       # print(cost[i])
        print(weights)
    return [weights, cost]





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

    run_algorthim('../data/home.txt')

import  numpy as np
import matplotlib.pyplot as plt


def set_data(datafile):
    data = np.loadtxt(datafile,delimiter=",")
    x = data[:,0]
    y = data[:,1]
    m = len(y)
    return x ,y , m


def plot_data(x,y):
    plt.figure(figsize=(10,6))
    plt.plot(x,y,'rx')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def calculate_hypothesis (x,weight0,weight1,training_set_length):
    hypothesis = []
    for i in range (training_set_length):
        h = weight1 *x[i] + weight0
        hypothesis.append(h)
    return hypothesis


def predict_value_of(x,weight0,weight1):
    guess = x*weight1 +weight0
    return guess


def compute_cost(y_guessed, y_real,training_set_length):
    cost = (1/2*training_set_length) * np.sum(np.square(y_guessed-y_real))
    return cost

def gradient_decent(x,y_real,y_guessed,weight0_initial,weight1_initial,learning_rate,training_set_length):
    weight0_temp = weight0_initial - learning_rate*(1/training_set_length)* np.sum(y_guessed-y_real)
    weight1_temp = weight1_initial - learning_rate*(1/training_set_length)* np.sum(y_guessed-y_real)*x
    return [weight0_temp, weight1_temp]


def stochastic_gradient_decent(x,y_real,y_guessed,weight0_initial,weight1_initial,learning_rate,training_set_length):
    hypothesis = np.zeros(training_set_length)
    weight0_temp = weight0_initial
    weight1_temp = weight1_initial
    for i in range (training_set_length):
        hypothesis[i] = predict_value_of(x[i],weight0_temp,weight1_temp)
        weight0_temp = weight0_initial - learning_rate * (1 / training_set_length) * np.sum(hypothesis[i] - y_real[i])
        weight1_temp = weight1_initial - learning_rate * (1 / training_set_length) * np.sum(hypothesis[i] - y_real[i]) * x
    return [weight0_temp,weight1_temp,hypothesis]


def run_gradient_decent(x,y,weight0_initial,weight1_initial,learning_rate,training_set_length,number_of_iterations):

    for i in range (number_of_iterations):
        y_guess= calculate_hypothesis(x, weight0_initial, weight1_initial, training_set_length)
        all_costs.append(compute_cost(y_guess, y_real,training_set_length))
        weight0,weight1 =  gradient_decent(x[i],y_real,y_guessed,weight0,weight1,learning_rate,training_set_length)


    return all_costs




def check_gradient_decent (cost,itrs):
    #print(cost)
    plt.figure(figsize=(10,6))
    plt.plot(range(itrs),cost,'rx')
    plt.xlabel("number of iterations ")
    plt.ylabel("cost function ")
    plt.show()



if __name__ == '__main__':
    X ,Y , M = set_data('../data/ex1data1.txt')
    plot_data(X,Y)
    j = run_gradient_decent(X,Y,0,0,0.2,M,100)
    check_gradient_decent(j,100)







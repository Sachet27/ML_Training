import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def cost(x1_train, x2_train, y_train, W1, W2, b):
    m= len(x1_train)
    return (0.5/m) * sum((W1*x1_train + W2* x2_train + b - y_train)**2)


def gradient_descent(x1_train, x2_train, y_train, learning_rate, epochs):
    W1= W2= b= 0
    m= len(x1_train)
    for _ in range(epochs):
        dJ_dW1= (1/m)* sum(x1_train* (W1 * x1_train+ W2 * x2_train + b - y_train))
        dJ_dW2= (1/m)* sum(x2_train* (W1 * x1_train+ W2 * x2_train + b - y_train))
        dJ_db= (1/m)* sum(W1 * x1_train+ W2 * x2_train + b - y_train)

        W1-= learning_rate*dJ_dW1
        W2-= learning_rate* dJ_dW2
        b-= learning_rate* dJ_db
    
    return W1, W2, b

def main():
    penguins= sns.load_dataset('penguins')
    penguins.dropna(subset= ['bill_length_mm', 'bill_depth_mm', 'body_mass_g'], inplace= True)
    penguins= penguins[penguins['species']=='Adelie']

    x1_train= penguins['bill_length_mm'].to_numpy()
    x2_train= penguins['bill_depth_mm'].to_numpy()
    y_train= penguins['body_mass_g'].to_numpy()

    x1_train = (x1_train - np.mean(x1_train)) / np.std(x1_train)
    x2_train = (x2_train - np.mean(x2_train)) / np.std(x2_train)
    y_train  = (y_train  - np.mean(y_train))  / np.std(y_train)


    axs= plt.axes(projection= '3d')
    axs.scatter(x1_train, x2_train, y_train)
    axs.set_xlabel('Bill Length (mm)')
    axs.set_ylabel('Bill Depth (mm)')
    axs.set_zlabel('Body Mass (g)')
    
    plt.show()


    w1, w2, b = gradient_descent(x1_train, x2_train, y_train, 0.0001, 5000)
    print(w1, w2, b)


    w_space= np.linspace(-5, 5, 1000)
    b_space= np.linspace(-5, 5, 1000)
    W, B= np.meshgrid(w_space, b_space)
    J= np.zeros_like(W)

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            J[i, j] = cost(x1_train, x2_train, y_train, W[i, j], w2, B[i, j]) 

    axs= plt.axes(projection= '3d')
    axs.plot_surface(W, B, J)
    axs.set_xlabel('Weight')
    axs.set_ylabel('Bias')
    axs.set_zlabel('Cost')
    plt.show()

if __name__=='__main__':
    main()
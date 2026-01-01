import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def linear_fit(x_train, y_train, lr, epochs):
    W= 0 
    b= 0
    m= len(x_train)

    for i in range(epochs):
        dE_dW = (-2/m)* sum(x_train*(y_train-(W*x_train+b)))
        dE_db = (-2/m)* sum(y_train-(W*x_train+b)) 
        W-= lr* dE_dW
        b-= lr*dE_db 

    return W, b

def main():
    penguins= sns.load_dataset('penguins')
    penguins.dropna(subset= ['bill_length_mm', 'body_mass_g'], inplace= True)
    x_train= penguins['bill_length_mm'].to_numpy()
    y_train= penguins['body_mass_g'].to_numpy()
    x_test= np.array([35.4, 30.8, 42.9])

    W, b= linear_fit(x_train, y_train, 0.0001, 700)
    reg_line_x= list(range(30, 60))
    reg_line_y= [W*x + b for x in reg_line_x]


    plt.scatter(x= x_train, y= y_train, marker= 'o', alpha= 0.7)
    plt.plot(reg_line_x, reg_line_y, c= 'r')
    plt.scatter(x= x_test, y= [W*x+b for x in x_test], c= 'black', lw= 1.5, marker= 'x', zorder=5)
    plt.title('Body Mass vs Bill Length Comparison')
    plt.xlabel('Bill Length (mm)')
    plt.ylabel('Body mass (g)')
    plt.show()



if __name__=='__main__':
    main()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def fit_bi_lr(x1_train, x2_train, y_train, alpha, epochs):
    W1= W2= b= 0
    m= len(x1_train)

    for i in range(epochs):
        dE_db= (-2/m)* sum(y_train- (b + W1*x1_train + W2*x2_train))
        dE_dW1= (-2/m)* sum(x1_train * (y_train- (b + W1*x1_train + W2*x2_train)))
        dE_dW2= (-2/m)* sum(x2_train* (y_train- (b + W1*x1_train + W2*x2_train)))

        b-= alpha* dE_db
        W1-= alpha* dE_dW1
        W2-= alpha* dE_dW2
    
    return (W1, W2 , b)



def main():
    planets= sns.load_dataset('planets')
    filt= planets['method']== 'Radial Velocity'
    df= planets.loc[filt]
    df.dropna(inplace= True)

    x1_train= df['mass'].to_numpy()
    x2_train= df['distance'].to_numpy()
    y_train= df['orbital_period'].to_numpy()

    # print(df)
    # plt.hist(data= df, x= 'mass')
    # plt.show()

    W1, W2, b = fit_bi_lr(x1_train, x2_train, y_train, 0.0001, 700)
    x_line= np.linspace(0, 6000, 100)
    y_line= np.linspace(0, 250, 100)
    
    X1, X2= np.meshgrid(x_line, y_line)
    Y_hat= b+ W1* X1+ W2* X2


    axs= plt.axes(projection= '3d')
    axs.scatter(x1_train, x2_train, y_train)
    axs.set_xlabel('Mass')
    axs.set_ylabel('Distance')
    axs.set_zlabel('Orbital Period')
    axs.set_title('Orbital Periods of Planets')
    axs.set_zlim(0, 4500)
    axs.set_ylim(0, 250)
    axs.set_xlim(0, 15)

    axs.plot_surface(X1, X2, Y_hat)
    plt.show()




if __name__=='__main__':
    main()
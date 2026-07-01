import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def gabriels_horn(ax):
    u= np.linspace(1, 50, 200)
    theta= np.linspace(0, 2* np.pi, 200)

    U, Theta = np.meshgrid(u, theta)
    #parameters
    X= U
    Y= np.cos(Theta) / U
    Z= np.sin(Theta) / U

    ax.plot_surface(X, Y, Z)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    plt.show()



def normal_dist_animation(start_size, stop_size, step, ax):
    for size in range(start_size, stop_size + 1, step):
        ax.clear()
        x_gaussian = np.random.normal(size= size)
        sns.histplot(x= x_gaussian, bins= 30, kde= True, ax= ax)
        ax.set_xlabel('Random Variate (z)')
        ax.set_ylabel('Frequency')
        ax.set_title('Normal Distribution')
        plt.pause(0.1)
    plt.show()



def main():
    ax= plt.axes(projection='3d')
    gabriels_horn(ax)
    
    fig, ax1= plt.subplots(1, 1, figsize= (10, 6))
    fig.suptitle('Various plots')

    normal_dist_animation(1, 2000, 100, ax1)




if __name__ == "__main__":
    main()
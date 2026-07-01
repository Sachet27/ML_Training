import numpy as np
import matplotlib.pyplot as plt


def main():
    x= np.linspace(-5, 5, 100)
    y= np.linspace(-5, 5, 100)

    X, Y= np.meshgrid(x, y)
    Z= np.exp(-(X**2 + Y**2))

    ax= plt.axes()
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title('Gaussian curve')
    
    for i in range(15):
        ax.clear()
        ax.contour(X, Y, Z, levels= i)
        plt.pause(0.2)
    plt.show()





if __name__ == "__main__":
    main()
from l2_regularized_lr_model import OneHotEncoder, StandardScaler, L2LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error, r2_score

def main():
    df= pd.read_csv('../../datasets/Housing.csv')

    #dealing with categorical data
    categorical_cols= list(df.select_dtypes('object'))

    #encoding and adding encoded cols to df
    encoder= OneHotEncoder()
    for col in categorical_cols:
        y= df[col].to_numpy()
        _, cats_to_index = encoder.fit(y)
        cats= list(cats_to_index.keys())
        new_cols= [f'{col}_{cat}' for cat in cats]
        encoded_col= encoder.transform(y)
        encoded_col_df= pd.DataFrame(encoded_col, columns= new_cols)
        # encoded_col_df= encoded_col_df.astype('category')
        df= pd.concat([df, encoded_col_df], axis=1)

    df= df.drop(columns= categorical_cols) 

    #splitting data
    X= df[['area', 'bedrooms', 'bathrooms', 'stories', 'parking',
       'mainroad_no', 'mainroad_yes', 'guestroom_no', 'guestroom_yes',
       'basement_no', 'basement_yes', 'hotwaterheating_no',
       'hotwaterheating_yes', 'airconditioning_no', 'airconditioning_yes',
       'prefarea_no', 'prefarea_yes', 'furnishingstatus_furnished',
       'furnishingstatus_semi-furnished', 'furnishingstatus_unfurnished']]
    y= df['price']

    X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 0.2, random_state= 42)

    #scaling our training data
    X_num_cols= ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
    X_cat_cols= [col for col in X_train if col not in X_num_cols]

    X_scaler= StandardScaler()
    X_train_mean, X_train_std= X_scaler.fit(X_train[X_num_cols].to_numpy())
    X_train_num_norm = X_scaler.transform(X_train[X_num_cols].to_numpy())
    X_train_norm = np.hstack([X_train_num_norm, X_train[X_cat_cols].to_numpy()]) 

    y_scaler= StandardScaler()
    y_train= y_train.to_numpy().reshape(-1, 1)
    y_train_mean, y_train_std = y_scaler.fit(y_train)
    y_train_norm= y_scaler.transform(y_train)

    # training our model
    alpha= 0.001
    lamda= 0.01
    epochs= 30000
    
    model= L2LinearRegression()
    model.fit(X_train_norm, y_train_norm, alpha, epochs, lamda)
    W, b = model.weights, model.bias
    cost_hist= model.get_cost_hist()


    #transforming test data
    X_test_num_norm = X_scaler.transform(X_test[X_num_cols].to_numpy())
    X_test_norm = np.hstack([X_test_num_norm, X_test[X_cat_cols].to_numpy()]) 

    y_test= y_test.to_numpy().reshape(-1,1)
    y_test_norm= y_scaler.transform(y_test) 

    #making predictions
    y_preds_norm= model.predict(X_test_norm)
    y_preds= y_scaler.detransform(y_preds_norm)
    
    rmse= root_mean_squared_error(y_test, y_preds)
    mpe= mean_absolute_percentage_error(y_test, y_preds)
    r2= r2_score(y_test, y_preds)
    print(f"\nRoot Mean squared error= {rmse:.2f}")
    print(f"Mean absolute percentage error= {(mpe * 100):.2f} %")
    print(f"R squared (Coefficient of Determination)= {r2:.2f}\n")    



    #plotting cost vs iterations
    fig, (ax0, ax1)= plt.subplots(1, 2, figsize= (12,4), gridspec_kw={'wspace': 0.25})
    fig.suptitle('No. of iterations vs cost',  fontfamily= 'Arial', fontsize= 16, y= 1.05)

    iters= np.arange(0, epochs, 5)
    ax0.plot(iters, cost_hist) 
    ax0.set_xlabel('Iterations')
    ax0.set_ylabel('Cost')
    ax0.set_title('Whole plot')

    scaled_start_range= int(epochs/5 - 250)

    ax1.plot(np.arange(scaled_start_range, len(cost_hist))* 5 , cost_hist[scaled_start_range:])
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Cost')
    ax1.set_title('Tail (Last 2500 iterations)')

    plt.show()

    #determining which numeric features matter the most
    print("Correlation between Housing Price and Features:")
    corr= df[['price','area', 'bedrooms', 'bathrooms', 'stories', 'parking']].corr()
    print(corr['price'].sort_values(ascending=False).iloc[1:])

    #plotting area vs price distributions
    X_num_norm = X_scaler.transform(X[X_num_cols].to_numpy())
    X_norm = np.hstack([X_num_norm, X[X_cat_cols].to_numpy()])  
    y_plot_norm= model.predict(X_norm)  

    x_plot= X['area']
    y_plot= y_scaler.detransform(y_plot_norm)   

    fig, ax= plt.subplots(1, 2, figsize= (12, 4))
    fig.suptitle("Actual vs Predicted Distribution")

    ax[0].scatter(x= X_train['area'], y= y_train, marker="o", c= "orange", alpha= 0.4, label= "Train Data")
    ax[0].scatter(x= X_test['area'], y= y_test, marker="o", c= "green", alpha= 0.4, label="Test Data")
    ax[0].set_xlabel("Area (sq. m)")
    ax[0].set_ylabel("Housing price (in $)")
    ax[0].set_title("Actual Housing price scatterplot")
    ax[0].legend()

    ax[1].scatter(x= x_plot, y= y_plot, marker="o", c= "red", alpha= 0.6, label="Predicted Data")
    ax[1].set_xlabel("Area (sq. m)")
    ax[1].set_ylabel("Housing price (in $)")
    ax[1].set_title("Predicted Housing price scatterplot")
    ax[1].legend()
    plt.show()




if __name__=='__main__':
    main()
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from KNN_model import KNN, Encoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


def main():
    df= sns.load_dataset('iris')
    
    #encoding species for checking correlation
    encoder= Encoder()
    encoded_species= encoder.fit_transform(df['species'].to_numpy()).flatten()
    df= pd.concat([df, pd.Series(encoded_species, name= "encoded_species")], axis= 1)

    # checking the correlation of each column to the target column
    print("Correlation of each column to iris species:\n")
    num_col= list(df.select_dtypes('number'))
    print(df[num_col].corr()['encoded_species'].sort_values(key=lambda val: abs(val), ascending= False))
    
    #splitting train_test_data
    X= df[['petal_length', 'petal_width', 'sepal_length']]
    y= df['species']
    X_train, X_test, y_train, y_test= train_test_split(X, y, random_state= 42)
    
    #training model
    X_train= X_train.to_numpy()
    y_train= y_train.to_numpy().reshape(-1, 1)
    test_X= X_test.to_numpy()

    model= KNN(k=3)
    model.fit(X_train, y_train)
    y_preds= model.predict(test_X).flatten()

    print("\nClassification report:\n")
    print(classification_report(y_true= y_test, y_pred= y_preds))

    #plotting confusion matrix
    cats= list(encoder.cats_to_index.keys())
    cm= confusion_matrix(y_test, y_preds)
    sns.heatmap(cm, annot= True, xticklabels=cats, yticklabels=cats, cmap= "Blues", linecolor="black", cbar= True)
    plt.title('Confusion matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    #plotting 3d distribution
    ax= plt.axes(projection= "3d")
    setosa= df['encoded_species']==0
    versicolor= df['encoded_species']==1
    virginica= df['encoded_species']==2

    ax.scatter(df['petal_length'][setosa], df['petal_width'][setosa], df['sepal_length'][setosa], c= "red", label= "Setosa")
    ax.scatter(df['petal_length'][versicolor], df['petal_width'][versicolor], df['sepal_length'][versicolor], c="green", label= "Versicolor")
    ax.scatter(df['petal_length'][virginica], df['petal_width'][virginica], df['sepal_length'][virginica], c="blue", label="Virginica")
    ax.set_xlabel('Petal length')
    ax.set_ylabel('Petal width')
    ax.set_zlabel('Sepal length')
    ax.set_title('3d Distribution')
    ax.legend()
    plt.show()   

    
    #plotting predicted 3d distribution
    fig, ax= plt.subplots(1, 2, figsize= (10, 6), subplot_kw= {'projection': '3d'})
    fig.suptitle('Comparison of True vs Predicted Labels for Test Data')
    true_setosa= y_test== 'setosa'
    true_versicolor= y_test== 'versicolor'
    true_virginica= y_test== 'virginica'

    pred_setosa= y_preds== 'setosa'
    pred_versicolor= y_preds== 'versicolor'
    pred_virginica= y_preds== 'virginica'

    ax[0].scatter(X_test['petal_length'][true_setosa], X_test['petal_width'][true_setosa], X_test['sepal_length'][true_setosa], c= "red", label= "Actual Setosa", alpha= 0.4)
    ax[0].scatter(X_test['petal_length'][true_versicolor], X_test['petal_width'][true_versicolor], X_test['sepal_length'][true_versicolor], c="green", label= "Actual Versicolor", alpha= 0.4)
    ax[0].scatter(X_test['petal_length'][true_virginica], X_test['petal_width'][true_virginica], X_test['sepal_length'][true_virginica], c="blue", label="Actual Virginica", alpha= 0.4)
    ax[0].set_xlabel('Petal length')
    ax[0].set_ylabel('Petal width')
    ax[0].set_zlabel('Sepal length')
    ax[0].set_title('Actual 3d Distribution')
    ax[0].legend()

    ax[1].scatter(X_test['petal_length'][pred_setosa], X_test['petal_width'][pred_setosa], X_test['sepal_length'][pred_setosa], label= "Predicted Setosa", c= "orange", alpha= 0.4)
    ax[1].scatter(X_test['petal_length'][pred_versicolor], X_test['petal_width'][pred_versicolor], X_test['sepal_length'][pred_versicolor], label= "Actual Versicolor", c= "green", alpha= 0.4)
    ax[1].scatter(X_test['petal_length'][pred_virginica], X_test['petal_width'][pred_virginica], X_test['sepal_length'][pred_virginica], label= "Actual Virginica", c="purple", alpha= 0.4)
    ax[1].set_xlabel('Petal length')
    ax[1].set_ylabel('Petal width')
    ax[1].set_zlabel('Sepal length')
    ax[1].set_title('Predicted 3d Distribution')
    ax[1].legend()
    plt.show()   



if __name__=='__main__':
    main()


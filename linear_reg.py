import streamlit as st 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

def hypothesis_line(coef,bias,X):
    return ((X * coef) + bias)

def make_meshgrid(x):
    a = np.arange((x[:,0].min()-1),(x[:,0].max()+1),0.01)
    b = np.arange((x[:,1].min()-1),(x[:,1].max()+1),0.01)    
    
    XX,yy = np.meshgrid(a,b)
    
    input_arr = np.array([XX.ravel(),yy.ravel()]).T
    
    return XX,yy,input_arr

plt.style.use('fivethirtyeight')

# Streamlit part
st.sidebar.header('Linear Regression')

features = st.sidebar.selectbox('Features',('2-Dimension','3-Dimension'))
st.header('Select the number of features you want to visualize ')


if features == '2-Dimension':
    fig,axs = plt.subplots()
    X,y = make_regression(n_samples=1000,n_features=1,noise = 10,random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.3, random_state=42)
    axs.scatter(X.T[0],y)
    
    origin = st.pyplot(fig)

    if st.sidebar.button('Run'):
        lin = LinearRegression().fit(X_train,y_train)
        y_pred = lin.predict(X_test)

        axs.plot(X_test,lin.predict(X_test),color = 'green')

        origin = st.pyplot(fig)
        st.subheader('Mean Absolute Error :'+ str(round(mean_absolute_error(y_test,y_pred),2)))
        st.subheader('Mean Squared Error :'+str(round(mean_squared_error(y_test,y_pred),2)))
        st.subheader('R2 Score  :'+str(round(r2_score(y_test,y_pred),2)))
        

elif features == '3-Dimension' :
    fig2 = plt.figure()
    X,y = make_regression(n_samples=1000,n_features=2,n_informative=2,noise=0.2,random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.3, random_state=42)
    
    # syntax for 3-D projection
    ax = fig2.add_subplot(111, projection='3d')
    ax.scatter(X.T[0],X.T[1],y, marker='.', color='red')

    origin2 = st.pyplot(fig2)

    if st.sidebar.button('Run'):
        lin2 = LinearRegression().fit(X_train,y_train)
        y_pred  = lin2.predict(X_test)
        coef = lin2.coef_
        bias = lin2.intercept_
        xs,ys,input_a = make_meshgrid(X)  
        zs = lin2.predict(input_a)
        ax.plot_surface(xs,ys,zs.reshape(xs.shape),alpha=0.15)
        origin2 = st.pyplot(fig2)
        
    
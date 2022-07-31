import streamlit as st 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge,Lasso,ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

def data(ax,noise):
    X,y= make_regression(n_features=1,n_samples=500,noise=noise,random_state=4)
    ax.scatter(X,y)
    return X,y

def hypothesis_line(coef,bias,X):
    return ((X * coef) + bias)

plt.style.use('fivethirtyeight')

# Streamlit part
st.sidebar.markdown('Regression Problem')
st.header('Graph representation')

regressor = st.sidebar.selectbox('Regressor',('Ridge','Lasso','Elasticnet'))
noise = float(st.number_input('Noise in data',float(0.0),float(10)))


alpha = float(st.sidebar.number_input('alpha',1))
max_iter = st.sidebar.slider('max_iter',10,500)
tol = st.sidebar.number_input('tol',(1*np.exp(-3)))
postive = st.sidebar.selectbox('Positive',('Yes','No'))

fig,axs = plt.subplots(figsize=(5,4))
X,y = data(axs,noise)
X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.3, random_state=42)

original = st.pyplot(fig)

if regressor == 'Ridge':

    solve = st.sidebar.selectbox('solver',('auto', '‘svd’', '‘cholesky’', '‘lsqr’', '‘sparse_cg’', '‘sag’', '‘saga’', 'lbfgs'),)
    if st.sidebar.button('Run'):
    
        rid = Ridge(alpha=alpha,max_iter=max_iter,tol=tol,positive=postive,solver=solve)
        rid.fit(X_train,y_train)
        coef = rid.coef_
        bias = rid.intercept_
        y_hypo=hypothesis_line(coef,bias,X_test)
        # axs.scatter(X_train,y_train)
        axs.plot(X_test,rid.predict(X_test),color='red')
        # axs.plot(X_test,y_hypo,color='green')
        original = st.pyplot(fig)
        y_pred = rid.predict(X_test)
        st.subheader('Mean Absolute Error :'+ str(round(mean_absolute_error(y_test,y_pred),2)))
        st.subheader('Mean Squared Error :'+str(round(mean_squared_error(y_test,y_pred),2)))
        st.subheader('R2 Score  :'+str(round(r2_score(y_test,y_pred),2)))

elif regressor == 'Lasso':

    selctio = st.sidebar.selectbox('selection',('cyclic', 'random'))
    if st.sidebar.button('Run'):

        las = Lasso(alpha=alpha,max_iter=max_iter,tol=tol,positive=postive,selection=selctio)
        las.fit(X_train,y_train)
        coef = las.coef_
        bias = las.intercept_
        y_hypo=hypothesis_line(coef,bias,X_test)
        # axs.scatter(X_train,y_train)
        axs.plot(X_test,las.predict(X_test),color='red')
        # axs.plot(X_test,y_hypo,color='green')
        original = st.pyplot(fig)
        y_pred = las.predict(X_test)
        st.subheader('Mean Absolute Error :'+ str(round(mean_absolute_error(y_test,y_pred),2)))
        st.subheader('Mean Squared Error :'+str(round(mean_squared_error(y_test,y_pred),2)))
        st.subheader('R2 Score  :'+str(round(r2_score(y_test,y_pred),2)))

elif regressor == 'Elasticnet':

    selctio = st.sidebar.selectbox('selection',('cyclic', 'random'))
    li_ra = st.sidebar.slider('l1_ratio',float(0.0),float(1.0))

    if st.sidebar.button('Run'):

        elas = ElasticNet(alpha=alpha,max_iter=max_iter,tol=tol,positive=postive,l1_ratio=li_ra)
        elas.fit(X_train,y_train)
        coef = elas.coef_
        bias = elas.intercept_
        y_hypo=hypothesis_line(coef,bias,X_test)
        # axs.scatter(X_train,y_train)
        axs.plot(X_test,elas.predict(X_test),color='red')
        # axs.plot(X_test,y_hypo,color='green')
        original = st.pyplot(fig)
        y_pred = elas.predict(X_test)
        st.subheader('Mean Absolute Error :'+ str(round(mean_absolute_error(y_test,y_pred),2)))
        st.subheader('Mean Squared Error :'+str(round(mean_squared_error(y_test,y_pred),2)))
        st.subheader('R2 Score  :'+str(round(r2_score(y_test,y_pred),2)))

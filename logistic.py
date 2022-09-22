import streamlit as st 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs,make_classification
from sklearn.metrics import accuracy_score,precision_score,confusion_matrix

def graph(dimens,ax):
    if dimens == 'Binary':
        X,y = make_blobs(n_features=2,centers=2,random_state=42)
        ax.scatter(X.T[0],X.T[1],c=y,cmap='rainbow')
        return X,y
    else:
        X,y = make_blobs(n_features=2,centers=4,random_state=42)
        ax.scatter(X.T[0],X.T[1],c=y,cmap='rainbow')
        return X,y
    
def make_meshgrid():
    a = np.arange((X[:,0].min()-1),(X[:,0].max()+1),0.01)
    b = np.arange((X[:,1].min()-1),(X[:,1].max()+1),0.01)    
    
    XX,yy = np.meshgrid(a,b)
    
    input_arr = np.array([XX.ravel(),yy.ravel()]).T
    
    return XX,yy,input_arr


plt.style.use('seaborn-whitegrid')

st.sidebar.markdown('LOGISTIC Regression Classifier')

dataset = st.sidebar.selectbox('Select Dataset',('Binary','Multiclass'))
penalty = st.sidebar.selectbox('Regularization',('l2','l1','elasticnet','none'))
c = float(st.sidebar.number_input('C',value=1.0))
solver = st.sidebar.selectbox(
'Solver',
('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga')
)
multiclass = st.sidebar.radio('Multiclass',('ovr','multinomial','auto'))
max_iter = int(st.sidebar.slider('Max_iter',1,500,100))
li_ratio = float(st.sidebar.number_input('l1_ratio(btw 0-1)'))

fig,axs = plt.subplots()

X,y = graph(dataset,axs)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
origin = st.pyplot(fig)

if st.sidebar.button('Run'):
    clf  = LogisticRegression(penalty=penalty,C = c,multi_class = multiclass,max_iter=max_iter,solver=solver,l1_ratio=li_ratio)
    clf.fit(X_train,y_train)
    
    y_pred = clf.predict(X_test)
    
    XX,yy,input_array = make_meshgrid()
    labels = clf.predict(input_array)
    
    plt.contourf(XX,yy,labels.reshape(XX.shape),alpha = 0.7,cmap='rainbow')
    
    plt.xlabel('Col1')
    plt.ylabel('Col2')
    origin = st.pyplot(fig)
    st.subheader('Accuracy of Logistic :'+ str(round(accuracy_score(y_test,y_pred),2)))
    st.subheader('Precision Of Logistic :'+str(round(precision_score(y_test,y_pred),2)))
    st.subheader('Confusion Matrix  :\n'+ str(confusion_matrix(y_test,y_pred)))
    
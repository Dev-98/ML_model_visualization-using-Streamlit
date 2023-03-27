from matplotlib import pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import BaggingClassifier,RandomForestClassifier
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,precision_score,f1_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs,make_classification,make_gaussian_quantiles,make_circles,make_moons

def make_graph(datas,ax):

    if datas == 'Omlet' :
        X,y = make_gaussian_quantiles(n_samples=1000,n_features=2,n_classes=2,cov=3,random_state=42)
        ax.scatter(X.T[0],X.T[1],c=y)
        return X,y
    
    elif datas == 'Moons' :
        X,y = make_moons(n_samples=1000,noise=0.1,random_state=42)
        ax.scatter(X.T[0],X.T[1],c=y)
        return X,y

    elif datas == 'Normal' :
        noise = st.number_input('Cluster',2,6,2) 
        X,y = make_blobs(n_samples=10000,n_features=2,centers=noise,random_state=42)
        ax.scatter(X.T[0],X.T[1],c=y)
        return X,y

    elif datas == 'Concentric circle' :
        X,y = make_circles(n_samples=1000,noise=0.05,random_state=42)
        ax.scatter(X.T[0],X.T[1],c=y)
        return X,y
    
    elif datas == 'Swords':
        x,y = make_classification(1000,n_features=2,n_classes=2
                          ,n_informative=2,n_redundant=0,n_clusters_per_class=1,random_state=42)
        
        ax.scatter(x.T[0],x.T[1],c=y)
        return x,y

def make_meshgrid():
    xi = np.arange((X[:,0].min()-1), (X[:,0].max()+1) ,0.01)
    yi = np.arange((X[:,1].min()-1), (X[:,1].max()+1) ,0.01)
    xx,yy = np.meshgrid(xi,yi)
    zz = np.array([xx.ravel(),yy.ravel()]).T

    return xx,yy,zz

# print(plt.style.available)
plt.style.use('Solarize_Light2')

st.title('Bagging classifier')

st.sidebar.markdown('Hyperparameters and Datasets')
data = st.sidebar.selectbox('Datasets',('Normal','Swords','Omlet','Moons','Concentric circle'))

with st.sidebar:
    form = st.form('Bagging')
    algo = form.selectbox('Choose Algoritms',('Logistic Regression','Naive Bayes','K-Nearest Neighbors','Decision Tree','Support Vector Machines','Random Forest',
        'SGDclassifier'),1)

    estm = form.number_input('N_estimators',1,100,10)
    samp = float(form.number_input('Max_Samples',0.0,1.0,1.0,0.1))
    feat = float(form.number_input('Max_Features',0.0,1.0,1.0,0.1))


    boot = bool(form.radio('Bootstrap',('True','False')))
    oob = bool(form.radio('OOB Score',('True','False'),1))
    form.form_submit_button()
    st.write(' Make sure to click Submit after every change ')
    st.write('   ')


fig,axs = plt.subplots()

X,y = make_graph(data,axs)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
origin = st.pyplot(fig)

if  algo == 'Logistic Regression':

    with st.sidebar:
        st.markdown('Hyperparam of Logistic')
        
        penalty = st.selectbox('Regularization',('l2','l1','elasticnet','none'))
        c = float(st.number_input('C',value=1.0))
        solver = st.selectbox(
        'Solver',
        ('lbfgs','newton-cg', 'liblinear', 'sag', 'saga'))
        max_iter = int(st.slider('Max_iter',1,500,step=100))

    clf = LogisticRegression(penalty=penalty,C=c,solver=solver,multi_class='multinomial',max_iter=max_iter)
    

elif algo == 'Naive Bayes':

    clf = GaussianNB()
    

elif algo == 'K-Nearest Neighbors':
    with st.sidebar:
        st.markdown('Hyperparam of KNN')

        n_neigh = st.slider('N_neighbour',1,10,step=5)
        weigh = st.radio('Weights',('uniform','distance'),0)
        algor  = st.selectbox('Algorithms',('auto','ball_tree','kd_tree','brute'))
    

    clf = KNeighborsClassifier(n_neighbors=n_neigh,weights=weigh,algorithm=algor)

elif algo == 'Decision Tree':
    
    with st.sidebar:

        st.markdown('Hyperparam of Decision Tree')

        max_d = st.number_input('Max_depth',10,500,step=100)
        crite = st.select_slider('Citerion',('gini','entropy','log_loss'))
        splite = st.radio('Splitter',('best','random'))
        max_feat = st.selectbox('Max_features_',(None,'sqrt','log2')) 
    

    clf = DecisionTreeClassifier(max_depth=max_d, criterion=crite,splitter=splite, max_features=max_feat)

elif algo == 'Support Vector Machines':
    with st.sidebar:
    
        st.markdown('Hyperparam of SVM')

        c_ = float(st.number_input('C_',value=1.0))
        kernel = st.selectbox('Kernel',
        ('rbf','sigmoid', 'linear', 'poly', 'precomputed'))       
        dec_fun_sha = st.radio('Decision_function_shape',('ovr','ovo'))
        max_iter = int(st.slider('max_iter',-1,500,-1))
   
    clf = SVC(C=c_, kernel=kernel, decision_function_shape=dec_fun_sha, max_iter=max_iter)
    

elif algo == 'Random Forest':
    with st.sidebar:
        
        st.markdown('Hyperparam of RNF')
        max_de = st.number_input('Max_depth_',10,500,step=100)
        crit = st.select_slider('Citerion_',('gini','entropy','log_loss'))
        max_fea = st.selectbox('max_features',(None,'sqrt','log2')) 
        n_est = st.number_input('N_estimators',1,500,step=10)
    
    clf = RandomForestClassifier(max_depth=max_de, criterion=crit, max_features=max_fea, n_estimators=n_est)
   

elif algo == 'SGDclassifier':
    with st.sidebar:
     
        st.markdown('Hyperparam of SGD')

        loss = st.selectbox('Loss',('hinge','log_loss','modified_huber','squared_hinge','perceptron','squared_error','huber','epsilon_insensitive',
        'squared_epsilon_insensitive'),)
        penalt = st.selectbox('Regularization_',('l2','l1','elasticnet'))
        shuffl= bool(st.radio('Shuffle_',('True','False')))
        lr = st.selectbox('Learning_r',('optimal','constant','invscaling','adaptive'))
        ear_sto= bool(st.radio('Early_stopping_',(True,False)))
        l1_ratio = float(st.number_input('l1_ratio_',0,1))
        n_iter_cha = int(st.number_input('N_iter_no_change',1,10,5))
        valid_frac = st.number_input('Validation_fration',0.0,1.0,0.1)
   
    clf = SGDClassifier(loss=loss,penalty=penalt,shuffle=shuffl,learning_rate=lr,early_stopping=ear_sto,l1_ratio=l1_ratio,n_iter_no_change=n_iter_cha,validation_fraction=valid_frac)
   
   

st.write('Press this button to run the algo')
if st.button('Run'):

    vot = BaggingClassifier(clf,n_estimators=estm,max_samples=samp,max_features=feat,bootstrap=boot)
    vot.fit(X_train,y_train)

    y_pred = vot.predict(X_test)
    
    XX,yy,input_array = make_meshgrid()
    zz = vot.predict(input_array)
    
    plt.contourf(XX,yy,zz.reshape(XX.shape),alpha = 0.7,cmap='rainbow')
    
    plt.xlabel('Col1')
    plt.ylabel('Col2')
    origin = st.pyplot(fig)
    st.subheader('Accuracy of Model :'+ str(round(accuracy_score(y_test,y_pred),2)))
    st.subheader('Precision Of Model :'+str(round(precision_score(y_test,y_pred,average='weighted'),2)))
    st.subheader('F1 Score  :'+ str(round(f1_score(y_test,y_pred),2)))
    
    st.snow()
    
from hashlib import algorithms_available
import streamlit as st 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import VotingClassifier,RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs,make_classification,make_gaussian_quantiles,make_circles,make_moons
from sklearn.metrics import accuracy_score,precision_score,r2_score


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

# def decision_boundary(clf):
#     xi = np.arange((X[:,0].min()-1), (X[:,0].max()+1) ,0.01)
#     yi = np.arange((X[:,1].min()-1), (X[:,1].max()+1) ,0.01)
#     xx,yy = np.meshgrid(xi,yi)
#     zz = np.array([xx.ravel(),yy.ravel()]).T

#     opt = clf.predict(zz)
#     plt.contourf(XX,yy,opt.reshape(XX.shape),alpha = 0.7,cmap='rainbow')



plt.style.use('seaborn-deep')

st.title('Voting classifier')

st.sidebar.markdown('Hyperparameters and Datasets')
data = st.sidebar.selectbox('Datasets',('Normal','Swords','Omlet','Moons','Concentric circle'))

algo = st.sidebar.multiselect('Choose Algoritms',('Logistic Regression','Naive Bayes','K-Nearest Neighbors','Decision Tree','Support Vector Machines','Random Forest',
    'SGDclassifier'))

vote = st.sidebar.radio('Voting type',('hard','soft'))

fig,axs = plt.subplots()

X,y = make_graph(data,axs)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
origin = st.pyplot(fig)

algo_list = []
for i in algo:
    if  i == 'Logistic Regression':
        st.sidebar.markdown('Hyperparam of Logistic')

        # penalty = st.sidebar.selectbox('Regularization',('l2','l1','elasticnet','none'))
        # c = float(st.sidebar.number_input('C',value=1.0))
        # solver = st.sidebar.selectbox(
        # 'Solver',
        # ('lbfgs','newton-cg', 'liblinear', 'sag', 'saga'))
        max_iter = int(st.sidebar.slider('Max_iter',1,500,100))
        # li_ratio = float(st.sidebar.number_input('l1_ratio(btw 0-1)',0.0,1.0))

        clf1 = LogisticRegression(multi_class='multinomial',max_iter=max_iter)
        algo_list.append(('Lo',clf1))
    
    elif i == 'Naive Bayes':

        clf2 = GaussianNB()
        algo_list.append(('NB',clf2))

    elif i == 'K-Nearest Neighbors':
        st.sidebar.markdown('Hyperparam of KNN')

        n_neigh = st.sidebar.slider('N_neighbour',1,10,5)
        weigh = st.sidebar.radio('Weights',('uniform','distance'),0)
        algor  = st.sidebar.selectbox('Algorithms',('auto','ball_tree','kd_tree','brute'))

        clf3 = KNeighborsClassifier(n_neighbors=n_neigh,weights=weigh,algorithm=algor)
        algo_list.append(('KNN',clf3))

    elif i == 'Decision Tree':
        # st.sidebar.markdown('Hyperparam of DT')

        # max_d = st.sidebar.number_input('Max_depth',10,500,100)
        # crite = st.sidebar.select_slider('Citerion',('gini','entropy','log_loss'))
        # splite = st.sidebar.radio('Splitter',('best','random'))
        # max_feat = st.sidebar.selectbox('Max_features_',(None,'sqrt','log2')) 


        clf4 = DecisionTreeClassifier()
        algo_list.append(('DC',clf4))
    
    elif i == 'Support Vector Machines':
        # st.sidebar.markdown('Hyperparam of SVM')

        # c_ = float(st.sidebar.number_input('C_',value=1.0))
        # kernel = st.sidebar.selectbox(
        # 'Kernel',
        # ('rbf','sigmoid', 'linear', 'poly', 'precomputed'))       
        # dec_fun_sha = st.sidebar.radio('Decision_function_shape',('ovr','ovo'))
        # max_iter = int(st.sidebar.slider('max_iter',-1,500,-1))

        if vote == 'soft':
            clf5 = SVC(probability=True)
        else:
            clf5 = SVC()

        algo_list.append(('SVM',clf5))
    
    elif i == 'Random Forest':
        # st.sidebar.markdown('Hyperparam of RNF')

        # max_de = st.sidebar.number_input('Max_depth_',10,500,100)
        # crit = st.sidebar.select_slider('Citerion_',('gini','entropy','log_loss'))
        # max_fea = st.sidebar.selectbox('max_features',(None,'sqrt','log2')) 
        # n_est = st.sidebar.number_input('N_estimators',1,500,10)

        clf6 = RandomForestClassifier()
        algo_list.append(('RFC',clf6))

    elif i == 'SGDclassifier':
        # st.sidebar.markdown('Hyperparam of SGD')
        # loss = st.sidebar.selectbox('Loss',('hinge','log_loss','modified_huber','squared_hinge','perceptron','squared_error','huber','epsilon_insensitive',
        # 'squared_epsilon_insensitive'),)
        # penalt = st.sidebar.selectbox('Regularization_',('l2','l1','elasticnet'))
        # shuffl= bool(st.sidebar.radio('Shuffle_',('True','False')))
        # lr = st.sidebar.selectbox('Learning_r',('optimal','constant','invscaling','adaptive'))
        # ear_sto= bool(st.sidebar.radio('Early_stopping_',(True,False)))
        # l1_ratio = float(st.sidebar.number_input('l1_ratio_',0,1))
        # n_iter_cha = int(st.sidebar.number_input('N_iter_no_change',1,10,5))
        # valid_frac = st.sidebar.number_input('Validation_fration',0.0,1.0,0.1)

        # clf7 = SGDClassifier(loss=loss,penalty=penalt,shuffle=shuffl,learning_rate=lr,early_stopping=ear_sto,l1_ratio=l1_ratio,n_iter_no_change=n_iter_cha,validation_fraction=valid_frac)
        clf7 = SGDClassifier()
        algo_list.append(('SGD',clf7))


if st.sidebar.button('Run'):

    vot = VotingClassifier(algo_list,voting=vote)
    vot.fit(X_train,y_train)

    y_pred = vot.predict(X_test)
    
    XX,yy,input_array = make_meshgrid()
    zz = vot.predict(input_array)
    
    plt.contourf(XX,yy,zz.reshape(XX.shape),alpha = 0.7,cmap='rainbow')
    
    plt.xlabel('Col1')
    plt.ylabel('Col2')
    origin = st.pyplot(fig)
    st.subheader('Accuracy of Logistic :'+ str(round(accuracy_score(y_test,y_pred),2)))
    st.subheader('Precision Of Logistic :'+str(round(precision_score(y_test,y_pred,average='weighted'),2)))
    st.subheader('R2 Score  :'+ str(round(r2_score(y_test,y_pred),2)))
    st.balloons()
    # st.snow()
    st.sidebar.balloons()
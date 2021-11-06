import streamlit as st
import pickle
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import sklearn
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import random
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import sklearn.model_selection as model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import RandomOverSampler


#应用标题
st.title('A machine learning-based predictive model for predicting bone metastasis in patients with renal cancer')

# conf
st.sidebar.markdown('## Variables')

#Diameter_G = st.sidebar.selectbox('Diameter.G',('<5cm','5-10cm','>10cm'),index=0)

Grade = st.sidebar.selectbox("Grade",('Well differentiated','Moderately differentiated','Poorly differentiated',
                                      'Undifferentiated; anaplastic','unknown'),index=0)
T = st.sidebar.selectbox("T stage",('T1','T2','T3','T4','TX'))
N = st.sidebar.selectbox("N stage",('N0','N1','N2','NX'))
Liver_metastasis = st.sidebar.selectbox("Liver metastases",('No','Yes'))
Brain_metastases = st.sidebar.selectbox("Brain metastases",('No','Yes'))
Pulmonary_metastasis = st.sidebar.selectbox("Lung metastases",('No','Yes'),index=0)
#Tumor_Size = st.sidebar.slider("Tumor size", 1, 999, value=450, step=1)
#steatosis = st.sidebar.selectbox("Steatosis",('No','Yes'),index=0)

#Lung_metastases = st.sidebar.selectbox("Lung metastases",('No','Yes'))
#st.sidebar.markdown('#  ')
# str_to_int

map = {'T1':0,'T2':1,'T3':2,'T4':3,'TX':4,'No':0,'Yes':1,'Well differentiated':0,'Moderately differentiated':1,
       'Poorly differentiated':2,'Undifferentiated; anaplastic':3,'unknown':4,'N0':0,'N1':1,'N2':2,'NX':3}
#map = {'White':0,'Black':1,'Other':2,'T1':0,'T2':1,'T3':2,'TX':3,'M0':0,'M1':1,'NX':2,'No':0,'Yes':1,}
#Age =map[Age]

Grade =map[Grade]
T =map[T]
N =map[N]
Liver_metastasis =map[Liver_metastasis]
Brain_metastases =map[Brain_metastases]
Pulmonary_metastasis =map[Pulmonary_metastasis]
#Bone_metastases =map[Bone_metastases]
#Lung_metastases =map[Lung_metastases]

# 数据读取，特征标注
thyroid_train = pd.read_csv('train.csv', low_memory=False)
features = ['Grade','T','N','Liver.metastasis','Pulmonary.metastasis','Brain.metastases']
target = 'Bone.metastases'
#处理数据不平衡
ros = RandomOverSampler(random_state=12, sampling_strategy='auto')
X_ros, y_ros = ros.fit_resample(thyroid_train[features], thyroid_train[target])

XGB = XGBClassifier(random_state=32,max_depth=3,n_estimators=34)
XGB.fit(X_ros, y_ros)
#RF = sklearn.ensemble.RandomForestClassifier(n_estimators=4,criterion='entropy',max_features='log2',max_depth=3,random_state=12)
#RF.fit(X_ros, y_ros)


sp = 0.5
#figure
is_t = (XGB.predict_proba(np.array([[Grade,T,N,Liver_metastasis,Pulmonary_metastasis,Brain_metastases]]))[0][1])> sp
prob = (XGB.predict_proba(np.array([[Grade,T,N,Liver_metastasis,Pulmonary_metastasis,Brain_metastases]]))[0][1])*1000//1/10

#st.write('is_t:',is_t,'prob is ',prob)
#st.markdown('## is_t:'+' '+str(is_t)+' prob is:'+' '+str(prob))

if is_t:
    result = 'High Risk'
else:
    result = 'Low Risk'
if st.button('Predict'):
    st.markdown('## Risk grouping for BM:  '+str(result))
    if result == 'Low Risk':
        st.balloons()
    st.markdown('## Probability of BM:  '+str(prob)+'%')
#st.markdown('## The risk of bone metastases is '+str(prob/0.0078*1000//1/1000)+' times higher than the average risk .')

#排版占行



st.title("")
st.title("")
st.title("")
st.title("")
#st.warning('This is a warning')
#st.error('This is an error')

#st.info('Information of the model: Auc: 0.874 ;Accuracy: 0.851 ;Sensitivity(recall): 0.750 ;Specificity :0.868 ')
#st.success('Affiliation: The First Affiliated Hospital of Nanchang University, Nanchnag university. ')






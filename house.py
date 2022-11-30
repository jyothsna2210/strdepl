import streamlit as st
import pandas as pd
import shap
import numpy as np
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
st.title(""" House price Prediction App This app predicts the House Price **""")

html_temp = """
<div style='background-color:yellow;padding:10px;'>
<h3> Streamlit ML App</h3>
</div>
"""

st.markdown(html_temp,unsafe_allow_html=True)
st.write('---')
video_file = open("istockphoto-1245236548-640_adpp_is.mp4", 'rb')


st.video(video_file)
df=datasets.load_boston()
df1=pd.DataFrame(df.data,columns=df.feature_names)
st.write(df1)
if st.checkbox("show all dtaa"):
    if st.button("Head"):
        st.write(df1.head())
    elif st.button("Tail"):
        st.write(df1.tail())
    else:
        st.write(df1.head(2))
dim=st.radio('What Dimenstions Do You Wnt to see?',('Rows','Columns','all'))
if dim=='Rows':
    st.text('Showing rows')
    st.write(df1.shape[0])
if dim=='Columns':
    st.text('Showing Columns')
    st.write(df1.shape[1])
else:
    st.text('Showing Shape of dataset')
    st.write(df1.shape)
st.write('This is a area_chart.')
st.area_chart(df1)
st.bar_chart(df1['AGE'], width=0, height=0, use_container_width=True)
fig=plt.figure()
st.write(sns.heatmap(df1.corr()))
st.pyplot()
st.balloons()

X=pd.DataFrame(df.data,columns=df.feature_names)
Y=pd.DataFrame(df.target,columns=['MEDV'])
st.sidebar.header('Specify Input Parameters')
def user_input_features():
    CRIM=st.sidebar.slider('CRIM',float(X.CRIM.min()),float(X.CRIM.max()),float(X.CRIM.mean()))
    ZN=st.sidebar.slider('ZN',float(X.ZN.min()),float(X.ZN.max()),float(X.ZN.mean()))
    INDUS=st.sidebar.slider('INDUS',float(X.INDUS.min()),float(X.INDUS.max()),float(X.INDUS.mean()))
    CHAS=st.sidebar.slider('CHAS',float(X.CHAS.min()),float(X.CHAS.max()),float(X.CHAS.mean()))
    NOX=st.sidebar.slider('NOX',float(X.NOX.min()),float(X.NOX.max()),float(X.NOX.mean()))
    RM=st.sidebar.slider('RMX',float(X.RM.min()),float(X.RM.max()),float(X.RM.mean()))
    AGE=st.sidebar.slider('AGE',float(X.AGE.min()),float(X.AGE.max()),float(X.AGE.mean()))
    DIS=st.sidebar.slider('DIS',float(X.DIS.min()),float(X.DIS.max()),float(X.DIS.mean()))
    RAD=st.sidebar.slider('RAD',float(X.RAD.min()),float(X.RAD.max()),float(X.RAD.mean()))
    TAX=st.sidebar.slider('TAX',float(X.TAX.min()),float(X.TAX.max()),float(X.TAX.mean()))
    PTRATIO=st.sidebar.slider('PTRATIO',float(X.PTRATIO.min()),float(X.PTRATIO.max()),float(X.PTRATIO.mean()))
    B=st.sidebar.slider('B',float(X.B.min()),float(X.B.max()),float(X.B.mean()))
    LSTAT=st.sidebar.slider('LSTAT',float(X.LSTAT.min()),float(X.LSTAT.max()),float(X.LSTAT.mean()))
    data={
        'CRIM':CRIM,
        'ZN':ZN,
        'INDUS':INDUS,
        'CHAS':CHAS,
        'NOX':NOX,
        'RM':RM,
        'AGE':AGE,
        'DIS':DIS,
        'RAD':RAD,
        'TAX':TAX,
        'PTRATIO':PTRATIO,
        'B':B,
        'LSTAT':LSTAT}
    features=pd.DataFrame(data,index=[0])
    return features
    
df=user_input_features()
st.header('Specified Input Parameters ')
st.write(df) 
st.write("---")
model=RandomForestRegressor()
model.fit(X,Y)
prediction=model.predict(df)
st.header('Prediction OF MEDV')
st.write(prediction)
st.write('---')
image = Image.open("price.jpeg")
st.image(image, caption='House price prediction')
explainer=shap.TreeExplainer(model)
shap_values=explainer.shap_values(X)
st.header('Feature Importance')  
plt.title('Feature importance based on SHAP values' )  
shap.summary_plot(shap_values,X)
st.write('---')
plt.title('Feature importance base on SHAP values (Bar)')
shap.summary_plot(shap_values,X,plot_type='Bar')
st.pyplot(bbox_inches='tight')










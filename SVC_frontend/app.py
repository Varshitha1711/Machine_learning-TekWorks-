import streamlit as st
import pandas as pd 
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from datetime import datetime

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing  import StandardScaler,LabelEncoder
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


def log(message):
    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

if "cleaned_saved" not in st.session_state:
    st.session_state.cleaned_saved=False

BASE_DIR=os.path.dirname(os.path.abspath(__file__))
RAW_DIR=os.path.join(BASE_DIR,"data","raw")
CLEANED_DIR=os.path.join(BASE_DIR,"data","cleaned")

os.makedirs(RAW_DIR,exist_ok=True)
os.makedirs(CLEANED_DIR,exist_ok=True)

log("Application started")
log(f"RAW_DIR {RAW_DIR}")
log(f"CLEANED_DIR {CLEANED_DIR}")

st.set_page_config("End-to-End SVM platform",layout="wide")
st.title("End-to-End SVM platform")

st.sidebar.header("SVM Settings")

kernel=st.sidebar.selectbox("kernel",["linear","poly","sigmoid","rbf"])
C=st.sidebar.slider("C (Regularartion factor)",0.01,10.0,1.0)
gamma=st.sidebar.selectbox("Gamma",["scale","auto"])

log(f"SVM Settings---> kernel={kernel},c={C},gamma={gamma}")

# Step 1: Data Ingestion

st.header("Step 1 :Data Ingestion")
log("Step 1 : Data ingestion Started ")

option =st.radio("Choose Data Source",["Download Dataset","Upload CSV"])
df=None
raw_path=None

if option =="Download Dataset":
    if st.button("Download Iris Dataset"):
        log("Downloading Iris Dataset...")
        url="https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
        response=requests.get(url)

        raw_path=os.path.join(RAW_DIR,"iris.csv")
        with open (raw_path,"wb") as f:
            f.write(response.content)

        df=pd.read_csv(raw_path)
        st.success("Dataset Downloaded successfully")
        log(f"Iris dataset saved at {raw_path}")
if option=="Upload CSV":
    upload_file=st.file_uploader("Upload CSV file",type=["csv"])
    if upload_file:
        raw_path=os.path.join(RAW_DIR,upload_file.name)
        with open (raw_path,"wb") as f:
            f.write(upload_file.getbuffer())
        df=pd.read_csv(raw_path)
        st.success("File uploaded successfuly")
        log(f"Uploaded Data  saved at {raw_path}")


# Step 2 :EDA

if df is not None:
    st.header("Step 2 :Exploratory Data Analysis")
    log("Step 2 Started :EDA ")
    st.dataframe(df.head())
    st.write ("Shape ",df.shape)
    st.write("Missing Values : ",df.isnull().sum())

    fig,ax=plt.subplots()
    sns.heatmap(df.corr(numeric_only=True),cmap="coolwarm",annot=True,ax=ax)
    st.pyplot(fig)

    log("EDA Completed")

#Step 3 : Data Cleaning 
if df is not None:
    st.header("Step 3 : Data Cleaning")
    strategy=st.selectbox(
        "Missing value Startegy",
        ["Mean ","Median ","Drop rows"],
    )
    df_clean=df.copy()
    if strategy=="Drop Rows":
        df_clean=df_clean.dropna()
    else:
        for col in df_clean.select_dtypes(include=np.number):
            if strategy=="Mean":
                df_clean[col]=df_clean[col].fillna(df_clean[col].mean())
            else:
                df_clean[col]=df_clean[col].fillna(df_clean[col].median())

    st.session_state.df_clean=df_clean
    st.success("Data Cleaning completed.")

else:
    st.info("Please complete Step 1(Data ingestion)...")


# Step 4 :Save Cleaned Data 

if st.button("Save cleaned Dataset :"):
    if st.session_state.df_clean is None:
        st.error("No cleaned dat found .Please complete step 3 ")
    else:
        timestamp=datetime.now().strftime("%Y%m%d %H%M%S")
        cleaned_filename=f"cleaned_dataset{timestamp}.csv"
        clean_path=os.path.join(CLEANED_DIR,cleaned_filename)
        st.session_state.df_clean.to_csv(clean_path,index=False)
        st.info(f"Saved st :{clean_path}")
        log(f"Cleaned dataset saved at {clean_path}")



# Step 5 :Load Cleaned Dataset

st.header("Step 5 :Loadd Cleaned Dataset")
clean_files=os.listdir(CLEANED_DIR)
if not clean_files:
    st.warning("No cleaned datasets found .Please save one  in step 4")
    log("No cleaned dataset available")

else:
    selected=st.selectbox("Select cleaned Dataset" ,clean_files)
    df_model=pd.read_csv(os.path.join(CLEANED_DIR,selected))
    st.success(f"Loaded Dataset :{selected}")
    log(f"Loaded cleaned dataset {selected}")

    st.dataframe(df_model.head())


#Step 6 : Train SVM
st.header("Step 6 : Train SVM")
log("Step 6 started : SVM Training")
target =st.selectbox("Select Target column",df_model.columns)
y=df_model[target]
if y.dtype=="object":
    y=LabelEncoder().fit_transform(y)
    log("Target column encoded")

#select numerical features only 

x=df_model.drop(columns=[target])
x=x.select_dtypes(include=np.number)

if x.empty:
    st.error("No numeric features available for training")
    st.stop()

#Sacle features

scaler =StandardScaler()
x=scaler.fit_transform(x)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)


model=SVC(kernel=kernel,C=C,gamma=gamma)
model.fit(x_train,y_train)

#Evaluate 

y_pred=model.predict(x_test)
acc=accuracy_score(y_test,y_pred)

st.success(f"Accuracy : {acc:.2f}")
log(f"SVM trained successfully  | Accuracy ={acc:.2f}")

cm=confusion_matrix(y_test,y_pred)
fig,ax=plt.subplots()
sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",ax=ax)
st.pyplot(fig)
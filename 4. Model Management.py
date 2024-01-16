# Databricks notebook source
#importing the data:
data = spark.table('credit_risk_cleaned')
data.display()

# COMMAND ----------

#creating a class with functions to preprocess,train and predict the data:
import mlflow

class RFwithPreprocess(mlflow.pyfunc.PythonModel):
    def __init__(self,params):
        """
        Initialize with the hyperparameter
        """
        self.params = params
        self.rf_model = None
        self.config = None

    def load_context(self,context=None,config_path=None):
        if context:
            config_path = context.artifacts['config_path']
        else:
            pass

        self.config = json.load(open(config_path))

    def preprocess_input(self,data):
        from sklearn.preprocessing import LabelEncoder
        df = data.copy()
        cat_cols = df.select_dtypes(exclude='number')
        le = LabelEncoder()
        for i in cat_cols:
            df[i] = le.fit_transform(df[i])
        return df

    def fit(self,X_train,y_train):
        from sklearn.ensemble import RandomForestClassifier
        preprocessed_input = self.preprocess_input(X_train)
        rf = RandomForestClassifier(**self.params)
        rf.fit(preprocessed_input,y_train)
        self.rf_model = rf

    def predict(self,context,model_input):
        preprocessed_input_test = self.preprocess_input(model_input.copy())
        return self.rf_model.predict(preprocessed_input_test)

# COMMAND ----------

import json
import os

params = {
    "n_estimators": 250, 
    "max_depth": 15
}

#designate a path:
config_path = f"/dbfs/dbfs/user/hive/credit_risk_cleaned/data.json"

#save the results:
with open(config_path,"w") as f:
    json.dump(params,f)

artifacts = {"config_path":config_path}

# COMMAND ----------

#using the created class:
from sklearn.model_selection import train_test_split
df = data.toPandas()
X = df.drop('loan_status',axis=1)
y = df['loan_status']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

model = RFwithPreprocess(params)
model.load_context(config_path=config_path)
model.config

# COMMAND ----------

model.fit(X_train,y_train)

# COMMAND ----------

#generating the predictions:
preds = model.predict(context=None,model_input=X_test)
preds

# COMMAND ----------

#Generate the model signature.
from mlflow.models.signature import infer_signature

signature = infer_signature(X_test,preds)
signature

# COMMAND ----------

#Generate the conda environment.
from sys import version_info
import sklearn

conda_env = {
    "channels" : ["defaults"],
    "dependencies" : [
        f"python={version_info.major}.{version_info.minor}.{version_info.micro}",
        "pip",
        {"pip" : ["mlflow",
                  f"scikit-learn=={sklearn.__version__}"]
         }
    ],
    "name" : "sklearn_env"
}

conda_env

# COMMAND ----------

#Save the model.
with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(
        "rf_preprocessed_model",
        python_model=model,
        artifacts=artifacts,
        conda_env=conda_env,
        signature=signature,
        input_example=X_test[:3]
    )

# COMMAND ----------

#Load the model in python_function format.
mlflow_pyfunc_model_path = f"runs:/{run.info.run_id}/rf_preprocessed_model"
loaded_model = mlflow.pyfunc.load_model(mlflow_pyfunc_model_path)

# COMMAND ----------

#Apply the model.
loaded_model.predict(X_test)

# COMMAND ----------



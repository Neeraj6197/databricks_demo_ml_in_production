# Databricks notebook source
#importing the dataset:
data = spark.table("credit_risk_transformed_data")
data.display()

# COMMAND ----------

#importing the libraries:
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# COMMAND ----------

#splitting the data:
pd_df = data.toPandas()
X = pd_df.drop('loan_status',axis=1)
y = pd_df["loan_status"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# COMMAND ----------

#building the model:
max_iter = 500
lr = LogisticRegression(max_iter=max_iter)
lr.fit(X_train,y_train)
input_example = X_train.head()
signature = infer_signature(X_train,pd.DataFrame(y_train))

with mlflow.start_run(run_name='Webhooks_demo') as run:
    mlflow.sklearn.log_model(lr,"model",signature = signature, input_example = input_example)
    mlflow.log_metric("acc_score",accuracy_score(y_test,lr.predict(X_test)))
    mlflow.log_param('max_iter',max_iter)
    run_id = run.info.run_id
    exp_id = run.info.experiment_id

# COMMAND ----------

#registering the model:
name = 'Webhook_demo_model'
model_uri = f"runs:/{run_id}/model"
model_details = mlflow.register_model(model_uri=model_uri,name=name)

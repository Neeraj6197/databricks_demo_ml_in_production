# Databricks notebook source
#laoding the transformed data:
data = spark.table('credit_risk_transformed_data')
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

with mlflow.start_run(run_name='LR_CR_Model') as run:
    mlflow.sklearn.log_model(lr,"model",signature = signature, input_example = input_example)
    mlflow.log_metric("acc_score",accuracy_score(y_test,lr.predict(X_test)))
    mlflow.log_param('max_iter',max_iter)
    run_id = run.info.run_id

# COMMAND ----------

# DBTITLE 1,Registering a Model
model_name = 'CreditRisk-LR-Model'
model_uri = f"runs:/{run_id}/model"

# COMMAND ----------

model_details = mlflow.register_model(model_uri = model_uri, name = model_name)
model_details

# COMMAND ----------

#importing mlflow client:
from mlflow.tracking.client import MlflowClient
client = MlflowClient()

# COMMAND ----------

#Check the status.
model_version_details = client.get_model_version(name=model_name,version=1)
model_version_details.status

# COMMAND ----------

#Now add a model description
client.update_registered_model(
    name=model_details.name,
    description="This LR Model predicts the credit risk of a customer",
)

# COMMAND ----------

#Add a version-specific description.
client.update_model_version(
    name=model_details.name,
    description="This model version was built using sklearn.",
    version=model_details.version
)

# COMMAND ----------

# DBTITLE 1,Deploying the model
#checking the model stage:
model_version_details.current_stage

# COMMAND ----------

#pushing the model to production stage:
client.transition_model_version_stage(
    name=model_details.name,
    version=model_details.version,
    stage='Production'
)

# COMMAND ----------

#re-checking the model's current stage:
model_version_details = client.get_model_version(
    name=model_details.name,
    version=model_details.version
)
model_version_details.current_stage

# COMMAND ----------

#Fetch the latest model using a pyfunc
import mlflow.pyfunc
model_version_uri = f"models:/{model_name}/1"
model_version_1 = mlflow.pyfunc.load_model(model_version_uri)

# COMMAND ----------

#applying the model:
model_version_1.predict(X_test)

# COMMAND ----------

# DBTITLE 1,Deploying a New Model Version
#building the model:
max_iter = 1000
lr = LogisticRegression(max_iter=max_iter)
lr.fit(X_train,y_train)
input_example = X_train.head()
signature = infer_signature(X_train,pd.DataFrame(y_train))

with mlflow.start_run(run_name='LR_CR_Model') as run:
    mlflow.sklearn.log_model(sk_model = lr,
                             artifact_path="sklearn-model",
                             registered_model_name=model_name,
                             signature = signature, 
                             input_example = input_example)
    mlflow.log_metric("acc_score",accuracy_score(y_test,lr.predict(X_test)))
    mlflow.log_param('max_iter',max_iter)
    run_id = run.info.run_id

# COMMAND ----------

#Use the search functionality to grab the latest model version.
model_versions_infos = client.search_model_versions(f"name = '{model_name}'")
new_model_version = max([i.version for i in model_versions_infos])
print(new_model_version)


# COMMAND ----------

#Add a description to this new version.
client.update_model_version(
    name=model_name,
    version=new_model_version,
    description="New version of createad model"
)

# COMMAND ----------

#Put this new model version into Staging
client.transition_model_version_stage(
    name=model_name,
    stage='Staging',
    version=new_model_version
)

# COMMAND ----------

#push that model into production.
client.transition_model_version_stage(
    name=model_name,
    version=new_model_version,
    stage='Production',
    archive_existing_versions=True
)

# COMMAND ----------

#Deleting the existing models, You cannot delete a model that is not first archived.
client.delete_model_version(
    name=model_name,
    version=3
)

# COMMAND ----------

#Archive version 4 of the model too.
client.transition_model_version_stage(
    name=model_name,
    version=new_model_version,
    stage='Archived'
)

# COMMAND ----------

#Now delete the entire registered model.
client.delete_registered_model(model_name)

# COMMAND ----------



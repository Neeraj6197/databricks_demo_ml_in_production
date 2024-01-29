# Databricks notebook source
#laoding the transformed data:
data = spark.table('credit_risk_transformed_data')
data.display()

# COMMAND ----------

pd_df = data.toPandas()
X = pd_df.drop('loan_status',axis=1)
y = pd_df["loan_status"]

# COMMAND ----------

import pandas as pd
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

with mlflow.start_run(run_name='Final_RF_Model') as run:
    rf = RandomForestClassifier()
    rf.fit(X,y)
    preds = rf.predict(X)
    mlflow.sklearn.log_model(rf,"Random_Forest_Model")
    score = accuracy_score(y,preds)
    mlflow.log_metric('score',score)

# COMMAND ----------

#creating a spark dataframe:
spark_df = spark.createDataFrame(X)
spark_df.display()

# COMMAND ----------

#loading the model:
model_uri = f"runs:/{run.info.run_id}/Random_Forest_Model"
model = mlflow.pyfunc.spark_udf(spark,model_uri)

# COMMAND ----------

#Apply the model as a standard UDF using the column names as the input to the function.
prediction_df = spark_df.withColumn('prediction',model(*spark_df.columns))
prediction_df.display()

# COMMAND ----------

# DBTITLE 1,Feature Store Batch Scoring:
#create a feature store client:
from databricks.feature_store import FeatureStoreClient
fs = FeatureStoreClient()

# COMMAND ----------

#creating a primary key
from pyspark.sql.functions import monotonically_increasing_id
df = spark.table('credit_risk_transformed_data').withColumn('customer_id',monotonically_increasing_id())
features_df = df.drop('loan_status')
features_df.display()

# COMMAND ----------

#inference data:
inference_data = df.select('customer_id','loan_status')
inference_data.display()

# COMMAND ----------

#creating a feature table:
table_name = 'credit_risk_feature_table'
table_name

# COMMAND ----------

results = fs.create_table(
    name=table_name,
    primary_keys=['customer_id'],
    df=features_df,
    schema=features_df.schema
    )

# COMMAND ----------

from databricks.feature_store import feature_table,FeatureLookup
feature_lookups = [FeatureLookup(table_name=table_name,feature_names=None,lookup_key='customer_id')]
feature_lookups

# COMMAND ----------

#creating the training data using feature lookups:
training_set = fs.create_training_set(inference_data,feature_lookups,label='loan_status',exclude_columns='customer_id')
training_set

# COMMAND ----------

# DBTITLE 1,Log a feature store packaged model.
model_name = table_name + "_model"
print(model_name)

# COMMAND ----------

import mlflow
from mlflow.models.signature import infer_signature
fs.log_model(
    model=rf,
    artifact_path="feature_store_model",
    flavor=mlflow.sklearn,
    training_set=training_set,
    registered_model_name=model_name,
    input_example=X.head(5),
    signature=infer_signature(X,y)
)

# COMMAND ----------

#Let's now perform batch scoring with the feature store model.
batch_input_df = inference_data.drop('loan_status')
model = f"models:/{model_name}/1"
with_predictions = fs.score_batch(model,batch_input_df)
with_predictions.display()

# COMMAND ----------



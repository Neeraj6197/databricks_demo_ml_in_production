# Databricks notebook source
!pip install --upgrade mlflow

# COMMAND ----------

#importing the cleaned data:
df = spark.table('credit_risk_cleaned')
df.display()

# COMMAND ----------

# DBTITLE 1,Deploying Spark Model
# Import Required Libraries
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import mlflow

# Transforming the Data
numcols = [i for (i, j) in df.dtypes if j != 'string' and i != 'loan_status']
catcols = [i for (i, j) in df.dtypes if j == 'string']
print('numcols:', numcols)
print('catcols:', catcols)

si_cols = [x+"_si" for x in catcols]
si = StringIndexer(inputCols=catcols, outputCols=si_cols, handleInvalid='skip')
inp_cols = numcols + si_cols
vc = VectorAssembler(inputCols=inp_cols, outputCol='features', handleInvalid='skip')

# Creating and Logging the Model
# Splitting the Data
train, val, test = df.randomSplit([0.6, 0.2, 0.2])

lr = LogisticRegression(featuresCol='features', labelCol='loan_status', maxIter=300)
stages = [si, vc, lr]
pipeline = Pipeline(stages=stages)

# specify mlflow==1.*
conda_env = {
    "channels": ["conda-forge"],
    "dependencies": [
        "python=3.9.5",
        "pip<=21.2.4",
        {"pip": ["mlflow==1.*", "cloudpickle==2.0.0","pyspark==3.3.0"]},
    ],
    "name": "mlflow-env",
}

with mlflow.start_run(run_name='LR_CR_Model') as run:
    mlflow.autolog()
    pipe_model = pipeline.fit(train)
    preds = pipe_model.transform(val)
    evaluator = MulticlassClassificationEvaluator(labelCol='loan_status', metricName='f1')
    score = evaluator.evaluate(preds)
    mlflow.spark.log_model(pipe_model,
                           'spark_model',
                           input_example=train.limit(1).toPandas(),
                           conda_env=conda_env)
    mlflow.log_metric('f1', score)


# COMMAND ----------

#register the model:
model_name = 'CreditRisk-LR-Model'
model_uri = f"runs:/{run.info.run_id}/spark_model"

model_details = mlflow.register_model(model_uri = model_uri, name = model_name)
model_details

# COMMAND ----------

#load back the model:
logged_model = f'runs:/{run.info.run_id}/spark_model'

# Load model
loaded_model = mlflow.spark.load_model(logged_model)

# COMMAND ----------

import requests
import time

def score_model(dataset, timeout_sec=300):
    start = int(time.time())
    print(f"Scoring {model_name}")

    url = f"{api_url}/model/{model_name}/1/invocations"
    
    # Assuming 'dataset' is a Spark DataFrame
    ds_records = dataset.toJSON().collect()
    ds_dict = {'dataframe_records': ds_records}
    
    while True:
        response = requests.request(method="POST", headers=headers, url=url, json=ds_dict)
        elapsed = int(time.time()) - start

        if response.status_code == 200:
            return response.json()
        elif elapsed > timeout_sec:
            raise Exception(f"Endpoint was not ready after {timeout_sec} seconds")
        elif response.status_code == 503:
            print("Temporarily unavailable, retrying in 5 seconds")
            time.sleep(5)
        else:
            raise Exception(f"Request failed with status {response.status_code}, {response.text}")



# COMMAND ----------


score_model(test)

# COMMAND ----------

test.limit(1).toJSON().collect()[0]

# COMMAND ----------



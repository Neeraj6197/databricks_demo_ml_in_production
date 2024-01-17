# Databricks notebook source
#importing the cleaned data:
df = spark.table('credit_risk_cleaned')
df.display()

# COMMAND ----------

#splitting the data to train and test:
from sklearn.model_selection import train_test_split

pd_df = df.toPandas()
pd_df

# COMMAND ----------

# DBTITLE 1,Creating & Logging the Model:
#importing the libraries:
import mlflow.sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# COMMAND ----------

#transforming the dataset:
le = LabelEncoder()
cat_cols = pd_df.select_dtypes(exclude='number')
tr_df = pd_df.copy()
for i in cat_cols:
    tr_df[i] = le.fit_transform(tr_df[i].values.reshape(-1,1))

tr_df

# COMMAND ----------

#saving the transformed data:
spark.createDataFrame(tr_df).write.mode("overwrite").saveAsTable('credit_risk_transformed_data')

# COMMAND ----------

#splitting the dataset:
X = tr_df.drop('loan_status',axis=1)
y = tr_df['loan_status']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
X_train.head()

# COMMAND ----------

#Now, let's train our model and log it with MLflow. This time, we will add a signature and input_examples when we log our model.
import mlflow
from mlflow.models.signature import infer_signature
import pandas as pd

with mlflow.start_run(run_name='Signature_Example') as run:
    rf = RandomForestClassifier()
    rf.fit(X_train,y_train)
    preds = rf.predict(X_test)
    score = accuracy_score(y_test,preds)
    mlflow.log_metric("score",score)

    #Log the model with signature and example:
    signature = infer_signature(X_train,pd.DataFrame(y_train))
    input_example = X_train.head()
    mlflow.sklearn.log_model(rf,'RF_Model',signature=signature,input_example=input_example)


# COMMAND ----------

# DBTITLE 1,Nested Runs
#Logging 2 child models with different params:
from sklearn.metrics import log_loss

with mlflow.start_run(run_name="Nested_Run_Example") as run:
    #creating nested runs:
    #model 1:
    with mlflow.start_run(run_name='RF_model_1',nested=True):
        rf = RandomForestClassifier(n_estimators=150,max_depth=10)
        rf.fit(X_train,y_train)
        preds = rf.predict(X_test)
        score = accuracy_score(y_test,preds)
        loss = log_loss(y_test,preds)
        mlflow.log_metric("score_model_1",score)
        mlflow.log_metric("Log_loss",loss)

    #model 2:
    with mlflow.start_run(run_name='RF_model_2',nested=True):
        rf2 = RandomForestClassifier(n_estimators=250,max_depth=15)
        rf2.fit(X_train,y_train)
        preds2 = rf2.predict(X_test)
        score = accuracy_score(y_test,preds2)
        mlflow.log_metric("score_model_2",score)
        loss = log_loss(y_test,preds2)
        mlflow.log_metric("Log_loss",loss)


# COMMAND ----------

# DBTITLE 1, Auto Logging
mlflow.autolog()
rf = RandomForestClassifier(n_estimators=150,max_depth=10)
rf.fit(X_train,y_train)

# COMMAND ----------

# DBTITLE 1,Hyper parameter tuning
#defining the objective function:

def objective_function(best_params):
    n_estimators = int(best_params['n_estimators'])
    max_depth = int(best_params['max_depth'])
    min_samples_leaf = int(best_params['min_samples_leaf'])
    min_samples_split=int(best_params["min_samples_split"])

    #creating the model:
    rf = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,
                                min_samples_leaf=min_samples_leaf,min_samples_split=min_samples_split)
    rf.fit(X_train,y_train)
    preds = rf.predict(X_test)
    loss = log_loss(y_test,preds)
    return loss

# COMMAND ----------

#defining the search space:
from hyperopt import hp,fmin,tpe,SparkTrials

search_space = {"n_estimators": hp.quniform("n_estimators", 100, 500, 5),
                "max_depth": hp.quniform("max_depth", 5, 20, 1),
                "min_samples_leaf": hp.quniform("min_samples_leaf", 1, 5, 1),
                "min_samples_split": hp.quniform("min_samples_split", 2, 6, 1)}


# COMMAND ----------

## Set parallelism (should be order of magnitude smaller than max_evals)
spark_trials = SparkTrials(parallelism=2)

with mlflow.start_run(run_name="Hyperopt_demo"):
    mlflow.autolog()
    best_params = fmin(fn=objective_function,
                       space=search_space,
                       algo=tpe.suggest,
                       max_evals=8,
                       trials=spark_trials)

# COMMAND ----------

best_params

# COMMAND ----------

# DBTITLE 1,Advanced Artifact Tracking
import matplotlib.pyplot as plt

with mlflow.start_run(run_name='SHAP_demo'):
    #creating the model:
    rf = RandomForestClassifier(n_estimators=int(best_params['n_estimators']),
                                max_depth=int(best_params['max_depth']),
                                min_samples_leaf=int(best_params['min_samples_leaf']),
                                min_samples_split=int(best_params["min_samples_split"]))
    rf.fit(X_train,y_train)

    #generating shap plots from first 5 records:
    mlflow.shap.log_explanation(rf.predict,X_train[:5])

    #generate feature importance plot:
    feature_importances = pd.Series(rf.feature_importances_,index=X_train.columns)
    fig,ax = plt.subplots()
    feature_importances.plot.bar(ax=ax)
    ax.set_title("Feature Importances using SHAP")

    #log the figure:
    mlflow.log_figure(fig,"Feature_Importance.png")

# COMMAND ----------



# Databricks notebook source
#importing a dummy dataset:
path = '/FileStore/credit_risk_data.csv'
data = spark.read.csv(path,header=True,inferSchema=True)
data.display()

# COMMAND ----------

#saving the data to a delta table:
data.write.format('delta').save('/dbfs/user/hive/credit_risk_cleaned')

# COMMAND ----------

#reading the delta table:
delta_df = spark.read.format('delta').load('/dbfs/user/hive/credit_risk_cleaned')
display(delta_df)

# COMMAND ----------

#droping a column and overwriting the table:
delta_df = delta_df.drop('member_id')
delta_df.write.format('delta').mode('overwrite').save('/dbfs/user/hive/credit_risk_cleaned')

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE HISTORY '/dbfs/user/hive/credit_risk_cleaned'

# COMMAND ----------

#reloading the older version:
delta_df = spark.read.format("delta").option('versionAsOf',0).load('/dbfs/user/hive/credit_risk_cleaned')
delta_df.display()

# COMMAND ----------

# DBTITLE 1,Feature Store:
#importing feature store:
from databricks import feature_store
fs = feature_store.FeatureStoreClient()
help(fs.create_table)

# COMMAND ----------

#creating a feature table:
table_name = 'credit_risk'
fs.create_table(
    name=table_name,
    schema=delta_df.schema,
    primary_keys=['member_id'],
    description='Original Credit risk Data'
)

# COMMAND ----------

#populating the data to the feature table:
fs.write_table(
    name=table_name,
    df=delta_df,
    mode='overwrite'
)

# COMMAND ----------

#reviewing the meta data of the feature table:
print(fs.get_table(table_name).description)
print(fs.get_table(table_name).path_data_sources)

# COMMAND ----------

delta_df.columns

# COMMAND ----------

#removing not required columns:
# list of columns to be removed: ('member_id','funded_amnt_inv','batch_enrolled','pymnt_plan','desc','title','zip_code','mths_since_last_delinq','mths_since_last_record','mths_since_last_major_derog','verification_status_joint')

new_df = delta_df.drop('funded_amnt_inv','batch_enrolled','pymnt_plan','desc','title','zip_code','mths_since_last_delinq','mths_since_last_record','mths_since_last_major_derog','emp_title','verification_status_joint')
display(new_df)

# COMMAND ----------

#overwriting the feature table:
fs.write_table(
    name=table_name,
    mode='overwrite',
    df=new_df
)

# COMMAND ----------

#reading the data from a feature table:
feature_df = fs.read_table(name=table_name)
feature_df.display()

# COMMAND ----------



#!/usr/bin/env python
# coding: utf-8

# In[132]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import warnings
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from prophet.diagnostics import cross_validation,performance_metrics
from prophet.plot import plot_cross_validation_metric
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


# In[133]:


#data pre processing 
data = pd.read_csv('removed duplicates/cleaned_concatenated_data_without_duplicates.csv')
columns_to_drop = ['LATITUDE', 'LONGITUDE']
data=data.drop(columns=columns_to_drop)


# In[ ]:


data['TIME'] = pd.to_datetime(data['TIME'])

data['TIME'] = pd.to_datetime(data['TIME'])


pandemic_start_date = '2020-03-13'
pandemic_end_date = '2021-07-01'


pre_pandemic_data = data[data['TIME'] < pandemic_start_date]
pandemic_data = data[(data['TIME'] >= pandemic_start_date) & (data['TIME'] <= pandemic_end_date)]
post_pandemic_data = data[data['TIME'] > pandemic_end_date]


# In[ ]:


data['bike_usage'] = (data['BIKE_STANDS'] - data['AVAILABLE_BIKES']) / data['BIKE_STANDS']
daily_usage_1 = data.groupby(data['TIME'].dt.date)['bike_usage'].mean()
daily_usage_1


# In[ ]:


pre_pandemic_data['bike_usage'] = (pre_pandemic_data['BIKE_STANDS'] - pre_pandemic_data['AVAILABLE_BIKES']) / pre_pandemic_data['BIKE_STANDS']
daily_bike_usage_pre_pandemic_data = pre_pandemic_data.groupby(pre_pandemic_data['TIME'].dt.date)['bike_usage'].mean()
daily_bike_usage_pre_pandemic_data


# In[ ]:


pandemic_data['bike_usage'] = (pandemic_data['BIKE_STANDS'] - pandemic_data['AVAILABLE_BIKES']) / pandemic_data['BIKE_STANDS']
daily_bike_usage_pandemic_data = pandemic_data.groupby(pandemic_data['TIME'].dt.date)['bike_usage'].mean()
daily_bike_usage_pandemic_data


# In[ ]:


post_pandemic_data['bike_usage'] = (post_pandemic_data['BIKE_STANDS'] - post_pandemic_data['AVAILABLE_BIKES']) / post_pandemic_data['BIKE_STANDS']
daily_bike_usage_post_pandemic_data = post_pandemic_data.groupby(post_pandemic_data['TIME'].dt.date)['bike_usage'].mean()
daily_bike_usage_post_pandemic_data


# In[ ]:


pandemic_data['TIME'] = pd.to_datetime(pandemic_data['TIME'])


pandemic_data['DATE'] = pandemic_data['TIME'].dt.date


numeric_columns = pandemic_data.select_dtypes(include=['float64', 'int64']).columns
daily_average_pandemic = pandemic_data.groupby('DATE')[numeric_columns].mean().reset_index()

daily_average_pandemic
    


# In[ ]:


pre_pandemic_data['TIME'] = pd.to_datetime(pre_pandemic_data['TIME'])


pre_pandemic_data['DATE'] = pre_pandemic_data['TIME'].dt.date


numeric_columns = pre_pandemic_data.select_dtypes(include=['float64', 'int64']).columns
daily_average_pre = pre_pandemic_data.groupby('DATE')[numeric_columns].mean().reset_index()


daily_average_pre
    


# In[ ]:


post_pandemic_data['TIME'] = pd.to_datetime(post_pandemic_data['TIME'])


post_pandemic_data['DATE'] = post_pandemic_data['TIME'].dt.date


numeric_columns = post_pandemic_data.select_dtypes(include=['float64', 'int64']).columns
daily_average_post = post_pandemic_data.groupby('DATE')[numeric_columns].mean().reset_index()


daily_average_post


# In[ ]:


plt.figure(figsize=(12, 6))
plt.plot(daily_average_pre['DATE'], daily_average_pre['bike_usage'], label='Pre-Pandemic', color='blue')
plt.plot(daily_average_pandemic['DATE'], daily_average_pandemic['bike_usage'], label='During-Pandemic', color='orange')
plt.plot(daily_average_post['DATE'], daily_average_post['bike_usage'], label='Post-Pandemic', color='green')
plt.xlabel('Date')
plt.ylabel('Bike Usage')
plt.title('Average Daily Bike Usage Over Time')
plt.legend()
plt.show()


plt.figure(figsize=(10, 6))
periods = pd.concat([
    daily_average_pre.assign(Period='Pre-Pandemic'),
    daily_average_pandemic.assign(Period='During-Pandemic'),
    daily_average_post.assign(Period='Post-Pandemic')
])

# Plotting Distribution of Bike Usage for Pre-Pandemic
plt.figure(figsize=(8, 6))
sns.histplot(data=daily_average_pre, x='bike_usage', kde=True, color='blue')
plt.xlabel('Bike Usage')
plt.ylabel('Frequency')
plt.title('Distribution of Bike Usage - Pre-Pandemic')
plt.show()

# Plotting Distribution of Bike Usage for During-Pandemic
plt.figure(figsize=(8, 6))
sns.histplot(data=daily_average_pandemic, x='bike_usage', kde=True, color='orange')
plt.xlabel('Bike Usage')
plt.ylabel('Frequency')
plt.title('Distribution of Bike Usage - During-Pandemic')
plt.show()

# Plotting Distribution of Bike Usage for Post-Pandemic
plt.figure(figsize=(8, 6))
sns.histplot(data=daily_average_post, x='bike_usage', kde=True, color='green')
plt.xlabel('Bike Usage')
plt.ylabel('Frequency')
plt.title('Distribution of Bike Usage - Post-Pandemic')
plt.show()



# In[ ]:


fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plotting Distribution of Bike Usage for Pre-Pandemic
sns.histplot(data=daily_average_pre, x='bike_usage', kde=True, color='blue', ax=axes[0])
axes[0].set_xlabel('Bike Usage')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of Bike Usage - Pre-Pandemic')

# Plotting Distribution of Bike Usage for During-Pandemic
sns.histplot(data=daily_average_pandemic, x='bike_usage', kde=True, color='orange', ax=axes[1])
axes[1].set_xlabel('Bike Usage')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Distribution of Bike Usage - During-Pandemic')

# Plotting Distribution of Bike Usage for Post-Pandemic
sns.histplot(data=daily_average_post, x='bike_usage', kde=True, color='green', ax=axes[2])
axes[2].set_xlabel('Bike Usage')
axes[2].set_ylabel('Frequency')
axes[2].set_title('Distribution of Bike Usage - Post-Pandemic')

plt.tight_layout()
plt.show()


# In[ ]:


import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plotting Aggregated Daily Bike Usage Over Time for Pre-Pandemic
axes[0].plot(daily_average_pre['DATE'], daily_average_pre['bike_usage'], marker='o', linestyle='-')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Average Bike Usage')
axes[0].set_title('Aggregated Daily Bike Usage Over Time (Pre-Pandemic)')
axes[0].tick_params(axis='x', rotation=45)

# Plotting Aggregated Daily Bike Usage Over Time for During-Pandemic
axes[1].plot(daily_average_pandemic['DATE'], daily_average_pandemic['bike_usage'], marker='o', linestyle='-')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Average Bike Usage')
axes[1].set_title('Aggregated Daily Bike Usage Over Time (During-Pandemic)')
axes[1].tick_params(axis='x', rotation=45)

# Plotting Aggregated Daily Bike Usage Over Time for Post-Pandemic
axes[2].plot(daily_average_post['DATE'], daily_average_post['bike_usage'], marker='o', linestyle='-')
axes[2].set_xlabel('Date')
axes[2].set_ylabel('Average Bike Usage')
axes[2].set_title('Aggregated Daily Bike Usage Over Time (Post-Pandemic)')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()


# In[ ]:


# Calculate statistical measures for bike usage
def calculate_statistics(data):
    mean_usage = data['bike_usage'].mean()
    median_usage = data['bike_usage'].median()
    std_dev_usage = data['bike_usage'].std()
    min_usage = data['bike_usage'].min()
    max_usage = data['bike_usage'].max()

    return {
        'Mean': mean_usage,
        'Median': median_usage,
        'Standard Deviation': std_dev_usage,
        'Minimum': min_usage,
        'Maximum': max_usage
    }

# Calculate statistics for pre-pandemic, during-pandemic, and post-pandemic data
statistics_pre_pandemic = calculate_statistics(daily_average_pre)
statistics_during_pandemic = calculate_statistics(daily_average_pandemic)
statistics_post_pandemic = calculate_statistics(daily_average_post)

# Print statistics
print("Statistics for Pre-Pandemic Bike Usage:")
print(statistics_pre_pandemic)

print("\nStatistics for During-Pandemic Bike Usage:")
print(statistics_during_pandemic)

print("\nStatistics for Post-Pandemic Bike Usage:")
print(statistics_post_pandemic)




# In[ ]:


#forecasting and training ARIMA model from pre period data and forecasting for during pandemic and after pandemic data 
def fit_arima_train(data):
    data['bike_usage'] = pd.to_numeric(data['bike_usage'], errors='coerce')
    model = ARIMA(data['bike_usage'], order=(5,0, 5))  
    model_fit = model.fit()
    return model_fit

def predict_arima(model, data):
    predictions = model.predict(start=0, end=len(data) - 1) 
    return predictions


train_data_pandemic_pre = daily_average_pre  
model_pandemic_pre = fit_arima_train(train_data_pandemic_pre)


test_data_pandemic = daily_average_pandemic  
pandemic_predictions = predict_arima(model_pandemic_pre, test_data_pandemic)



test_data_post_pandemic = daily_average_post  
pandemic_predictions_post = predict_arima(model_pandemic_pre, test_data_post_pandemic)


# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

# Plotting actual vs predicted pandemic bike usage
axes[0].plot(test_data_pandemic['DATE'], test_data_pandemic['bike_usage'], marker='o', linestyle='-', label='Actual - Pandemic', color='blue')
axes[0].plot(test_data_pandemic['DATE'], pandemic_predictions, marker='o', linestyle='-', label='Predicted - Pandemic', color='orange')

axes[0].set_xlabel('Date')
axes[0].set_ylabel('Average Bike Usage')
axes[0].set_title('Actual vs Predicted Daily Bike Usage Over Time (Pandemic)')
axes[0].tick_params(rotation=45)
axes[0].legend()

# Plotting actual vs predicted post-pandemic bike usage
axes[1].plot(test_data_post_pandemic['DATE'], test_data_post_pandemic['bike_usage'], marker='o', linestyle='-', label='Actual - Post-Pandemic', color='green')
axes[1].plot(test_data_post_pandemic['DATE'], pandemic_predictions_post, marker='o', linestyle='-', label='Predicted - Post-Pandemic', color='red')

axes[1].set_xlabel('Date')
axes[1].set_ylabel('Average Bike Usage')
axes[1].set_title('Actual vs Predicted Daily Bike Usage Over Time (Post-Pandemic)')
axes[1].tick_params(rotation=45)
axes[1].legend()

plt.tight_layout()
plt.show()


# In[ ]:


#Evaluation of ARIMA model 
def calculate_mape(y_true, y_pred):
    mask = y_true != 0 
    y_true_masked = y_true[mask]
    y_pred_masked = y_pred[mask]
    return np.mean(np.abs((y_true_masked - y_pred_masked) / y_true_masked)) * 100
def calculate_mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

# Calculate MAPE for during-pandemic predictions
mape_during_pandemic = calculate_mape(test_data_pandemic['bike_usage'].values, pandemic_predictions)
rmse_during_pandemic = np.sqrt(mean_squared_error(test_data_pandemic['bike_usage'].values, pandemic_predictions))
accuracy_during_pandemic = 100 - mape_during_pandemic
mae_during_pandemic = calculate_mae(test_data_pandemic['bike_usage'].values, pandemic_predictions)
print("Evaluation Metrics - During-Pandemic ARIMA:")
print(f"MAPE : {mape_during_pandemic:.2f}%")
print(f"RMSE : {rmse_during_pandemic:.2f}")
print(f"MAE : {mae_during_pandemic:.2f}")
print("\n")
# Calculate MAPE for post-pandemic predictions
mape_post_pandemic = calculate_mape(test_data_post_pandemic['bike_usage'].values, pandemic_predictions_post)
rmse_post_pandemic = np.sqrt(mean_squared_error(test_data_post_pandemic['bike_usage'].values, pandemic_predictions_post))
mae_post_pandemic = calculate_mae(test_data_post_pandemic['bike_usage'].values, pandemic_predictions_post)
accuracy_post_pandemic = 100 - mape_post_pandemic
print("Evaluation Metrics -Post-Pandemi ARIMA:")
print(f"MAPE : {mape_post_pandemic:.2f}%")
print(f"RMSE : {rmse_post_pandemic:.2f}")
print(f"MAE : {mae_post_pandemic:.2f}")

print("\n")

# Calculate combined accuracy for both periods
combined_mae = (mae_during_pandemic + mae_post_pandemic) / 2
combined_mape = (mape_during_pandemic + mape_post_pandemic) / 2
combined_rmse = (rmse_during_pandemic + rmse_post_pandemic) / 2
combined_accuracy = (accuracy_during_pandemic + accuracy_post_pandemic) / 2
print("Evaluation Metrics - During and After Pandemic combined ARIMA:")
print(f"MAE: {combined_mae:.2f}")
print(f"MAPE: {combined_mape:.2f}%")
print(f"RMSE: {combined_rmse:.2f}")



# In[ ]:


#Using Prophet model to forecast data duriong and post pandemic data 
daily_average_pre = daily_average_pre.rename(columns={'DATE': 'ds', 'bike_usage': 'y'})
daily_average_pandemic = daily_average_pandemic.rename(columns={'DATE': 'ds', 'bike_usage': 'y'})
daily_average_post = daily_average_post.rename(columns={'DATE': 'ds', 'bike_usage': 'y'})


def forecast_prophet(data):
    model = Prophet()
    model.fit(data)
    future = model.make_future_dataframe(periods=200)  
    forecast = model.predict(future)
    return model, forecast


model_pre, forecast_pre = forecast_prophet(daily_average_pre)
model_pandemic, forecast_pandemic = forecast_prophet(daily_average_pandemic)
model_post, forecast_post = forecast_prophet(daily_average_post)


fig, ax = plt.subplots(3, 1, figsize=(12, 8))


model_pre.plot(forecast_pre, ax=ax[0], xlabel='Date', ylabel='Bike Usage', plot_cap=False)
ax[0].plot(daily_average_pre['ds'], daily_average_pre['y'], color='black', label='Actual - Pre-Pandemic', linestyle='--')
ax[0].set_title('Forecast - Pre-Pandemic')

model_pandemic.plot(forecast_pandemic, ax=ax[1], xlabel='Date', ylabel='Bike Usage', plot_cap=False)
ax[1].plot(daily_average_pandemic['ds'], daily_average_pandemic['y'], color='black', label='Actual - During-Pandemic', linestyle='--')
ax[1].set_title('Forecast - During-Pandemic')

model_post.plot(forecast_post, ax=ax[2], xlabel='Date', ylabel='Bike Usage', plot_cap=False)
ax[2].plot(daily_average_post['ds'], daily_average_post['y'], color='black', label='Actual - Post-Pandemic', linestyle='--')
ax[2].set_title('Forecast - Post-Pandemic')


ax[0].legend()
ax[1].legend()
ax[2].legend()

plt.tight_layout()
plt.show()


df_cv = cross_validation(model_pandemic, horizon='250 days', period='90 days', initial='200 days')
df_metrics = performance_metrics(df_cv)


metrics = ['mse', 'rmse', 'mae', 'mape', 'mdape', 'smape', 'coverage']


for metric in metrics:
    fig = plot_cross_validation_metric(df_cv, metric=metric)


# In[ ]:


# evaluating Prophet model 


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


mae_pre = mean_absolute_error(daily_average_pre['y'], forecast_pre['yhat'][:len(daily_average_pre)])
mse_pre = mean_squared_error(daily_average_pre['y'], forecast_pre['yhat'][:len(daily_average_pre)])
mape_pre = mean_absolute_percentage_error(daily_average_pre['y'], forecast_pre['yhat'][:len(daily_average_pre)])
rmse_pre = np.sqrt(mse_pre)


mae_pandemic = mean_absolute_error(daily_average_pandemic['y'], forecast_pandemic['yhat'][:len(daily_average_pandemic)])
mse_pandemic = mean_squared_error(daily_average_pandemic['y'], forecast_pandemic['yhat'][:len(daily_average_pandemic)])
mape_pandemic = mean_absolute_percentage_error(daily_average_pandemic['y'], forecast_pandemic['yhat'][:len(daily_average_pandemic)])
rmse_pandemic = np.sqrt(mse_pandemic)

mae_post = mean_absolute_error(daily_average_post['y'], forecast_post['yhat'][:len(daily_average_post)])
mse_post = mean_squared_error(daily_average_post['y'], forecast_post['yhat'][:len(daily_average_post)])
mape_post = mean_absolute_percentage_error(daily_average_post['y'], forecast_post['yhat'][:len(daily_average_post)])
rmse_post = np.sqrt(mse_post)

predicted_values_post = forecast_post['yhat'][:len(daily_average_post)]
actual_values_post = daily_average_post['y']
accuracy_ratio_post = np.mean(np.abs(actual_values_post - predicted_values_post) / np.abs(actual_values_post))
accuracy_percentage_post = (1 - accuracy_ratio_post) * 100

predicted_values_pandemic = forecast_pandemic['yhat'][:len(daily_average_pandemic)]
actual_values_pandemic = daily_average_pandemic['y']
accuracy_ratio_pandemic = np.mean(np.abs(actual_values_pandemic - predicted_values_pandemic) / np.abs(actual_values_pandemic))
accuracy_percentage_pandemic = (1 - accuracy_ratio_pandemic) * 100

print("Evaluation Metrics - During-Pandemic Prophet:")
print(f"MAE: {mae_pandemic}")
print(f"MSE: {mse_pandemic}")
print(f"RMSE: {rmse_pandemic}")
print(f"MAPE: {mape_pandemic:.2f}%")
print("\n")

print("Evaluation Metrics - Post-Pandemic Prophet:")
print(f"MAE: {mae_post}")
print(f"MSE: {mse_post}")
print(f"MAPE: {mape_post:.2f}%")
print(f"RMSE: {rmse_post}")

print("\n")


predicted_values_during_after = np.concatenate((forecast_pandemic['yhat'][:len(daily_average_pandemic)], forecast_post['yhat'][:len(daily_average_post)]))
actual_values_during_after = np.concatenate((daily_average_pandemic['y'], daily_average_post['y']))


# Calculating MAE, MSE, and RMSE for during and after pandemic together
mae_during_after = mean_absolute_error(actual_values_during_after, predicted_values_during_after)
mse_during_after = mean_squared_error(actual_values_during_after, predicted_values_during_after)
rmse_during_after = np.sqrt(mse_during_after)

# Calculating accuracy percentage for during and after pandemic together
accuracy_ratio_during_after = np.mean(np.abs(actual_values_during_after - predicted_values_during_after) / np.abs(actual_values_during_after))
accuracy_percentage_during_after = (1 - accuracy_ratio_during_after) * 100

mape_during_after = mean_absolute_percentage_error(actual_values_during_after, predicted_values_during_after)

print("Evaluation Metrics - During and After Pandemic combined Prophet:")
print(f"MAE: {mae_during_after}")
print(f"MSE: {mse_during_after}")
print(f"RMSE: {rmse_during_after}")
print(f"MAPE: {mape_during_after:.2f}%")




# In[ ]:


predicted_values_during_after = np.concatenate((forecast_pandemic['yhat'][:len(daily_average_pandemic)], forecast_post['yhat'][:len(daily_average_post)]))
actual_values_during_after = np.concatenate((daily_average_pandemic['y'], daily_average_post['y']))


# Calculate MAE, MSE, and RMSE for during and after pandemic together
mae_during_after = mean_absolute_error(actual_values_during_after, predicted_values_during_after)
mse_during_after = mean_squared_error(actual_values_during_after, predicted_values_during_after)
rmse_during_after = np.sqrt(mse_during_after)

# Calculate accuracy percentage for during and after pandemic together
accuracy_ratio_during_after = np.mean(np.abs(actual_values_during_after - predicted_values_during_after) / np.abs(actual_values_during_after))
accuracy_percentage_during_after = (1 - accuracy_ratio_during_after) * 100

mape_during_after = mean_absolute_percentage_error(actual_values_during_after, predicted_values_during_after)

print("Evaluation Metrics - During and After Pandemic:")
print(f"MAE: {mae_during_after}")
print(f"MSE: {mse_during_after}")
print(f"RMSE: {rmse_during_after}")
print(f"MAPE: {mape_during_after:.2f}%")
print(f"Accuracy Percentage: {accuracy_percentage_during_after:.2f}%")


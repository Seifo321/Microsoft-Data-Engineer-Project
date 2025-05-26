import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose

# Load historical inventory data
try:
    df_store = pd.read_excel('E:\Training\Data\Data Engineer DEPI\Technical\Final Project\data\BikeStore.xlsx', sheet_name='sales stores')
    df_staff = pd.read_excel('E:\Training\Data\Data Engineer DEPI\Technical\Final Project\data\BikeStore.xlsx', sheet_name='sales staffs')
    df_orders = pd.read_excel('E:\Training\Data\Data Engineer DEPI\Technical\Final Project\data\BikeStore.xlsx', sheet_name='sales orders')
    df_order_items = pd.read_excel('E:\Training\Data\Data Engineer DEPI\Technical\Final Project\data\BikeStore.xlsx', sheet_name='sales order_items')
    df_customers = pd.read_excel('E:\Training\Data\Data Engineer DEPI\Technical\Final Project\data\BikeStore.xlsx', sheet_name='sales customers')
    df_stocks = pd.read_excel('E:\Training\Data\Data Engineer DEPI\Technical\Final Project\data\BikeStore.xlsx', sheet_name='production stocks')
    df_product = pd.read_excel('E:\Training\Data\Data Engineer DEPI\Technical\Final Project\data\BikeStore.xlsx', sheet_name='production products')
    df_categories = pd.read_excel('E:\Training\Data\Data Engineer DEPI\Technical\Final Project\data\BikeStore.xlsx', sheet_name='production categories')
    df_brand = pd.read_excel('E:\Training\Data\Data Engineer DEPI\Technical\Final Project\data\BikeStore.xlsx', sheet_name='production brands')

    # Clean and preprocess the data
    df_orders['order_date'] = pd.to_datetime(df_orders['order_date'], errors='coerce')
    df_orders.dropna(subset=['order_date'], inplace=True)

except Exception as e:
    print("Error loading data:", e)

# Data Overview
df_store.head()
df_staff.head()
df_orders.head()
df_order_items.head()
df_customers.head()
df_stocks.head()
df_product.head()
df_categories.head()
df_brand.head()

# Data Overview
df_store.info()
df_staff.info() 
df_orders.info()
df_order_items.info()

df_store.describe(include='all').T
df_staff.describe(include='all').T
df_orders.describe(include='all').T
df_order_items.describe(include='all').T

# Data Clean
df_store.isna().sum().sum()
df_store.duplicated().sum()
df_staff.isnull().sum().sum()
df_staff.fillna(0, inplace=True)
df_staff.isnull().sum().sum()
df_staff.duplicated().sum()
df_orders.isnull().sum().sum()
orders_null = df_orders[df_orders.isnull().any(axis=1)]
orders_null
df_orders.duplicated().sum()
df_order_items.isnull().sum().sum()
df_order_items.duplicated().sum()

# Merge the DataFrames on 'store_id'
df_merged = pd.merge(df_staff, df_store, on='store_id', how='inner')

# Display the merged DataFrame
df_merged

missing_data = {
    "store": df_store.isnull().sum(),
    "staff": df_staff.isnull().sum(),
    "orders": df_orders.isnull().sum(),
    "order_items": df_order_items.isnull().sum(),
    "customers": df_customers.isnull().sum(),
    "stocks": df_stocks.isnull().sum(),
    "products": df_product.isnull().sum(),
    "categories": df_categories.isnull().sum(),
    "brands": df_brand.isnull().sum(),
}
print(missing_data)

df_orders['order_date'] = pd.to_datetime(df_orders['order_date'])
df_orders.set_index('order_date', inplace=True)

# Resample orders by month to see trends
monthly_orders = df_orders.resample('M').size()

# Plot monthly sales orders
plt.figure(figsize=(10, 6))
monthly_orders.plot()
plt.title('Monthly Sales Orders')
plt.xlabel('Date')
plt.ylabel('Number of Orders')
plt.grid(True)
plt.show()

df_orders['order_id'].plot(figsize=(10, 6), title='Order ID Time Series Plot', xlabel='Date', ylabel='Order ID')
plt.show()

df_orders['order_id'].diff().dropna().plot(figsize=(10, 6), title='Differenced Order ID Time Series Plot', xlabel='Date', ylabel='Differenced Order ID')
plt.show()

decomposition = seasonal_decompose(df_orders['order_id'], model='additive', period=12)
decomposition.plot()
plt.show()


rolmean = df_orders['order_id'].rolling(window=12).mean()
rolstd = df_orders['order_id'].rolling(window=12).std()

plt.plot(df_orders['order_id'], label='Original')
plt.plot(rolmean, label='Rolling Mean', color='red')
plt.plot(rolstd, label='Rolling Std', color='black')
plt.title('Rolling Mean & Standard Deviation')
plt.legend(loc='best')
plt.show()


# Prepare the data for forecasting (grouping by date)
df_orders['order_date'] = df_orders['order_date'].dt.date
daily_orders = df_orders.groupby('order_date').size()

# Split data into train and test sets
train_size = int(len(daily_orders) * 0.8)
train, test = daily_orders[:train_size], daily_orders[train_size:]

# Cross-validation function
def time_series_cv(model_class, train_data, steps=12):
    errors = []
    for i in range(steps):
        train_cv, test_cv = train_data[:-steps+i], train_data[-steps+i:-steps+i+1]
        model = model_class(train_cv)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=1)
        errors.append(mean_absolute_error(test_cv, forecast))
    return np.mean(errors)

# Fit ARIMA model
try:
    arima_order = (5, 1, 0)  # Adjust as necessary
    arima_model = ARIMA(train, order=arima_order)
    arima_fit = arima_model.fit()
    arima_forecast = arima_fit.forecast(steps=len(test))

    # Cross-validation for ARIMA
    arima_cv_error = time_series_cv(lambda x: ARIMA(x, order=arima_order), train)
    print(f"Cross-validated MAE for ARIMA: {arima_cv_error}")

except Exception as e:
    print("Error fitting ARIMA model:", e)

# Fit Exponential Smoothing model
try:
    exp_smoothing_model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=12)
    exp_smoothing_fit = exp_smoothing_model.fit()
    exp_smoothing_forecast = exp_smoothing_fit.forecast(steps=len(test))

    # Cross-validation for Exponential Smoothing
    es_cv_error = time_series_cv(lambda x: ExponentialSmoothing(x, trend='add', seasonal='add', seasonal_periods=12), train)
    print(f"Cross-validated MAE for Exponential Smoothing: {es_cv_error}")

except Exception as e:
    print("Error fitting Exponential Smoothing model:", e)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(daily_orders.index, daily_orders.values, label='Historical Orders')
plt.plot(test.index, arima_forecast, label='ARIMA Forecast', color='red')
plt.plot(test.index, exp_smoothing_forecast, label='Exponential Smoothing Forecast', color='green')
plt.title('Demand Forecasting')
plt.xlabel('Date')
plt.ylabel('Number of Orders')
plt.legend()
plt.show()


# Assume we have some actual values for comparison
actual_values = daily_orders[-len(test):]  # Actual values from the test set

# Evaluate the ARIMA model
try:
    mae_arima = mean_absolute_error(actual_values, arima_forecast)
    mse_arima = mean_squared_error(actual_values, arima_forecast)
    r2_arima = r2_score(actual_values, arima_forecast)

    print(f"ARIMA - MAE: {mae_arima}, MSE: {mse_arima}, R²: {r2_arima}")

except Exception as e:
    print("Error evaluating ARIMA model:", e)

# Evaluate the Exponential Smoothing model
try:
    mae_es = mean_absolute_error(actual_values, exp_smoothing_forecast)
    mse_es = mean_squared_error(actual_values, exp_smoothing_forecast)
    r2_es = r2_score(actual_values, exp_smoothing_forecast)

    print(f"Exponential Smoothing - MAE: {mae_es}, MSE: {mse_es}, R²: {r2_es}")

except Exception as e:
    print("Error evaluating Exponential Smoothing model:", e)

# Prepare data for plotting
metrics = ['MAE', 'MSE', 'R²']
arima_scores = [mae_arima, mse_arima, r2_arima]
es_scores = [mae_es, mse_es, r2_es]

# Bar width
bar_width = 0.35
x = np.arange(len(metrics))

# Create bar plots
plt.figure(figsize=(10, 6))
plt.bar(x - bar_width/2, arima_scores, width=bar_width, label='ARIMA', color='blue')
plt.bar(x + bar_width/2, es_scores, width=bar_width, label='Exponential Smoothing', color='orange')

# Add labels and title
plt.xlabel('Metrics')
plt.ylabel('Scores')
plt.title('Model Evaluation Metrics Comparison')
plt.xticks(x, metrics)
plt.legend()

# Show the plot
plt.grid(axis='y')
plt.tight_layout()
plt.show()


# Part 4: Testing the Forecast Models
# Use the forecasts to predict future orders
future_steps = 30  # Define how many days you want to predict
try:
    future_arima_forecast = arima_fit.forecast(steps=future_steps)
    future_exp_smoothing_forecast = exp_smoothing_fit.forecast(steps=future_steps)

    # Create a date range for future predictions
    future_dates = pd.date_range(start=daily_orders.index[-1] + pd.Timedelta(days=1), periods=future_steps)

    # Plot future forecasts
    plt.figure(figsize=(12, 6))
    plt.plot(daily_orders.index, daily_orders.values, label='Historical Orders')
    plt.plot(test.index, arima_forecast, label='ARIMA Forecast', color='red')
    plt.plot(test.index, exp_smoothing_forecast, label='Exponential Smoothing Forecast', color='green')
    plt.plot(future_dates, future_arima_forecast, label='Future ARIMA Forecast', linestyle='--', color='orange')
    plt.plot(future_dates, future_exp_smoothing_forecast, label='Future Exponential Smoothing Forecast', linestyle='--', color='purple')
    plt.title('Demand Forecasting with Future Predictions')
    plt.xlabel('Date')
    plt.ylabel('Number of Orders')
    plt.legend()
    plt.show()

except Exception as e:
    print("Error predicting future orders:", e)


# Set the number of future periods you want to forecast (e.g., next 12 days)
future_steps = 12

# Merge df_order_items with df_orders to associate order dates with products
df_merged = pd.merge(
    df_order_items,
    df_orders[['order_id', 'order_date']],
    on='order_id'
)

# Convert 'order_date' to datetime format
df_merged['order_date'] = pd.to_datetime(df_merged['order_date'], errors='coerce')

# Prepare the data for forecasting (grouping by product and date)
daily_product_demand = df_merged.groupby(['product_id', df_merged['order_date'].dt.date]).size().unstack(fill_value=0)

# Initialize dictionary to store forecasts
product_forecasts = {}

# Fit ARIMA model for each product to forecast future demand
for product_id in daily_product_demand.index:
    try:
        product_demand = daily_product_demand.loc[product_id]
        train_size = int(len(product_demand) * 0.8)
        train = product_demand[:train_size]

        # Fit ARIMA model (adjust order as necessary)
        arima_order = (5, 1, 0)  # ARIMA parameters (p,d,q)
        arima_model = ARIMA(train, order=arima_order)
        arima_fit = arima_model.fit()

        # Forecast future demand
        future_forecast = arima_fit.forecast(steps=future_steps)
        product_forecasts[product_id] = future_forecast

    except Exception as e:
        print(f"Error forecasting for product {product_id}: {e}")

# Convert forecasts to a DataFrame for easier handling
forecast_df = pd.DataFrame(product_forecasts)

# Calculate the total forecast demand for all products
total_forecast_demand = forecast_df.sum(axis=0)

# Calculate percentage of demand for each product
percentage_demand = (forecast_df.div(total_forecast_demand, axis=1)) * 100

# Prepare the output DataFrame with product IDs and their forecasted percentage demand
output = pd.DataFrame({
    'product_id': forecast_df.columns,
    'percentage_demand': percentage_demand.mean(axis=0)  # Average percentage across the forecast period
})

# Sort the output by percentage of demand
output_sorted = output.sort_values(by='percentage_demand', ascending=False)

# Print the results
print(output_sorted)


# Plotting the forecasted percentage demand
plt.figure(figsize=(12, 6))
plt.bar(output_sorted['product_id'], output_sorted['percentage_demand'], color='skyblue')
plt.xlabel('Product ID')
plt.ylabel('Percentage Demand')
plt.title('Forecasted Percentage Demand for Products')
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()

# Set the style for the plots
sns.set(style="whitegrid")

# Create a bar plot for the percentage of demand for each product
plt.figure(figsize=(12, 6))
sns.barplot(x='percentage_demand', y='product_id', data=output_sorted, palette='viridis')

# Adding titles and labels
plt.title('Forecasted Percentage Demand by Product', fontsize=16)
plt.xlabel('Percentage Demand (%)', fontsize=12)
plt.ylabel('Product ID', fontsize=12)
plt.xlim(0, 100)  # Adjust x-axis limit for better visualization
plt.grid(axis='x')

# Show the plot
plt.show()

# Prepare the data for visualization
plt.figure(figsize=(14, 7))

for product_id in daily_product_demand.index:
    try:
        product_demand = daily_product_demand.loc[product_id]
        train_size = int(len(product_demand) * 0.8)
        train = product_demand[:train_size]

        # Fit ARIMA model
        arima_model = ARIMA(train, order=arima_order)
        arima_fit = arima_model.fit()

        # Forecast future demand
        future_forecast = arima_fit.forecast(steps=future_steps)

        # Create a time index for future steps
        future_index = pd.date_range(start=product_demand.index[-1] + pd.Timedelta(days=1), periods=future_steps)

        # Plot historical and forecasted demand
        plt.plot(product_demand.index, product_demand, label=f'Historical Demand for Product {product_id}')
        plt.plot(future_index, future_forecast, label=f'Forecasted Demand for Product {product_id}', linestyle='--')

    except Exception as e:
        print(f"Error forecasting for product {product_id}: {e}")

# Adding titles and labels
plt.title('Historical and Forecasted Demand by Product', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Number of Orders', fontsize=12)
plt.legend()
plt.grid()
plt.show()

# Create a line plot for all product forecasts
plt.figure(figsize=(12, 8))

for product_id in forecast_df.columns:
    plt.plot(range(len(forecast_df[product_id])), forecast_df[product_id], marker='o', label=f'Product ID {product_id}')

plt.title('Forecast Demand for All Products')
plt.xlabel('Days Ahead')
plt.ylabel('Forecast Demand')
plt.xticks(range(future_steps), [f'Day {i+1}' for i in range(future_steps)])
plt.legend()
plt.grid()
plt.show()

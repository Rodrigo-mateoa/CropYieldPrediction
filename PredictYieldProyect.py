# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Global variables
FILE_PATH = "/Users/rodrigom/Desktop/Crop Yield"

# Function to read raw data
def read_data():
   country = "Peru"
   item = None
   data = pd.read_csv(f"{FILE_PATH}/yield_df.csv")

   if item is not None:
    filtered_data = data[(data['Area'] == country) & (data['Item'] == item)]
   else:
    filtered_data = data[data['Area'] == country]
   return filtered_data

# Function to train the model using
# the default LinearRegression 
def linear_regression(X, y):
    # Adding a column of ones for the intercept
    A = np.vstack([X, np.ones(len(X))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, c

# Plot using X and Log(Y) values 
def plot_logmodel(X, y, m, c):
    plt.figure(figsize=(10, 5))
    plt.scatter(X, np.log(y), color='blue')  # Assuming y > 0
    plt.plot(X, np.log(m * X + c), 'r', label=f'log(y) = log({m:.2f}x + {c:.2f})')
    plt.xlabel('Year')
    plt.ylabel('Log of Yield')
    plt.title('Linear Regression Model')
    plt.legend(loc='upper left')
    plt.grid(True)

    # Adjusting the plot for X and Y 
    plt.xlim(np.min(X), np.max(X)) 
    plt.ylim(np.min(np.log(y)), np.max(np.log(y)))
    plt.show()

def plot_model(X, y, m, c):
    plt.figure(figsize=(10, 5))
    plt.scatter(X, y, color='blue')
    plt.plot(X, m*X + c, 'r', label=f'y = {m:.2f}x + {c:.2f}')
    plt.xlabel('Feature')
    plt.ylabel('Yield')
    plt.title('Linear Regression Model')
    plt.legend()
    plt.grid(True)

    # Establecer automáticamente los límites para incluir todos los datos
    # plt.xlim(X.min(), X.max()) 
    plt.ylim(y.min(), y.max())

    plt.show()

# Function to calculate Mean Absolute Error
def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# Function to determien the Mean Absolute percentage error 
def mean_absolute_percentage_error(y_true, y_pred):
    # Avoid division by zero and replace zero with a small number (epsilon)
    y_true = np.where(y_true == 0, np.finfo(float).eps, y_true)  # Replace 0 with small epsilon
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # Multiply by 100 to get percentage
    return mape

# Function to calculate R squared
def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

# Main function 
def main(feature_column):
    data = read_data()

    # Loading the data
    X = data[feature_column].values
    y = data['hg/ha_yield'].values

    # Splitting the dataset into training and testing sets
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"Testing: {feature_column}")

    # Calculate regression parameters
    m, c = linear_regression(X_train, y_train)
    print(f"Model parameters: Slope = {m}, Intercept = {c}")

    # Making predictions
    y_pred_test = m * X_test + c

    # Calculating MAE and R^2
    mae = mean_absolute_error(y_test, y_pred_test)
    mape = mean_absolute_percentage_error(y_test, y_pred_test)
    r2 = r_squared(y_test, y_pred_test)
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Absolute Percentage Error: {mape}")
    print(f"R^2 Score: {r2}")

    # Plotting the data with the model
    #plot_logmodel(X_train, y_train, m, c)
    plot_model(X_train, y_train, m, c)

 # Point of entry 
if __name__ == "__main__":
    main('average_rain_fall_mm_per_year')


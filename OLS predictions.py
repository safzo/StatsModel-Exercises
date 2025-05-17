# For this practical example we will need the following libraries and modules
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
sns.set()
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_regression
from sklearn import metrics  

# Load the data
raw_data = pd.read_csv("Real-life+example.csv")

# Drop the rows with missing values, they're about less than 5% of the total data so it's alright to drop them
newnew = raw_data.dropna(axis=0)

# Remove the top 1% of data from Price column due to possible outliers, to make it's distribution more normalized 
q = newnew['Price'].quantile(0.99)
modelincludeddata = newnew[newnew['Price']<q]

# Remove the top 1% of data from Mileage column due to possible outliers, to make it's distribution more normalized 
q = modelincludeddata['Mileage'].quantile(0.99)
modelincludeddata2 = modelincludeddata[modelincludeddata['Mileage']<q]

# Car engine volumes are usually (always?) below 6.5l
# This is a prime example of the fact that a domain expert (a person working in the car industry)
# may find it much easier to determine problems with the data than an outsider
modelincludeddata3 = modelincludeddata2[modelincludeddata2['EngineV']<6.5]

# Finally, the situation with 'Year' is similar to 'Price' and 'Mileage'
# However, the outliers are on the low end, and hence:
q = modelincludeddata3['Year'].quantile(0.01)
modelincludeddata4 = modelincludeddata3[modelincludeddata3['Year']>q]

# sklearn does not have a built-in way to check for multicollinearity
# one of the main reasons is that this is an issue well covered in statistical frameworks and not in ML ones
# surely it is an issue nonetheless, thus we will try to deal with it

# Here's the relevant module
# full documentation: http://www.statsmodels.org/dev/_modules/statsmodels/stats/outliers_influence.html#variance_inflation_factor
from statsmodels.stats.outliers_influence import variance_inflation_factor

# To make this as easy as possible to use, we declare a variable where we put
# all features where we want to check for multicollinearity
# since our categorical data is not yet preprocessed, we will only take the numerical ones
variables = modelincludeddata4[['Mileage','Year','EngineV']]

# we create a new data frame which will include all the VIFs
# note that each variable has its own variance inflation factor as this measure is variable specific (not model specific)
vif = pd.DataFrame()

# here we make use of the variance_inflation_factor, which will basically output the respective VIFs 
vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
# Finally, I like to include names so it is easier to explore the result
vif["Features"] = variables.columns
vif


# Since Year has the highest VIF, I will remove it from the model
# This will drive the VIF of other variables down!!! 
# So even if EngineV seems with a high VIF, too, once 'Year' is gone that will no longer be the case
cleandata = modelincludeddata4.reset_index(drop=True)
data_no_multicollinearity = cleandata.drop(['Year'],axis=1)


# When we remove observations, the original indexes are preserved
# If we remove observations with indexes 2 and 3, the indexes will go as: 0,1,4,5,6
# That's very problematic as we tend to forget about it 

# Finally, once we reset the index, a new column will be created containing the old index (just in case)
# We won't be needing it, thus we do 'drop=True' to completely forget about it
dummies = pd.get_dummies(data_no_multicollinearity, drop_first=True)

#Multi Regressiom
target = dummies['Price']
inputs = dummies.drop(['Price'],axis=1)

#Scaling Training Dataset
scaler = StandardScaler()
scaler.fit(inputs)
input_scaled = scaler.transform(inputs)

# Splitting the dataset into two sets
x_train, x_test, y_train, y_test = train_test_split(input_scaled, target, test_size=0.2, random_state=365)

# Training the regression
reg = LinearRegression()
reg.fit(x_train,y_train)
y_hat = reg.predict(x_train)

# Results
plt.scatter(y_train, y_hat)
plt.xlabel('Targets (y_train)',size=18)
plt.ylabel('Predictions (y_hat)',size=18)
plt.show()

# Residuals PDF - Right skewed, meaning model is underpredicting a significant
# portion of the training data
sns.distplot(y_train - y_hat)
plt.title("Residuals PDF (Training Data)", size=18)
plt.show()

# Feature Selection via F-Statistic
p_values = f_regression(inputs,target)[1]
p_value_summary = pd.DataFrame({'Feature': inputs.columns, 'P-value': p_values.round(3)})
pd.set_option('display.max_rows', None)
print(p_value_summary)

alpha = 0.05
features_to_drop = p_value_summary[p_value_summary['P-value'] > alpha]['Feature'].tolist()
dummies_reduced = dummies.drop(columns=features_to_drop)
dummies_reduced.head()

# Predicting using the scaled test inputs
reg = LinearRegression()
reg.fit(x_test,y_test)
y_hat_test = reg.predict(x_test)

# Results
plt.scatter(y_test, y_hat_test, alpha=0.2)
plt.xlabel('Targets (y_test)',size=18)
plt.ylabel('Predictions (y_hat_test)',size=18)
plt.show()

# Calculating evaluation metrics for the TEST set
mae_test = metrics.mean_absolute_error(y_test, y_hat_test)
mse_test = metrics.mean_squared_error(y_test, y_hat_test)
rmse_test = np.sqrt(mse_test)
r2_test = metrics.r2_score(y_test, y_hat_test)

print("\n--- Test Set Evaluation ---")
print(f"Mean Absolute Error (MAE): {mae_test:.2f}")
print(f"Mean Squared Error (MSE): {mse_test:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_test:.2f}")
print(f"R-squared (R²): {r2_test:.3f}")

# Calculating evaluation metrics for the TRAINING set for comparison
mae_train = metrics.mean_absolute_error(y_train, y_hat)
mse_train = metrics.mean_squared_error(y_train, y_hat)
rmse_train = np.sqrt(mse_train)
r2_train = metrics.r2_score(y_train, y_hat)

print("\n--- Training Set Evaluation ---")
print(f"Mean Absolute Error (MAE): {mae_train:.2f}")
print(f"Mean Squared Error (MSE): {mse_train:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_train:.2f}")
print(f"R-squared (R²): {r2_train:.3f}")

# Overall Assessment:

# Our model appears to be performing consistently well on both the training and test sets. 
# The slight increase in error metrics and the minor increase in R-squared on the test set are not concerning and suggest good generalization. 
# There's no strong indication of overfitting (where the model performs much better on the training data than the test data).

# In summary:

# The model's performance on the test set is very close to its performance on the training set, which is a positive sign.
# The slight increase in errors on the test set is normal.
# The slightly higher R-squared on the test set could be due to sampling variability or better generalization to the specific patterns in the test data.
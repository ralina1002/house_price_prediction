from sklearn.datasets import load_boston
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Step 1: Data Collection
boston = load_boston()

# Step 2: Data Exploration
data = pd.DataFrame(boston.data, columns=boston.feature_names)
data['PRICE'] = boston.target

# Step 3: Data Cleaning
# The dataset we're using is already clean, so no cleaning tasks are performed

# Step 4: Exploratory Data Analysis
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.distplot(data['PRICE'], bins=30)
plt.show()

# Step 5: Feature Engineering
scaler = StandardScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# Step 6: Model Selection
X = data_scaled.drop('PRICE', axis=1)
y = data_scaled['PRICE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lm = LinearRegression()
lm.fit(X_train, y_train)

# Step 7: Model Evaluation
y_pred = lm.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('Root Mean Squared Error: ', np.sqrt(mse))
print('R^2 Score: ', r2)

import numpy as np

# Assuming you have a trained model called 'model'
model = ...

# Assuming you have new data for prediction called 'X_new'
X_new = ...

# Make predictions using the loaded model
predictions = model.predict(X_new)

# Calculate prediction interval
alpha = 0.74  # Confidence level (e.g., 95% confidence interval)
mse = 34592865297081.17  # MSE from model evaluation
n = X_new.shape[0]  # Number of samples

# Calculate the standard deviation of the residuals
residuals = y_actual - predictions
std_residuals = np.sqrt(np.sum(residuals ** 2) / (n - 2))

# Calculate the prediction interval
z_score = np.abs(stats.norm.ppf((1 - alpha) / 2))
prediction_interval = z_score * std_residuals

# Show the prediction interval to the user
lower_bound = predictions - prediction_interval
upper_bound = predictions + prediction_interval

for lower, upper in zip(lower_bound, upper_bound):
    print(f"Prediction interval: [{lower}, {upper}]")

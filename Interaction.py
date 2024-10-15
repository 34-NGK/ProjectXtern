import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv('cleaned_data.csv')

# Step 1: Calculate the average for each column (excluding 'User ID')
column_means = data.drop(columns=['User ID']).mean()

# Step 2: Get the top 10 users based on Virtual Merchandise Purchases (show only User ID and Purchase count)
top_10_users = data[['User ID', 'Virtual Merchandise Purchases']].nlargest(10, 'Virtual Merchandise Purchases')
print("Top 10 Users Based on Virtual Merchandise Purchases:")
print(top_10_users)

# Step 3: Compare top 10 users' values to the averages and show where they exceeded the average
columns_to_compare = ['Fan Challenges Completed', 'Predictive Accuracy (%)', 'Sponsorship Interactions (Ad Clicks)',
                      'Time on Live 360 (mins)', 'Real-Time Chat Activity (Messages Sent)']

# Compare each top 10 user to the averages, show if they are above or below for each column
comparison_result = top_10_users[['User ID']].copy()
for col in columns_to_compare:
    comparison_result[col] = data.loc[top_10_users.index, col] > column_means[col]
    comparison_result[col] = comparison_result[col].apply(lambda x: 'Above Average' if x else 'Below Average')

# Step 4: Display results where users exceeded the average
print("\nTop 10 Users and Columns Where They Exceeded the Average:")
print(comparison_result)

# Step 5: Count the number of top 10 users exceeding the average for each column
exceed_counts = (data.loc[top_10_users.index, columns_to_compare] > column_means[columns_to_compare]).sum()

# Step 6: Identify columns where more than 50% of the top 10 users exceeded the average
high_impact_columns = exceed_counts[exceed_counts > len(top_10_users) / 2].index

# Step 7: Print the columns where more than 50% of the top 10 users exceeded the average
print("\nColumns where More than 50% of Top 10 Users were Above Average:")
print(high_impact_columns)

# Step 8: Perform a regression analysis to check how each column affects Virtual Merchandise Purchases
# Define the independent variables (excluding 'User ID' and 'Virtual Merchandise Purchases') and the dependent variable
X = data[columns_to_compare]
y = data['Virtual Merchandise Purchases']

# Add a constant to the independent variables (for the intercept)
X = sm.add_constant(X)

# Perform the regression analysis
model = sm.OLS(y, X).fit()

# Print the results of the regression analysis
print("\nOLS Regression Results:")
print(model.summary())


print("I was hoping there would be a great coorelation between my average analysis and using the regression analysis to see which columns are most impactful. However, the results are not as expected.")
print("this means there is need for more  analysis to understand the relationship between the columns and the dependent variable.  I will need to perform more. ")


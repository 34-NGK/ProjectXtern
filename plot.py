import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the dataset
data = pd.read_csv('cleaned_data.csv')

# List of columns to check correlation with 'Virtual Merchandise Purchases'
columns_to_check = ['Fan Challenges Completed', 'Predictive Accuracy (%)', 
                    'Sponsorship Interactions (Ad Clicks)', 'Time on Live 360 (mins)', 
                    'Real-Time Chat Activity (Messages Sent)']

# Iterate over the columns and calculate correlation with 'Virtual Merchandise Purchases'
for column in columns_to_check:
    correlation = data[column].corr(data['Virtual Merchandise Purchases'])
    print(f'Correlation between {column} and Virtual Merchandise Purchases: {correlation}')
    
    # Plot the figure
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=column, y='Virtual Merchandise Purchases', data=data)
    plt.title(f'{column} vs Virtual Merchandise Purchases')

    # Save the plot as a PNG file
    file_name = f'{column.replace(" ", "_").lower()}_vs_purchases.png'
    plt.savefig(file_name)
    print(f"Plot saved as '{file_name}'")


print("Using Scatterplot I was able to conclude that Fan Challenges Completed and Sponsorship Interaction has the most direct correlation with Virtual Merchandise Purchases. This suggests that users who are more engaged those activities are more likely to make purchases. ")

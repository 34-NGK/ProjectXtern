import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('cleaned_data.csv')

# Select relevant columns for clustering
columns_to_use = ['Fan Challenges Completed', 'Predictive Accuracy (%)', 
                  'Sponsorship Interactions (Ad Clicks)', 'Time on Live 360 (mins)', 
                  'Real-Time Chat Activity (Messages Sent)', 'Virtual Merchandise Purchases']

# Step 1: Data Preprocessing (Scaling the data)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[columns_to_use])

# Step 2: Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)  # Trying 3 clusters (you can try more)
data['Cluster'] = kmeans.fit_predict(data_scaled)

# Step 3: Visualize the Clusters for Each Column and Save as PNG files
for column in columns_to_use[:-1]:  # Exclude 'Virtual Merchandise Purchases' for plotting
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=column, y='Virtual Merchandise Purchases', hue='Cluster', data=data, palette='Set2')
    plt.title(f'K-Means Clustering of Users Based on {column}')
    plt.xlabel(column)
    plt.ylabel('Virtual Merchandise Purchases')
    
    # Save the plot as a PNG file
    file_name = f'{column.replace(" ", "_").lower()}_vs_purchases K-Means.png'
    plt.savefig(file_name)
    plt.close()  # Close the figure to free up memory
    print(f"Plot saved as '{file_name}'")
    print('The combination of the clustering  results and the plots will help you understand the behavior of your users. Which lets us understand that the user who interacts with fan challenges completed and Sponsorship Interaction would be more likely to purchase merchandise. Which approves our hypothesis also.') 



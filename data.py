import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of users and items
num_users = 20000
num_items = 2000
num_entries = num_users * num_items

# Generate sequential user IDs from 1000 to 1099
user_ids = np.arange(1000, 1000 + num_users)

# Generate sequential item IDs from 1 to 20
item_ids = np.tile(np.arange(1, num_items + 1), num_users)

# Generate random ratings
ratings = np.random.randint(0, 6, num_entries)  # Ratings from 1 to 5, and 0 means the user hasn't rated the item.

# Create a DataFrame
data = {'user_id': np.repeat(user_ids, num_items),
        'prod_id': item_ids,
        'rating': ratings}
df = pd.DataFrame(data)

# Display the first few rows of the dataset
print(df.head())

# Save the DataFrame as a CSV file
df.to_csv('data.csv', index=False)



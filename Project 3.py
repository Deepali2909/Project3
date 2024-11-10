#!/usr/bin/env python
# coding: utf-8

# # Import Required Libraries

# In[1]:


# Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Display settings for Jupyter notebooks
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style="whitegrid")


# 
# # Load the Dataset

# In[2]:


# Load the Dataset
# Note: Replace with the file path if necessary
data = pd.read_excel(r"C:\Users\Deepkiran\OneDrive\Desktop\housing_data.csv")

# Preview the data
data.head()


# # Data Cleaning

# In[3]:


# Check for missing values
missing_values = data.isnull().sum()
missing_values = missing_values[missing_values > 0].sort_values(ascending=False)
print("Columns with missing values:", missing_values)


# In[4]:


# Check for missing values
print("Missing values in each column:")
display(data.isnull().sum())


# In[5]:


# Handle missing values by filling with the median of each column
data = data.fillna(data.median(numeric_only=True))

# Verify that there are no missing values left
print("Missing values after filling with median:")
display(data.isnull().sum())


# In[6]:


# Drop duplicate rows
data = data.drop_duplicates()

# Verify data after cleaning
data.info()


# # Exploratory Data Analysis (EDA)
# 

# # Univariate Analysis 

# Analyzing House Prices

# The histograms and correlation heatmap reveal which features have strong associations with house prices.

# In[7]:


# Display all column names in the dataset
print(data.columns)


#  Distribution of House Prices

# In[8]:


# Plot histogram and KDE for 'SalePrice' (assuming this column represents house prices)
plt.figure(figsize=(10, 6))
sns.histplot(data['SalePrice'], kde=True, bins=30)
plt.title("Distribution of House Prices")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()


# # Multivariate Analysis

# Correlation Heatmap

# In[9]:


# Calculate correlation matrix for numerical columns only
corr_matrix = data.select_dtypes(include=[np.number]).corr()

# Plot heatmap of correlations
plt.figure(figsize=(15, 12))
sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Matrix of Numerical Features")
plt.show()


# In[10]:


# Identify and display top 10 features most correlated with SalePrice
top_corr_features = corr_matrix["SalePrice"].abs().sort_values(ascending=False)[1:11]
print("Top 10 features most correlated with SalePrice:\n", top_corr_features)


# # Feature Engineering

# Create New Features: Price per Square Foot and Property Age

# New features like Price_per_SqFt and Property_Age provide deeper insights into the property’s value and allow for better comparisons.

# In[11]:


# Feature: Price per Square Foot
data['Price_per_SqFt'] = data['SalePrice'] / data['GrLivArea']

# Feature: Age of the Property (assuming 'YearBuilt' is available)
data['Property_Age'] = 2024 - data['YearBuilt']

# Check new features
data[['Price_per_SqFt', 'Property_Age']].describe()


# # Feature Engineering and Size Impact

# Analyze Impact of Bedrooms, Kitchen, Bathroom, and Square Footage on Prices

# Boxplots help determine how specific amenities influence house prices, useful for pricing and marketing.

# In[12]:


# Scatter plot of square footage vs. house price
plt.figure(figsize=(10, 6))
sns.scatterplot(x='GrLivArea', y='SalePrice', data=data)
plt.title("Square Footage vs. House Price")
plt.xlabel("Living Area (sqft)")
plt.ylabel("House Price")
plt.show()


# In[13]:


# Box plot for number of bedrooms vs. house price
plt.figure(figsize=(10, 6))
sns.boxplot(x='BedroomAbvGr', y='SalePrice', data=data)
plt.title("Number of Bedrooms vs. House Price")
plt.xlabel("Number of Bedrooms")
plt.ylabel("House Price")
plt.show()


# In[14]:


# Box plot for kitchen vs. house price
plt.figure(figsize=(10, 6))
sns.boxplot(x='KitchenAbvGr', y='SalePrice', data=data)
plt.title("Kitchen vs. House Price")
plt.xlabel("Kitchen")
plt.ylabel("House Price")
plt.show()


# In[15]:


# Box plot for number of bathroom vs. house price
plt.figure(figsize=(10, 6))
sns.boxplot(x='BsmtFullBath', y='SalePrice', data=data)
plt.title("Number of Bathroom vs. House Price")
plt.xlabel("Number of Bathroom")
plt.ylabel("House Price")
plt.show()



# # Market Trends Analysis

# Time Series Analysis

# Examine historical pricing trends, evaluate customer preferences, and segment properties based on amenity profiles.

# In[16]:


# Line plot of average sale price over years (assuming 'YrSold' is the year sold column)
yearly_price = data.groupby('YrSold')['SalePrice'].mean()
plt.figure(figsize=(10, 6))
plt.plot(yearly_price.index, yearly_price.values, marker='o')
plt.title("Average Sale Price Over Years")
plt.xlabel("Year")
plt.ylabel("Average Sale Price")
plt.show()


# # Customer Preferences Analysis

# Analyze Impact of Swimming Pool, Garden and Garage on House Prices

# If a reviews column exists, this section counts mentions of each amenity in customer reviews and visualizes their frequency in a bar plot. This helps assess the perceived value of each amenity.

# In[17]:


# Plot histogram and KDE for Swimming Pool Area(sqft)
plt.figure(figsize=(10, 6))
sns.histplot(data['PoolArea'], kde=True, bins=30)
plt.title("Swimming Pool Area in House Prices")
plt.xlabel("Swimming Pool Area(sqft)")
plt.ylabel("House Price")
plt.show()


# In[18]:


# Box plot for number of Garage Area vs. house price
plt.figure(figsize=(10, 6))
sns.histplot(x='GarageArea', y='SalePrice', data=data)
plt.title("Garage Area(sqft) vs. House Price")
plt.xlabel("Garage Area(sqft)")
plt.ylabel("House Price")
plt.show()


# In[19]:


reviews = data['SaleCondition'].str.lower()
amenity_keywords = ['Swimming Pool', 'Garage Area', 'Garden Area']
for keyword in amenity_keywords:
    count = reviews.str.contains(keyword).sum()
    print(f'Number of reviews mentioning {keyword}: {count}')


# In[20]:


amenity_counts = {keyword: reviews.str.contains(keyword).sum() for keyword in amenity_keywords}
plt.figure(figsize=(10, 6))
sns.scatterplot(x=list(amenity_counts.keys()), y=list(amenity_counts.values()))
plt.title('Frequency of Amenity Mentions in Reviews')
plt.xlabel('Amenity')
plt.ylabel('Number of Mentions')
plt.show()


# # Actionable Recommendations

# Based on the findings, provide insights and recommendations, such as:
#     
# For Pricing: 
#     Suggest pricing strategies based on high-impact features or amenities. For instance, properties with garages and gardens may be priced higher in certain segments.
# 
# For Marketing: 
#     Emphasize certain amenities in marketing materials depending on what customers value most or mention positively in reviews.
# 
# For Investment:
#     Identify features or property types with the highest growth potential based on market trends and customer preferences. This could guide investors on which types of properties to acquire or upgrade.
# 
# Customer Satisfaction:
#     Improve amenities that are frequently mentioned in reviews and align with what drives price increases, as this can increase both property value and customer satisfaction.
#         

# In[21]:


# Loop through each column in the dataset
for column in data.columns:
    plt.figure(figsize=(10, 6))
    
    # Check if the column is numerical
    if data[column].dtype in ['float64', 'int64']:
        # Plot histogram with KDE for numerical columns
        sns.histplot(data[column], kde=True)
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
    else:
        # Plot count plot for categorical columns
        sns.countplot(x=column, data=data)
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Count")
        
    plt.show()


# # Summary

# The insights gained from this analysis can provide a solid foundation for decision-making in pricing strategy, marketing, customer satisfaction improvements, and future property investments. By focusing on customer preferences, high-impact amenities, and market segments, real estate stakeholders can make data-driven decisions that align with current market demands and maximize property value.
# 

# In[ ]:





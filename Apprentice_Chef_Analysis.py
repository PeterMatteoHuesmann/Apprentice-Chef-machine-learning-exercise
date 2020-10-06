#!/usr/bin/env python
# coding: utf-8

# <br><h2>Assignment A2 - Apprentice Chef Data Analysis - Classification Assignment</h2>
# 
#     
#  
#  Author: Peter Huesmann 
#  Cohort: 1
#  Team: 1
#  Date: 06.02.2020
#  Version: 1.5
# 
# The purpose of this analysis is to determine the factors that influence cross-selling success for th Halfway There cross-selling promotion and build a model to predict cross-selling success.
# The analysis is based on the previous regression analysis.
# 
# Thinking on the problem suggests that since wine is alcoholic and adding a governmental ID is voluntary, changing this aspect of the sign up process might make this promotion easier. Since meals cost a maximum of 23 dollars and non alcoholic drinks cost a maximum of 5 dollars all average meal spendings over 28 dollars include alcohol and might therefore be interested in the promotion. We will also need to look for people that order meals during the week and not just on weekends, possibly based on the number of total meals. We will implement feature engineering for various features to find correlation.

# In[202]:


################################################################################
# Import Packages
################################################################################

import pandas as pd                                   # data science essentials
import matplotlib.pyplot as plt                       # essential graphical output
import seaborn as sns                                 # enhanced graphical output
import statsmodels.formula.api as smf                 # regression modeling
import numpy as np                                    # Math
from sklearn.model_selection import train_test_split  # train/test split
from sklearn.neighbors import KNeighborsClassifier    # KNN for classification
from sklearn.preprocessing import StandardScaler      # standard scaler
from sklearn.model_selection import GridSearchCV      # Hypertuning
from sklearn.ensemble import RandomForestClassifier   # random forest
from sklearn.metrics import confusion_matrix          # confusion matrix
from sklearn.metrics import roc_auc_score             # auc score
from sklearn.linear_model import LogisticRegression   # logistic regression
from sklearn.tree import DecisionTreeClassifier       # classification trees
from sklearn.metrics import make_scorer               # customizable scorer
from sklearn.ensemble import GradientBoostingClassifier #gbm
from sklearn.neighbors import KNeighborsRegressor    # KNN for regression
from sklearn.externals.six import StringIO           # saves objects in memory
from IPython.display import Image                    # displays on frontend
from sklearn.model_selection import GridSearchCV     # hyperparameter tuning
from sklearn.metrics import make_scorer              # customizable scorer

# setting pandas print options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# In[92]:


################################################################################
# Load Data
################################################################################

file = pd.read_excel('Apprentice_Chef_Dataset.xlsx')
original_df = file


# <h4>Data exploration - Assumptions on variable types</h4>
# Looking at the different variables in the original dataset we can find only 3 continuous variables, a number of variables that could be either binary or categorical depending on interpretation and a large number of count variables. We also have 3 discrete variables that need to be removed for the final models.

# In[ ]:


################################################################################
# Assumptions on variable type
################################################################################

#original_df.info() 

""" 
#CONTINUOUS OR INTERVAL OR ORDINAL
REVENUE
AVG_TIME_PER_SITE_VISIT
AVG_PREP_VID_TIME

# BINARY
CROSS_SELL_SUCCESS
MOBILE_NUMBER
TASTES_AND_PREFERENCES
PACKAGE_LOCKER
REFRIGERATED_LOCKER
MASTER_CLASSES_ATTENDED
MEDIAN_MEAL_RATING

# COUNT
UNIQUE_MEALS_PURCH
CONTACTS_W_CUSTOMER_SERVICE
MOBILE_LOGINS
PC_LOGINS
EARLY_DELIVERIES
LATE_DELIVERIES
FOLLOWED_RECOMMENDATIONS_PCT
LARGEST_ORDER_SIZE
AVG_CLICKS_PER_VISIT
MEDIAN_MEAL_RATING?
MASTER_CLASSES_ATTENDED?

# CATEGORICAL
TOTAL_MEALS_ORDERED
PRODUCT_CATEGORIES_VIEWED
CANCELLATIONS_BEFORE_NOON
CANCELLATIONS_AFTER_NOON
WEEKLY_PLAN
TOTAL_PHOTOS_VIEWED
m_FAMILY_NAME
MEDIAN_MEAL_RATING

# DISCRETE
Name 
Email
First Name
Family Name

"""


# In[407]:


################################################################################
# Data exploration Categorical variable counts
################################################################################


print(f"""
NAME 
-------------
{original_df['NAME'].value_counts().sort_index()}

EMAIL
-------------
{original_df['EMAIL'].value_counts().sort_index()}
     
FIRST_NAME
-------------
{original_df['FIRST_NAME'].value_counts().sort_index()}
          
FAMILY_NAME
-------------
{original_df['FAMILY_NAME'].value_counts().sort_index()}
          
MEDIAN_MEAL_RATING
-------------
{original_df['MEDIAN_MEAL_RATING'].value_counts().sort_index()}

CONTACTS_W_CUSTOMER_SERVICE
-------------
{original_df['CONTACTS_W_CUSTOMER_SERVICE'].value_counts().sort_index()}

CANCELLATIONS_BEFORE_NOON
-------------
{original_df['CANCELLATIONS_BEFORE_NOON'].value_counts().sort_index()}

CANCELLATIONS_AFTER_NOON
-------------
{original_df['CANCELLATIONS_AFTER_NOON'].value_counts().sort_index()}

EARLY_DELIVERIES
-------------
{original_df['EARLY_DELIVERIES'].value_counts().sort_index()}

LATE_DELIVERIES
-------------
{original_df['LATE_DELIVERIES'].value_counts().sort_index()}

AVG_CLICKS_PER_VISIT
-------------
{original_df['AVG_CLICKS_PER_VISIT'].value_counts().sort_index()}
""")

# Name, First Name and Family name are to diverse to provide much value. They are discrete
# There should be no correlation between people with the same first name 
# Median_MeaL_Rating can be used
# EMAIL needs to be transformed into only the provider and diveded into junk, personal and prfessional
# contacts with customer service has very few extreme outlyers - transform into percentage or group 
# CANCELLATIONS_BEFORE_NOON has very few extreme outlyers - transform into percentage or group
# CANCELLATIONS_AFTER_NOON - very high null value - make binary?
# Early deliveries - high null value - make binary or percentage? 
# Late deliveries - very diverse but clear trend - make a percentage or group
# Average Clicks per visit can be grouped together


# <h4>Data exploration - Visual exploration</h4>
# Visual exploration through Histograms and Scatterplots.
# Results are explained in the outlier flags
# 

# In[113]:


################################################################################
# Visual EDA Histograms - commented out 
################################################################################
"""
l = original_df.columns.drop(['NAME','EMAIL','FIRST_NAME','FAMILY_NAME'])

for x in l:
    
    fig, ax = plt.subplots(figsize = (18, 18))
    plt.subplot(2, 2, 1)
    sns.distplot(original_df[x],
             bins  = 'fd',
             color = 'g')
    plt.xlabel(f'{x}')
"""


# In[112]:


##############################################################################
# Visual EDA Scatterplots - commented out
################################################################################
"""
l = original_df.columns.drop(['NAME','EMAIL','FIRST_NAME','FAMILY_NAME'])

for x in l:
    
    fig, ax = plt.subplots(figsize = (18, 18))
    plt.subplot(2, 2, 1)
    sns.scatterplot(x = original_df[x],
                y = original_df['REVENUE'],
                color = 'g')
    plt.xlabel(f'{x}')
"""


# In[408]:


################################################################################
# Data exploration detailed
################################################################################
original_df.loc[:, :].quantile([0.1,
                            0.20,
                            0.3,
                            0.40,
                            0.5,
                            0.60,
                            0.7,
                            0.80,
                            0.9,
                            1.00])

# The 90-100 percentile has a massive revenue growth
# Meal ratings are very average
# cross sell success is  high over 60%


# <h2>Feature Engineering</h2>

# In[93]:


################################################################################
# Feature Engineering Missing value flag and imputing Family Name
################################################################################

# looping over columns with missing values
for col in original_df:

    # creating columns with 1s if missing and 0 if not
    if original_df[col].isnull().astype(int).sum() > 0:
        original_df['m_'+col] = original_df[col].isnull().astype(int)

# Creating a missing value fill for FAMILY_NAME      
# We impute missing since we do not have any information on potential Family names
fill = 'Missing'

# Imputing 'FAMILY_NAME'
original_df['FAMILY_NAME'] = original_df['FAMILY_NAME'].fillna(fill)


# <h4>Feature Engineering - E-Mail group</h4>
# Based on the junk, professional and personal groupings in the assignment data, 
# we create a new grouping to see if the grouping has any influence on the 
# cross sell success.

# In[94]:


################################################################################
# Feature Engineering  Creating Email grouping based on assignment data
################################################################################


# Creating a new data frame with split value columns  for EMAIL
new = original_df["EMAIL"].str.split("@", n = 1, expand = True) 
  
# Making separate first name column from new data frame 
original_df["EMAIL_PROVIDER"]= new[1] 

# Creating Groups for each category based on case information
professional = [
"mmm.com", "amex.com" ,"apple.com"  ,"boeing.com","caterpillar.com","chevron.com",
"cisco.com" ,"cocacola.com" ,"disney.com" ,"dupont.com"  ,"exxon.com","ge.org","goldmansacs.com",
"homedepot.com","ibm.com" ,"intel.com" ,"jnj.com" ,"jpmorgan.com" ,"mcdonalds.com" ,"merck.com" ,
"microsoft.com","nike.com","pfizer.com" ,"pg.com" ,"travelers.com" ,"unitedtech.com"  ,
"unitedhealth.com" ,"verizon.com" ,"visa.com" ,"walmart.com"
]
junk = [
"me.com","aol.com", "hotmail.com", "live.com", "msn.com", "passport.com"
]
personal = [
"gmail.com", "yahoo.com", "protonmail.com"
]


placeholder_lst = []
# looping to group observations by domain type

for domain in original_df["EMAIL_PROVIDER"]:
        if  domain in personal:
            placeholder_lst.append('personal')
            
        elif domain in professional:
            placeholder_lst.append('professional')
        
        elif domain in junk:
            placeholder_lst.append('junk')
        else:
            print('error')
        
# Creating new EMAIL_GROUP column
df = pd.DataFrame(placeholder_lst)
df = pd.get_dummies(df,prefix=['EMAIL_GROUP'])
                                   

# Concatenating with original DataFrame
original_df['EMAIL_GROUP'] = pd.Series(placeholder_lst)
original_df['EMAIL_GROUP_JUNK'] = df['EMAIL_GROUP_junk']
original_df['EMAIL_GROUP_PROFESSIONAL'] = df['EMAIL_GROUP_professional']
original_df['EMAIL_GROUP_PERSONAL'] = df['EMAIL_GROUP_personal']


# In[95]:


################################################################################
# Feature Engineering and One hot encode Median Meal Rating
################################################################################


# One hot encoding MEDIAN_MEAL_RATING
one_hot_RATING       = pd.get_dummies(original_df['MEDIAN_MEAL_RATING'])



# Joining codings together
original_df = original_df.join([one_hot_RATING])
original_df.columns = ['REVENUE', 'CROSS_SELL_SUCCESS', 'NAME', 'EMAIL', 'FIRST_NAME', 'FAMILY_NAME', 'TOTAL_MEALS_ORDERED', 'UNIQUE_MEALS_PURCH', 'CONTACTS_W_CUSTOMER_SERVICE', 'PRODUCT_CATEGORIES_VIEWED', 'AVG_TIME_PER_SITE_VISIT', 'MOBILE_NUMBER', 'CANCELLATIONS_BEFORE_NOON', 'CANCELLATIONS_AFTER_NOON', 'TASTES_AND_PREFERENCES', 'MOBILE_LOGINS', 'PC_LOGINS', 'WEEKLY_PLAN', 'EARLY_DELIVERIES', 'LATE_DELIVERIES', 'PACKAGE_LOCKER', 'REFRIGERATED_LOCKER', 'FOLLOWED_RECOMMENDATIONS_PCT', 'AVG_PREP_VID_TIME', 'LARGEST_ORDER_SIZE', 'MASTER_CLASSES_ATTENDED', 'MEDIAN_MEAL_RATING', 'AVG_CLICKS_PER_VISIT', 'TOTAL_PHOTOS_VIEWED', 'm_FAMILY_NAME', 'EMAIL_PROVIDER', 'EMAIL_GROUP', 'EMAIL_GROUP_JUNK', 'EMAIL_GROUP_PROFESSIONAL', 'EMAIL_GROUP_PERSONAL', "MEDIAN_MEAL_RATING_1", "MEDIAN_MEAL_RATING_2", "MEDIAN_MEAL_RATING_3", "MEDIAN_MEAL_RATING_4", "MEDIAN_MEAL_RATING_5"]


# <h4>Feature Engineering - E-Mail group</h4>
# Master classes attended has a very high null value and very few values above 1. 
# It makes sense to encode it as a binary attended or not attended category.

# In[96]:


################################################################################
# Feature Engineering Creating Master classes attended binary group
################################################################################


# Creating Master classes attended binary grouping
yes =[1,2,3]
no =[0]
placeholder_lst = []

# looping to group observations master classes attendence
for domain in original_df["MASTER_CLASSES_ATTENDED"]:
        if  domain in yes:
            placeholder_lst.append('Yes')
            
        elif domain in no:
            placeholder_lst.append('No')
        else:
            print('error')
        
df = pd.DataFrame(placeholder_lst)
df = pd.get_dummies(df,prefix=["MASTER_CLASSES_ATTENDED"])

# Concatenating with original DataFrame
original_df["MASTER_CLASSES_ATTENDED_YES"] = df['MASTER_CLASSES_ATTENDED_Yes']
original_df["MASTER_CLASSES_ATTENDED_NO"] = df['MASTER_CLASSES_ATTENDED_No']


# <h4>Feature Engineering - Engineered columns</h4>
# Description in code for each variable

# In[97]:


################################################################################
# Feature Engineering Customer Service Contact
################################################################################
original_df['CUSTOMER_SERVICE_CONTACT_PER_MEAL'] = original_df['CONTACTS_W_CUSTOMER_SERVICE']/original_df['TOTAL_MEALS_ORDERED']
#This column is created to determine the relationship between the number of 
#orders and customer service contact to see if someone has overly much 
#contact with customer service compared with others based on the number of 
#orders


# In[98]:


################################################################################
# Feature Engineering  Average Spending 
################################################################################

#Creating a variable for High and low spend based on case information
original_df['AVERAGE_SPEND'] = original_df['REVENUE']/original_df['TOTAL_MEALS_ORDERED']

#Used to determine who buys drinks / alcoholic drinks / expensive meals


# In[99]:


################################################################################
# Feature Engineering Early Delivery Percentage
################################################################################

original_df['AVERAGE_EARLY_DELIVERIES'] = original_df['EARLY_DELIVERIES']/original_df['TOTAL_MEALS_ORDERED']

#used to determine the number of early deliveries by total meals
#it shows if someone got his meals early more often than others


# In[100]:


################################################################################
# Feature Engineering late Delivery Percentage
################################################################################

original_df['AVERAGE_LATE_DELIVERIES'] = original_df['LATE_DELIVERIES']/original_df['TOTAL_MEALS_ORDERED']

#Same as early deliveries


# In[101]:


################################################################################
# Feature Engineering Cancellations Before Noon Percentage
################################################################################

original_df['AVERAGE_CANCELLATIONS_BEFORE_NOON'] = original_df['CANCELLATIONS_BEFORE_NOON']/original_df['TOTAL_MEALS_ORDERED']

#used to determine who canceles before noon more often than others based on the 
#total number of meals as a comparison


# In[102]:


################################################################################
# Feature Engineering Average clicks per visit by total meals 
################################################################################

original_df['AVERAGE_AVG_CLICKS_PER_VISIT'] = original_df['AVG_CLICKS_PER_VISIT']/original_df['TOTAL_MEALS_ORDERED']

#used to determine if people that buy more meals take more or less clicks on the website


# In[103]:


################################################################################
# Feature Engineering Average clicks per visit by time spent on website
################################################################################

original_df['AVERAGE_AVG_CLICKS_PER_VISIT'] = original_df['AVG_CLICKS_PER_VISIT']/original_df['AVG_TIME_PER_SITE_VISIT']

#used to determine if people that spent more time on the website interact with it or not


# In[104]:


################################################################################
# Feature Engineering Cancelations after noon percentage
################################################################################

original_df['AVERAGE_CANCELLATIONS_AFTER_NOON'] = original_df['CANCELLATIONS_AFTER_NOON']/original_df['TOTAL_MEALS_ORDERED']

#similar to cancelations before noon


# <h4>Feature Engineering - Cancellations after noon binary</h4>
# This group has a very high null value and very few values above 1. 
# It makes sense to encode it as a binary attended or not attended category.
# It is used as an alternative to the non binary group to see if it makes a 
# difference

# In[105]:


################################################################################
# Feature Engineering Creating cancellations after noon binary group
################################################################################


# Cancelations after noon binary grouping
yes =[1,2,3]
no =[0]
placeholder_lst = []

# looping to group observations for cancellations after noon
for domain in original_df["CANCELLATIONS_AFTER_NOON"]:
        if  domain in yes:
            placeholder_lst.append('Yes')
            
        elif domain in no:
            placeholder_lst.append('No')
        else:
            print('error')
        
df = pd.DataFrame(placeholder_lst)
df = pd.get_dummies(df,prefix=["CANCELLATIONS_AFTER_NOON"])

# Concatenating with original DataFrame
original_df["CANCELLATIONS_AFTER_NOON_YES"] = df['CANCELLATIONS_AFTER_NOON_Yes']
original_df["CANCELLATIONS_AFTER_NOON_NO"] = df['CANCELLATIONS_AFTER_NOON_No']


# <h4>Feature Engineering - Cancellations before noon binary</h4>
# This group has a very high null value and very scattered values above 4. 
# It is used as an alternative to the non binary group to see if it makes a 
# difference

# In[106]:


################################################################################
# Feature Engineering Creating cancellations before noon group
###############################################################################

# Cancelations before noon grouping
fourormore =[4,5,6,7,8,9,10,11,12,13]
three =[3]
two =[2]
one =[1]
zero =[0]
placeholder_lst = []

# looping to group observations for cancellations before noon
for domain in original_df["CANCELLATIONS_BEFORE_NOON"]:
        if  domain in one:
            placeholder_lst.append('1')
        elif domain in two:
            placeholder_lst.append('2')
        elif domain in three:
            placeholder_lst.append('3')
        elif domain in fourormore:
            placeholder_lst.append('4_or_more')
        elif domain in zero:
            placeholder_lst.append('0')
        else:
            print('error')
        
df = pd.DataFrame(placeholder_lst)
df = pd.get_dummies(df,prefix=["CANCELLATIONS_BEFORE_NOON"])

# Concatenating with original DataFrame
original_df["CANCELLATIONS_BEFORE_NOON_ONE"] = df['CANCELLATIONS_BEFORE_NOON_1']
original_df["CANCELLATIONS_BEFORE_NOON_TWO"] = df['CANCELLATIONS_BEFORE_NOON_2']
original_df["CANCELLATIONS_BEFORE_NOON_THREE"] = df['CANCELLATIONS_BEFORE_NOON_3']
original_df["CANCELLATIONS_BEFORE_NOON_FOUR_OR_MORE"] = df['CANCELLATIONS_BEFORE_NOON_4_or_more']
original_df["CANCELLATIONS_BEFORE_NOON_ZERO"] = df['CANCELLATIONS_BEFORE_NOON_0']


# In[107]:


################################################################################
# Feature Engineering Creating Number of recommended meals
################################################################################


original_df['NUMBER_OF_RECOMMENDED_MEALS'] = (original_df['FOLLOWED_RECOMMENDATIONS_PCT']/100)*original_df['TOTAL_MEALS_ORDERED']

#used to determine if the total number of recommended meals is more or 
#less important than the percentage. Is it more influential if someone 
#ordered 10 meals and 9 where recommended or ordered 200 and 30 where recommended


# In[108]:


################################################################################
# Feature Engineering Creating Percentage of time spent on videos
################################################################################


original_df['PERCENTAGE_OF_TIME_SPENT_ON_VIDEOS'] = original_df['AVG_PREP_VID_TIME']/original_df['AVG_TIME_PER_SITE_VISIT']

#used to determine how much of their time is spent on videos and if 
#we should focus our marketing here 
#if most tiem is spent here


# In[109]:


################################################################################
# Feature Engineering Creating Weekly Plan by Revenue
################################################################################

original_df['WEEKLY_PLAN_BY_REVNUE'] = original_df['WEEKLY_PLAN']/original_df['REVENUE']

#I honestly just wanted to see if something happened but nothing did


# In[110]:


################################################################################
# Feature Engineering Creating Weekly Plan by total meals ordered
################################################################################

original_df['WEEKLY_PLAN_BY_TOTAL_MEALS'] = original_df['WEEKLY_PLAN']/original_df['TOTAL_MEALS_ORDERED']

#used to determine the percentage of meals ordered through weekly plans


# <h4>Feature Engineering - Outliers</h4>
# Outliers are based on the visual data exploration from before.
# The reasons for each flag are described in short after the flag definitions.
# Please compare them with the visual data for more insight.

# In[114]:


################################################################################
# Feature Engineering Outlier flags
total_meals_ordered_low            = 15  # Spike in graph, trend change
total_meals_ordered_high           = 150 # More spread from here onwards
total_meals_ordered_change         = 270 # Data becomes less frequent and more scattered
unique_meals_purch_low             = 9   # Dramatic drop in frequency
unique_meals_purch_change          = 1   # Highpoint at 0, data changes after
contacts_w_customer_service_low    = 3   # Very few datapoints below 3
contacts_w_customer_service_high   = 12  # After 12 dramatic drop in frequency
contacts_w_customer_service_change = 9   # Strange bump from 10 to 12
avg_time_per_site_visit_low        = 25  # Spike in graph
avg_time_per_site_visit_high       = 250 # Very low frequency 
avg_time_per_site_visit_change     = 100 # Spike in graph
cancellations_before_noon_low      = 0   # Spike in graph
cancellations_before_noon_high     = 6   # Almost exponential until here
cancellations_before_noon_change   = 8   # Very few datapoints after 8, very scattered
cancellations_after_noon_low_high  = 1   # Spike at one
mobile_logins_low                  = 4   # Very few datapoints before 5
mobile_logins_high                 = 6   # Very few datapoints after 6
pc_logins_low                      = 1   # Spike in graph
pc_logins_high                     = 2   # Spike in graph
weekly_plan_low                    = 1   # Spike at 0, Many people never try weekly plan
weekly_plan_high                   = 13  # Very scattered data
early_deliveries_low               = 1   # Spike at 0, does not happen often
late_deliveries_high               = 9   # Until here almost exponential
late_deliveries_change             = 12  # Strange spike at 13 after that very scattered
followed_recommendations_pct_high  = 80  # Very scattered data aboe 80, very weird shape in general
avg_prep_vid_time_low              = 80  # Very few datapoints below 80
avg_prep_vid_time_high             = 290 # Very scattered data
largest_order_size_low             = 2   # Very few people order only one meal
largest_order_size_high            = 8   # Very few datapoints beyond 8
largest_order_size_change          = 5   # Data becomes more scattered
avg_clicks_per_visit_low           = 8   # Very few datapoints below 8
avg_clicks_per_visit_high          = 17  # very few datapoints above 18
avg_clicks_per_visit_change        = 10  # Trend change, goes down
total_photos_viewed_low            = 1   # Spike in graph
total_photos_viewed_high           = 335 # Data scatters, 90% percential
average_spend_high                 = 23  # Maximum price for meals, beverages above
average_spend_low                  = 15  # Minimum order prices
average_spend_alcohol              = 28 # Maximum meal price 23 plus maximum non alcoholic drink price 5
revenue_change                     = 2000 # Change in graph
revenue_high                       = 4000 # Very scattered data   
customer_service_contact_per_meal  = 0.25 # Change in data
number_of_recommended_meals        = 60   # Change in data
percentage_of_time_spent_on_video  = 4  # Change in distribution


# <h4>Feature Engineering - Outlier creation</h4>
# Creation of outlier columns. The same code with different variable repeated.
# Outlier points are softcoded in case of changes

# In[124]:


##############################################################################
# Feature Engineering Create Outlier columns                          
##############################################################################

# Total Meals Ordered outlier columns
original_df['OUT_TOTAL_MEALS_ORDERED'] = 0

condition_high = original_df.loc[0:,'OUT_TOTAL_MEALS_ORDERED'][original_df['TOTAL_MEALS_ORDERED'] > total_meals_ordered_high]
condition_low = original_df.loc[0:,'OUT_TOTAL_MEALS_ORDERED'][original_df['TOTAL_MEALS_ORDERED'] < total_meals_ordered_low]

original_df['OUT_TOTAL_MEALS_ORDERED'].replace(to_replace = condition_high,
                                value      = 1,
                                inplace    = True)

original_df['OUT_TOTAL_MEALS_ORDERED'].replace(to_replace = condition_low,
                                value      = 1,
                                inplace    = True)

original_df['CHANGE_TOTAL_MEALS_ORDERED'] = 0
condition_change = original_df.loc[0:,'CHANGE_TOTAL_MEALS_ORDERED'][original_df['TOTAL_MEALS_ORDERED'] > total_meals_ordered_change]

original_df['CHANGE_TOTAL_MEALS_ORDERED'].replace(to_replace = condition_change,
                                   value      = 1,
                                   inplace    = True)


# Unique Meals Purchased outlier columns
original_df['OUT_UNIQUE_MEALS_PURCH'] = 0

condition_low = original_df.loc[0:,'OUT_UNIQUE_MEALS_PURCH'][original_df['UNIQUE_MEALS_PURCH'] > unique_meals_purch_low]

original_df['OUT_UNIQUE_MEALS_PURCH'].replace(to_replace = condition_low,
                                value      = 1,
                                inplace    = True)

original_df['CHANGE_UNIQUE_MEALS_PURCH'] = 0

condition = original_df.loc[0:,'CHANGE_UNIQUE_MEALS_PURCH'][original_df['UNIQUE_MEALS_PURCH'] == unique_meals_purch_change ]

original_df['CHANGE_UNIQUE_MEALS_PURCH'].replace(to_replace = condition,
                                       value      = 1,
                                       inplace    = True)

# Contacts with Customer Service outlier columns
original_df['OUT_CONTACTS_W_CUSTOMER_SERVICE'] = 0

condition_high = original_df.loc[0:,'OUT_CONTACTS_W_CUSTOMER_SERVICE'][original_df['CONTACTS_W_CUSTOMER_SERVICE'] > contacts_w_customer_service_high]
condition_low = original_df.loc[0:,'OUT_CONTACTS_W_CUSTOMER_SERVICE'][original_df['CONTACTS_W_CUSTOMER_SERVICE'] < contacts_w_customer_service_low]

original_df['OUT_CONTACTS_W_CUSTOMER_SERVICE'].replace(to_replace = condition_high,
                                value      = 1,
                                inplace    = True)
original_df['OUT_CONTACTS_W_CUSTOMER_SERVICE'].replace(to_replace = condition_low,
                                value      = 1,
                                inplace    = True)

original_df['CHANGE_CONTACTS_W_CUSTOMER_SERVICE'] = 0

condition = original_df.loc[0:,'CHANGE_CONTACTS_W_CUSTOMER_SERVICE'][original_df['CONTACTS_W_CUSTOMER_SERVICE'] > contacts_w_customer_service_change]

original_df['CHANGE_CONTACTS_W_CUSTOMER_SERVICE'].replace(to_replace = condition,
                                   value      = 1,
                                   inplace    = True)

# Average Time per Site Visit outlier columns
original_df['OUT_AVG_TIME_PER_SITE_VISIT'] = 0

condition_high = original_df.loc[0:,'OUT_AVG_TIME_PER_SITE_VISIT'][original_df['AVG_TIME_PER_SITE_VISIT'] > avg_time_per_site_visit_high]
condition_low = original_df.loc[0:,'OUT_AVG_TIME_PER_SITE_VISIT'][original_df['AVG_TIME_PER_SITE_VISIT'] < avg_time_per_site_visit_low]

original_df['OUT_AVG_TIME_PER_SITE_VISIT'].replace(to_replace = condition_high,
                                value      = 1,
                                inplace    = True)

original_df['OUT_AVG_TIME_PER_SITE_VISIT'].replace(to_replace = condition_low,
                                value      = 1,
                                inplace    = True)

original_df['CHANGE_AVG_TIME_PER_SITE_VISIT'] = 0

condition = original_df.loc[0:,'CHANGE_AVG_TIME_PER_SITE_VISIT'][original_df['AVG_TIME_PER_SITE_VISIT'] < avg_time_per_site_visit_change]

original_df['CHANGE_AVG_TIME_PER_SITE_VISIT'].replace(to_replace = condition,
                                      value      = 1,
                                      inplace    = True)

# Cancelations Before Noon outlier columns
original_df['OUT_CANCELLATIONS_BEFORE_NOON'] = 0

condition_high = original_df.loc[0:,'OUT_CANCELLATIONS_BEFORE_NOON'][original_df['CANCELLATIONS_BEFORE_NOON'] > cancellations_before_noon_high]
condition_low = original_df.loc[0:,'OUT_CANCELLATIONS_BEFORE_NOON'][original_df['CANCELLATIONS_BEFORE_NOON'] < cancellations_before_noon_low]

original_df['OUT_CANCELLATIONS_BEFORE_NOON'].replace(to_replace = condition_high,
                                value      = 1,
                                inplace    = True)

original_df['OUT_CANCELLATIONS_BEFORE_NOON'].replace(to_replace = condition_low,
                                value      = 1,
                                inplace    = True)

original_df['CHANGE_CANCELLATIONS_BEFORE_NOON'] = 0

condition = original_df.loc[0:,'CHANGE_CANCELLATIONS_BEFORE_NOON'][original_df['CANCELLATIONS_BEFORE_NOON'] > cancellations_before_noon_change]

original_df['CHANGE_CANCELLATIONS_BEFORE_NOON'].replace(to_replace = condition,
                                   value      = 1,
                                   inplace    = True)


# Cancelations After Noon outlier columns
original_df['OUT_CANCELLATIONS_AFTER_NOON'] = 0

condition_high = original_df.loc[0:,'OUT_CANCELLATIONS_AFTER_NOON'][original_df['CANCELLATIONS_AFTER_NOON'] > cancellations_after_noon_low_high]
condition_low = original_df.loc[0:,'OUT_CANCELLATIONS_AFTER_NOON'][original_df['CANCELLATIONS_AFTER_NOON'] < cancellations_after_noon_low_high]

original_df['OUT_CANCELLATIONS_AFTER_NOON'].replace(to_replace = condition_high,
                                value      = 1,
                                inplace    = True)

original_df['OUT_CANCELLATIONS_AFTER_NOON'].replace(to_replace = condition_low,
                                value      = 1,
                                inplace    = True)

# Mobile Logins outlier columns
original_df['OUT_MOBILE_LOGINS'] = 0

condition_high = original_df.loc[0:,'OUT_MOBILE_LOGINS'][original_df['MOBILE_LOGINS'] > mobile_logins_high]
condition_low = original_df.loc[0:,'OUT_MOBILE_LOGINS'][original_df['MOBILE_LOGINS'] < mobile_logins_low]

original_df['OUT_MOBILE_LOGINS'].replace(to_replace = condition_high,
                                value      = 1,
                                inplace    = True)

original_df['OUT_MOBILE_LOGINS'].replace(to_replace = condition_low,
                                value      = 1,
                                inplace    = True)

# PC Logins outlier columns
original_df['OUT_PC_LOGINS'] = 0

condition_high = original_df.loc[0:,'OUT_PC_LOGINS'][original_df['PC_LOGINS'] > pc_logins_high]
condition_low = original_df.loc[0:,'OUT_PC_LOGINS'][original_df['PC_LOGINS'] < pc_logins_low]

original_df['OUT_PC_LOGINS'].replace(to_replace = condition_high,
                                value      = 1,
                                inplace    = True)

original_df['OUT_PC_LOGINS'].replace(to_replace = condition_low,
                                value      = 1,
                                inplace    = True)

# Weekly Plan outlier columns
original_df['OUT_WEEKLY_PLAN'] = 0

condition_high = original_df.loc[0:,'OUT_WEEKLY_PLAN'][original_df['WEEKLY_PLAN'] > weekly_plan_high]
condition_low = original_df.loc[0:,'OUT_WEEKLY_PLAN'][original_df['WEEKLY_PLAN'] < weekly_plan_low]

original_df['OUT_WEEKLY_PLAN'].replace(to_replace = condition_high,
                                value      = 1,
                                inplace    = True)

original_df['OUT_WEEKLY_PLAN'].replace(to_replace = condition_low,
                                value      = 1,
                                inplace    = True)

# Early Deliveries outlier columns
original_df['OUT_EARLY_DELIVERIES'] = 0

condition_low = original_df.loc[0:,'OUT_EARLY_DELIVERIES'][original_df['EARLY_DELIVERIES'] < early_deliveries_low]

original_df['OUT_EARLY_DELIVERIES'].replace(to_replace = condition_low,
                                value      = 1,
                                inplace    = True)

# Late Deliveries outlier columns
original_df['OUT_LATE_DELIVERIES'] = 0

condition_high = original_df.loc[0:,'OUT_LATE_DELIVERIES'][original_df['LATE_DELIVERIES'] > late_deliveries_high]

original_df['OUT_LATE_DELIVERIES'].replace(to_replace = condition_high,
                                value      = 1,
                                inplace    = True)

original_df['CHANGE_LATE_DELIVERIES'] = 0

condition = original_df.loc[0:,'CHANGE_LATE_DELIVERIES'][original_df['LATE_DELIVERIES'] > late_deliveries_change ]

original_df['CHANGE_LATE_DELIVERIES'].replace(to_replace = condition,
                                   value      = 1,
                                   inplace    = True)

# Followed Recommendations Percentage outlier columns
original_df['OUT_FOLLOWED_RECOMMENDATIONS_PCT'] = 0

condition_high = original_df.loc[0:,'OUT_FOLLOWED_RECOMMENDATIONS_PCT'][original_df['FOLLOWED_RECOMMENDATIONS_PCT'] > followed_recommendations_pct_high]

original_df['OUT_FOLLOWED_RECOMMENDATIONS_PCT'].replace(to_replace = condition_high,
                                value      = 1,
                                inplace    = True)

# Average Preperation Video Time outlier columns
original_df['OUT_AVG_PREP_VID_TIME'] = 0

condition_high = original_df.loc[0:,'OUT_AVG_PREP_VID_TIME'][original_df['AVG_PREP_VID_TIME'] > avg_prep_vid_time_high]
condition_low = original_df.loc[0:,'OUT_AVG_PREP_VID_TIME'][original_df['AVG_PREP_VID_TIME'] < avg_prep_vid_time_low]

original_df['OUT_AVG_PREP_VID_TIME'].replace(to_replace = condition_high,
                                value      = 1,
                                inplace    = True)

original_df['OUT_AVG_PREP_VID_TIME'].replace(to_replace = condition_low,
                                value      = 1,
                                inplace    = True)

# Largest Order Size outlier columns
original_df['OUT_LARGEST_ORDER_SIZE'] = 0

condition_high = original_df.loc[0:,'OUT_LARGEST_ORDER_SIZE'][original_df['LARGEST_ORDER_SIZE'] > largest_order_size_high]
condition_low = original_df.loc[0:,'OUT_LARGEST_ORDER_SIZE'][original_df['LARGEST_ORDER_SIZE'] < largest_order_size_low]

original_df['OUT_LARGEST_ORDER_SIZE'].replace(to_replace = condition_high,
                                value      = 1,
                                inplace    = True)

original_df['OUT_LARGEST_ORDER_SIZE'].replace(to_replace = condition_low,
                                value      = 1,
                                inplace    = True)

original_df['CHANGE_LARGEST_ORDER_SIZE'] = 0

condition = original_df.loc[0:,'CHANGE_LARGEST_ORDER_SIZE'][original_df['LARGEST_ORDER_SIZE'] > largest_order_size_change]

original_df['CHANGE_LARGEST_ORDER_SIZE'].replace(to_replace = condition,
                                   value      = 1,
                                   inplace    = True)

# Average Clicks per Visit outlier columns
original_df['OUT_AVG_CLICKS_PER_VISIT'] = 0

condition_high = original_df.loc[0:,'OUT_AVG_CLICKS_PER_VISIT'][original_df['AVG_CLICKS_PER_VISIT'] > avg_clicks_per_visit_high]
condition_low = original_df.loc[0:,'OUT_AVG_CLICKS_PER_VISIT'][original_df['AVG_CLICKS_PER_VISIT'] < avg_clicks_per_visit_low]

original_df['OUT_AVG_CLICKS_PER_VISIT'].replace(to_replace = condition_high,
                                value      = 1,
                                inplace    = True)
original_df['OUT_AVG_CLICKS_PER_VISIT'].replace(to_replace = condition_low,
                                value      = 1,
                                inplace    = True)

original_df['CHANGE_AVG_CLICKS_PER_VISIT'] = 0

condition = original_df.loc[0:,'CHANGE_AVG_CLICKS_PER_VISIT'][original_df['AVG_CLICKS_PER_VISIT'] > avg_clicks_per_visit_change]

original_df['CHANGE_AVG_CLICKS_PER_VISIT'].replace(to_replace = condition,
                                   value      = 1,
                                   inplace    = True)
# Total Photos viewed outlier columns
original_df['OUT_TOTAL_PHOTOS_VIEWED'] = 0

condition_high = original_df.loc[0:,'OUT_TOTAL_PHOTOS_VIEWED'][original_df['TOTAL_PHOTOS_VIEWED'] > total_photos_viewed_high]
condition_low = original_df.loc[0:,'OUT_TOTAL_PHOTOS_VIEWED'][original_df['TOTAL_PHOTOS_VIEWED'] < total_photos_viewed_low]

original_df['OUT_TOTAL_PHOTOS_VIEWED'].replace(to_replace = condition_high,
                                value      = 1,
                                inplace    = True)

original_df['OUT_TOTAL_PHOTOS_VIEWED'].replace(to_replace = condition_low,
                                value      = 1,
                                inplace    = True)

#Creating a variable for High and low spend based on case information
original_df['AVERAGE_SPEND'] = original_df['REVENUE']/original_df['TOTAL_MEALS_ORDERED']

#Average Spending outlier columns
original_df['AVERAGE_SPEND_HIGH'] = 0

condition = original_df.loc[0:,'AVERAGE_SPEND_HIGH'][original_df['AVERAGE_SPEND'] > average_spend_high]

original_df['AVERAGE_SPEND_HIGH'].replace(to_replace = condition,
                                      value      = 1,
                                      inplace    = True)

original_df['AVERAGE_SPEND'] = original_df['REVENUE']/original_df['TOTAL_MEALS_ORDERED']


original_df['AVERAGE_SPEND_LOW'] = 0

condition = original_df.loc[0:,'AVERAGE_SPEND_LOW'][original_df['AVERAGE_SPEND'] < average_spend_low]

original_df['AVERAGE_SPEND_LOW'].replace(to_replace = condition,
                                      value      = 1,
                                      inplace    = True)

original_df['AVERAGE_SPEND_ALCOHOL'] = 0

condition = original_df.loc[0:,'AVERAGE_SPEND_ALCOHOL'][original_df['AVERAGE_SPEND'] > average_spend_alcohol]

original_df['AVERAGE_SPEND_ALCOHOL'].replace(to_replace = condition,
                                      value      = 1,
                                      inplace    = True)

#Customer service contact per meal Outlier columns

original_df['CUSTOMER_SERVICE_CONTACT_PER_MEAL_OUT'] = 0

condition = original_df.loc[0:,'CUSTOMER_SERVICE_CONTACT_PER_MEAL_OUT'][original_df['CUSTOMER_SERVICE_CONTACT_PER_MEAL'] > customer_service_contact_per_meal ]

original_df['CUSTOMER_SERVICE_CONTACT_PER_MEAL_OUT'].replace(to_replace = condition,
                                      value      = 1,
                                      inplace    = True)

#Number of recommended meals Outlier columns

original_df['NUMBER_OF_RECOMMENDED_MEALS_OUT'] = 0

condition = original_df.loc[0:,'NUMBER_OF_RECOMMENDED_MEALS_OUT'][original_df['NUMBER_OF_RECOMMENDED_MEALS'] >number_of_recommended_meals ]

original_df['NUMBER_OF_RECOMMENDED_MEALS_OUT'].replace(to_replace = condition,
                                      value      = 1,
                                      inplace    = True)


#Percentage of tiem spent on videos Outlier columns

original_df['PERCENTAGE_OF_TIME_SPENT_ON_VIDEOS_OUT'] = 0

condition = original_df.loc[0:,'PERCENTAGE_OF_TIME_SPENT_ON_VIDEOS_OUT'][original_df['PERCENTAGE_OF_TIME_SPENT_ON_VIDEOS'] > percentage_of_time_spent_on_video ]

original_df['PERCENTAGE_OF_TIME_SPENT_ON_VIDEOS_OUT'].replace(to_replace = condition,
                                      value      = 1,
                                      inplace    = True)



# In[159]:


##############################################################################
#Saving and reloading data                       
##############################################################################

original_df.to_excel('original_df_expanded.xlsx',
                 index = False)
# loading saved file
original_df = pd.read_excel('original_df_expanded.xlsx')


# <h4>Feature Engineering - Setup for functions</h4>
# Functions that will be usefull in modeling are set up.

# In[179]:


########################################
# Setup for functions
########################################
def optimal_neighbors(X_data,
                      y_data,
                      standardize = True,
                      pct_test=0.25,
                      seed=802,
                      response_type='reg',
                      max_neighbors=20,
                      show_viz=True):
    """
Exhaustively compute training and testing results for KNN across
[1, max_neighbors]. Outputs the maximum test score and (by default) a
visualization of the results.
PARAMETERS
----------
X_data        : explanatory variable data
y_data        : response variable
standardize   : whether or not to standardize the X data, default True
pct_test      : test size for training and validation from (0,1), default 0.25
seed          : random seed to be used in algorithm, default 802
response_type : type of neighbors algorithm to use, default 'reg'
    Use 'reg' for regression (KNeighborsRegressor)
    Use 'class' for classification (KNeighborsClassifier)
max_neighbors : maximum number of neighbors in exhaustive search, default 20
show_viz      : display or surpress k-neigbors visualization, default True
"""    
    
    
    if standardize == True:
        # optionally standardizing X_data
        scaler             = StandardScaler()
        scaler.fit(X_data)
        X_scaled           = scaler.transform(X_data)
        X_scaled_df        = pd.DataFrame(X_scaled)
        X_data             = X_scaled_df



    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_data,
                                                        y_data,
                                                        test_size = pct_test,
                                                        random_state = seed)


    # creating lists for training set accuracy and test set accuracy
    training_accuracy = []
    test_accuracy = []
    
    
    # setting neighbor range
    neighbors_settings = range(1, max_neighbors + 1)


    for n_neighbors in neighbors_settings:
        # building the model based on response variable type
        if response_type == 'reg':
            clf = KNeighborsRegressor(n_neighbors = n_neighbors)
            clf.fit(X_train, y_train)
            
        elif response_type == 'class':
            clf = KNeighborsClassifier(n_neighbors = n_neighbors)
            clf.fit(X_train, y_train)            
            
        else:
            print("Error: response_type must be 'reg' or 'class'")
        
        
        # recording the training set accuracy
        training_accuracy.append(clf.score(X_train, y_train))
    
        # recording the generalization accuracy
        test_accuracy.append(clf.score(X_test, y_test))


    # optionally displaying visualization
    if show_viz == True:
        # plotting the visualization
        fig, ax = plt.subplots(figsize=(12,8))
        plt.plot(neighbors_settings, training_accuracy, label = "training accuracy")
        plt.plot(neighbors_settings, test_accuracy, label = "test accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("n_neighbors")
        plt.legend()
        plt.show()
    
    
    # returning optimal number of neighbors
    print(f"The optimal number of neighbors is: {test_accuracy.index(max(test_accuracy))+1}")
    return test_accuracy.index(max(test_accuracy))+1


########################################
# visual_cm
########################################
def visual_cm(true_y, pred_y, labels = None):
    """
Creates a visualization of a confusion matrix.

PARAMETERS
----------
true_y : true values for the response variable
pred_y : predicted values for the response variable
labels : , default None
    """
    # visualizing the confusion matrix

    # setting labels
    lbls = labels
    

    # declaring a confusion matrix object
    cm = confusion_matrix(y_true = true_y,
                          y_pred = pred_y)


    # heatmap
    sns.heatmap(cm,
                annot       = True,
                xticklabels = lbls,
                yticklabels = lbls,
                cmap        = 'Blues',
                fmt         = 'g')


    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix of the Classifier')
    plt.show()
    
########################################
# plot_feature_importances
########################################
def plot_feature_importances(model, train, export = False):
    """
    Plots the importance of features from a CART model.
    
    PARAMETERS
    ----------
    model  : CART model
    train  : explanatory variable training data
    export : whether or not to export as a .png image, default False
    """
    
    # declaring the number
    n_features = X_train.shape[1]
    
    # setting plot window
    fig, ax = plt.subplots(figsize=(12,9))
    
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(pd.np.arange(n_features), train.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    
    if export == True:
        plt.savefig('Tree_Leaf_50_Feature_Importance.png')


# <h4>Data exploration - Correlations for model decisions</h4>
# We set up a Pearson correlation to gain insight and determine which variables to
# include in the models.
# It shows that only 11 of 84 variables have a correlation of over 0.1.
# All of them are related to either followed recommendations, email group, 
# cancellations before noon or mobile number.
# 
# Followed recommendations has the highest correlation followed by Email grouping.
# Professional E-mails have a positive correlation, Junk a negative correlation and
# Personal has no significant correlation. It shows that people with Junk E-mails
# have less interest in the service, while professionals are more interested. Since
# personal E-mails, such as g-mail are often used both seriously and as junk it 
# is difficult to establish a correlation.  Since mobile number is also significant
# we can determine that people that are actually interested and give real information
# are more likely to accept the promotion. It might also be that we could not 
# reach them with the promotion since they did not look in their junk mail. 
# It depends on how the promotion was set up, if they sent emails, made calls or 
# advertised it in the recommendations. Cancellations before noon most likely implies
# that people on a weekly plan that order meals for a week are more likely to 
# get the weekly promotion. Only people that buy their meals regularly would need
# to cancel their meals since they do not order it on the day but on a plan.
# Since those with zero cancellations have negative correlations and those with
# high cancellations have positive correlations it is likely those people on plans are
# more likely to order the promotion.

# In[160]:


##############################################################################
#Data Exploration - Significance - Correlations                        
##############################################################################

# printing (Pearson) correlations with Cross_SELL_SUCCESS
df_corr = original_df.corr().round(2)
print(df_corr.loc['CROSS_SELL_SUCCESS'].sort_values(ascending = False))


# In[ ]:


##############################################################################
#Setup for logistic regression to determine p-values              
##############################################################################
#for val in original_df_data:
#    print(f"{val} +")


# In[ ]:


##############################################################################
#Logistic Regression to determine p-values - commented out            
##############################################################################
"""
# instantiating a logistic regression model object
logistic_full = smf.logit(formula = f"""CROSS_SELL_SUCCESS ~ REVENUE +
TOTAL_MEALS_ORDERED +
UNIQUE_MEALS_PURCH +
CONTACTS_W_CUSTOMER_SERVICE +
PRODUCT_CATEGORIES_VIEWED +
AVG_TIME_PER_SITE_VISIT +
MOBILE_NUMBER +
TASTES_AND_PREFERENCES +
MOBILE_LOGINS +
PC_LOGINS +
WEEKLY_PLAN +
EARLY_DELIVERIES +
LATE_DELIVERIES +
PACKAGE_LOCKER +
REFRIGERATED_LOCKER +
FOLLOWED_RECOMMENDATIONS_PCT +
AVG_PREP_VID_TIME +
LARGEST_ORDER_SIZE +
AVG_CLICKS_PER_VISIT +
TOTAL_PHOTOS_VIEWED +
m_FAMILY_NAME +
EMAIL_GROUP_JUNK +
EMAIL_GROUP_PROFESSIONAL +
MEDIAN_MEAL_RATING_2 +
MEDIAN_MEAL_RATING_3 +
MEDIAN_MEAL_RATING_4 +
MEDIAN_MEAL_RATING_5 +
MASTER_CLASSES_ATTENDED_YES +
CUSTOMER_SERVICE_CONTACT_PER_MEAL +
AVERAGE_SPEND +
AVERAGE_EARLY_DELIVERIES +
AVERAGE_LATE_DELIVERIES +
AVERAGE_CANCELLATIONS_BEFORE_NOON +
AVERAGE_AVG_CLICKS_PER_VISIT +
AVERAGE_CANCELLATIONS_AFTER_NOON +
CANCELLATIONS_AFTER_NOON_YES +
CANCELLATIONS_BEFORE_NOON_TWO +
CANCELLATIONS_BEFORE_NOON_THREE +
CANCELLATIONS_BEFORE_NOON_FOUR_OR_MORE +
CANCELLATIONS_BEFORE_NOON_ZERO +
NUMBER_OF_RECOMMENDED_MEALS +
PERCENTAGE_OF_TIME_SPENT_ON_VIDEOS +
WEEKLY_PLAN_BY_REVNUE +
WEEKLY_PLAN_BY_TOTAL_MEALS +
OUT_TOTAL_MEALS_ORDERED +
CHANGE_TOTAL_MEALS_ORDERED +
OUT_UNIQUE_MEALS_PURCH +
CHANGE_UNIQUE_MEALS_PURCH +
OUT_CONTACTS_W_CUSTOMER_SERVICE +
CHANGE_CONTACTS_W_CUSTOMER_SERVICE +
OUT_AVG_TIME_PER_SITE_VISIT +
CHANGE_AVG_TIME_PER_SITE_VISIT +
OUT_CANCELLATIONS_BEFORE_NOON +
CHANGE_CANCELLATIONS_BEFORE_NOON +
OUT_CANCELLATIONS_AFTER_NOON +
OUT_MOBILE_LOGINS +
OUT_PC_LOGINS +
OUT_WEEKLY_PLAN +
OUT_EARLY_DELIVERIES +
OUT_LATE_DELIVERIES +
CHANGE_LATE_DELIVERIES +
OUT_FOLLOWED_RECOMMENDATIONS_PCT +
OUT_AVG_PREP_VID_TIME +
OUT_LARGEST_ORDER_SIZE +
CHANGE_LARGEST_ORDER_SIZE +
OUT_AVG_CLICKS_PER_VISIT +
CHANGE_AVG_CLICKS_PER_VISIT +
OUT_TOTAL_PHOTOS_VIEWED +
AVERAGE_SPEND_HIGH +
AVERAGE_SPEND_LOW +
AVERAGE_SPEND_ALCOHOL +
CUSTOMER_SERVICE_CONTACT_PER_MEAL_OUT +
NUMBER_OF_RECOMMENDED_MEALS_OUT +
PERCENTAGE_OF_TIME_SPENT_ON_VIDEOS_OUT """,
                           data    = original_df_train)


# fitting the model object
results_logistic = logistic_full.fit()


# checking the results SUMMARY
results_logistic.summary()
"""


# <h4>Candidate modeling </h4>
# Based on the full model1, we have created a number of alternative models,
# based on their p-values, correlation score and model response.

# In[399]:


##############################################################################
# Feature Engineering Defining X_variables for candidate modeling                     
##############################################################################
# Defining data to be used in models (X-Variables)

 
model1 = [
'REVENUE',
'TOTAL_MEALS_ORDERED',
'UNIQUE_MEALS_PURCH',
'CONTACTS_W_CUSTOMER_SERVICE',
'PRODUCT_CATEGORIES_VIEWED',
'AVG_TIME_PER_SITE_VISIT',
'MOBILE_NUMBER',
'TASTES_AND_PREFERENCES',
'MOBILE_LOGINS',
'PC_LOGINS',
'WEEKLY_PLAN',
'EARLY_DELIVERIES',
'LATE_DELIVERIES',
'PACKAGE_LOCKER',
'REFRIGERATED_LOCKER',
'FOLLOWED_RECOMMENDATIONS_PCT',
'AVG_PREP_VID_TIME',
'LARGEST_ORDER_SIZE',
'AVG_CLICKS_PER_VISIT',
'TOTAL_PHOTOS_VIEWED',
'm_FAMILY_NAME',
'EMAIL_GROUP_JUNK',
'EMAIL_GROUP_PROFESSIONAL',
'MEDIAN_MEAL_RATING_2',
'MEDIAN_MEAL_RATING_3',
'MEDIAN_MEAL_RATING_4',
'MEDIAN_MEAL_RATING_5',
'MASTER_CLASSES_ATTENDED_YES',
'CUSTOMER_SERVICE_CONTACT_PER_MEAL',
'AVERAGE_SPEND',
'AVERAGE_EARLY_DELIVERIES',
'AVERAGE_LATE_DELIVERIES',
'AVERAGE_CANCELLATIONS_BEFORE_NOON',
'AVERAGE_AVG_CLICKS_PER_VISIT',
'AVERAGE_CANCELLATIONS_AFTER_NOON',
'CANCELLATIONS_AFTER_NOON_YES',
'CANCELLATIONS_BEFORE_NOON_TWO',
'CANCELLATIONS_BEFORE_NOON_THREE',
'CANCELLATIONS_BEFORE_NOON_FOUR_OR_MORE',
'CANCELLATIONS_BEFORE_NOON_ZERO',
'NUMBER_OF_RECOMMENDED_MEALS',
'PERCENTAGE_OF_TIME_SPENT_ON_VIDEOS',
'WEEKLY_PLAN_BY_REVNUE',
'WEEKLY_PLAN_BY_TOTAL_MEALS',
'OUT_TOTAL_MEALS_ORDERED',
'CHANGE_TOTAL_MEALS_ORDERED',
'OUT_UNIQUE_MEALS_PURCH',
'CHANGE_UNIQUE_MEALS_PURCH',
'OUT_CONTACTS_W_CUSTOMER_SERVICE',
'CHANGE_CONTACTS_W_CUSTOMER_SERVICE',
'OUT_AVG_TIME_PER_SITE_VISIT',
'CHANGE_AVG_TIME_PER_SITE_VISIT',
'OUT_CANCELLATIONS_BEFORE_NOON',
'CHANGE_CANCELLATIONS_BEFORE_NOON',
'OUT_CANCELLATIONS_AFTER_NOON',
'OUT_MOBILE_LOGINS',
'OUT_PC_LOGINS',
'OUT_WEEKLY_PLAN',
'OUT_EARLY_DELIVERIES',
'OUT_LATE_DELIVERIES',
'CHANGE_LATE_DELIVERIES',
'OUT_FOLLOWED_RECOMMENDATIONS_PCT',
'OUT_AVG_PREP_VID_TIME',
'OUT_LARGEST_ORDER_SIZE',
'CHANGE_LARGEST_ORDER_SIZE',
'OUT_AVG_CLICKS_PER_VISIT',
'CHANGE_AVG_CLICKS_PER_VISIT',
'OUT_TOTAL_PHOTOS_VIEWED',
'AVERAGE_SPEND_HIGH',
'AVERAGE_SPEND_LOW',
'AVERAGE_SPEND_ALCOHOL',
'CUSTOMER_SERVICE_CONTACT_PER_MEAL_OUT',
'NUMBER_OF_RECOMMENDED_MEALS_OUT',
'PERCENTAGE_OF_TIME_SPENT_ON_VIDEOS_OUT']

# Variables with 0.05 correlation or more
model2    = [
"FOLLOWED_RECOMMENDATIONS_PCT",           
"NUMBER_OF_RECOMMENDED_MEALS",             
"NUMBER_OF_RECOMMENDED_MEALS_OUT",          
"EMAIL_GROUP_PROFESSIONAL" ,                 
"CANCELLATIONS_BEFORE_NOON",               
"OUT_FOLLOWED_RECOMMENDATIONS_PCT",         
"AVERAGE_CANCELLATIONS_BEFORE_NOON",        
"MOBILE_NUMBER",                             
"CANCELLATIONS_BEFORE_NOON_THREE",         
"CANCELLATIONS_BEFORE_NOON_ZERO",           
"EMAIL_GROUP_JUNK",
"CANCELLATIONS_BEFORE_NOON_FOUR_OR_MORE",   
"CANCELLATIONS_BEFORE_NOON_TWO",           
"TASTES_AND_PREFERENCES",                  
"REFRIGERATED_LOCKER",                      
"OUT_CANCELLATIONS_BEFORE_NOON",             
"CANCELLATIONS_AFTER_NOON_NO",              
"MASTER_CLASSES_ATTENDED_YES",               
"CANCELLATIONS_AFTER_NOON_YES",             
"CANCELLATIONS_AFTER_NOON",                 
"PC_LOGINS",                                
"MASTER_CLASSES_ATTENDED_NO",               
"OUT_CONTACTS_W_CUSTOMER_SERVICE"          
 ]
    
# Variables with 0.1 correlation or more - good for random forrest
model3    = [
"FOLLOWED_RECOMMENDATIONS_PCT",           
"NUMBER_OF_RECOMMENDED_MEALS",             
"NUMBER_OF_RECOMMENDED_MEALS_OUT",          
"EMAIL_GROUP_PROFESSIONAL" ,                 
"CANCELLATIONS_BEFORE_NOON",               
"OUT_FOLLOWED_RECOMMENDATIONS_PCT",         
"AVERAGE_CANCELLATIONS_BEFORE_NOON",        
"MOBILE_NUMBER",                             
"CANCELLATIONS_BEFORE_NOON_THREE",         
"CANCELLATIONS_BEFORE_NOON_ZERO",           
"EMAIL_GROUP_JUNK"
 ]

# Variables with 0.05 p-value or less
model4    = [
"TOTAL_MEALS_ORDERED",           
"MOBILE_NUMBER",             
"TASTES_AND_PREFERENCES",          
"PC_LOGINS" ,                 
"EARLY_DELIVERIES",               
"FOLLOWED_RECOMMENDATIONS_PCT",         
"EMAIL_GROUP_JUNK",        
"EMAIL_GROUP_PROFESSIONAL",                             
"CANCELLATIONS_BEFORE_NOON_TWO",         
"CHANGE_AVG_CLICKS_PER_VISIT",           
"AVERAGE_SPEND_HIGH",
"AVERAGE_SPEND_LOW",
"AVERAGE_SPEND_ALCOHOL",
"NUMBER_OF_RECOMMENDED_MEALS_OUT",
 ]

# combination of model 3 and 4 
model5    = [
"FOLLOWED_RECOMMENDATIONS_PCT",                       
"EMAIL_GROUP_PROFESSIONAL" ,  
"NUMBER_OF_RECOMMENDED_MEALS", 
"CANCELLATIONS_BEFORE_NOON",                               
"MOBILE_NUMBER",  
"EMAIL_GROUP_JUNK"
 ]

# alternative of model 5 best gbm
model6    = [
"FOLLOWED_RECOMMENDATIONS_PCT",                       
"EMAIL_GROUP_PROFESSIONAL" ,  
"CANCELLATIONS_BEFORE_NOON", 
"MOBILE_NUMBER",
"TASTES_AND_PREFERENCES",
"EMAIL_GROUP_JUNK"
 ]


# In[400]:


##############################################################################
#Feature Engineering Scaling Data                
##############################################################################

original_df_data  =  original_df.loc[ : ,model6]


# Preparing response variable data
original_df_target = original_df.loc[:, 'CROSS_SELL_SUCCESS']

# INSTANTIATING a StandardScaler() object
scaler = StandardScaler()


# FITTING the scaler with original_df_data
scaler.fit(original_df_data)


# TRANSFORMING our data after fit
X_scaled = scaler.transform(original_df_data)


# Converting scaled data into a DataFrame
X_scaled_df = pd.DataFrame(X_scaled)
X_scaled_df.columns = original_df_data.columns


# In[401]:


##############################################################################
#Train Test Split                
##############################################################################

# Preparing training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
            X_scaled_df,
            original_df_target,
            test_size = 0.25,
            random_state = 222,
            stratify = original_df_target)

# merging training data for statsmodels
original_df_train = pd.concat([X_train, y_train], axis = 1)


# <h2>Building Models </h2>
# Comparing Logistic Regression, K-Neighbors, Trees, Random Forests and
# Gradient Boosting to see which model is best after Hyperparameter tuning.
# Hyperparameter tuning is commented out.

# In[402]:


##############################################################################
#Logistic Regression model              
##############################################################################

# INSTANTIATING a logistic regression model
logreg = LogisticRegression(solver = 'lbfgs',
                            C = 1,
                            random_state = 802)


# FITTING the training data
logreg_fit = logreg.fit(X_train, y_train)


# PREDICTING based on the testing set
logreg_pred = logreg_fit.predict(X_test)


# SCORING the results
print('Training ACCURACY:', logreg_fit.score(X_train, y_train).round(4))
print('Testing  ACCURACY:', logreg_fit.score(X_test, y_test).round(4))
print('Testing  ACCURACY:',roc_auc_score(y_true  = y_test,
              y_score = logreg_pred).round(4))


# In[ ]:


##############################################################################
# GridSearchCV for logistic Regression
##############################################################################
"""
# declaring a hyperparameter space
C_space          = pd.np.arange(0.1, 3.0, 0.1)
warm_start_space = [True, False]


# creating a hyperparameter grid
param_grid = {'C'          : C_space,
              'warm_start' : warm_start_space}


# INSTANTIATING the model object without hyperparameters
lr_tuned = LogisticRegression(solver = 'lbfgs',
                              max_iter = 1000,
                              random_state = 802)


# GridSearchCV object
lr_tuned_cv = GridSearchCV(estimator  = lr_tuned,
                           param_grid = param_grid,
                           cv         = 3,
                           scoring    = make_scorer(roc_auc_score,
                                                    needs_threshold = False))


# FITTING to the FULL DATASET (due to cross-validation)
lr_tuned_cv.fit(original_df_data, original_df_target)


# printing the optimal parameters and best score
print("Tuned Parameters  :", lr_tuned_cv.best_params_)
print("Tuned CV AUC      :", lr_tuned_cv.best_score_.round(4))
"""


# <h2> K-Neighbors and setup </h2> 
# Setting up optimal neighbors and instantiating model.
# setup is commented out

# In[176]:


##############################################################################
# Finding optimal number of neighbors
##############################################################################

# determining the optimal number of neighbors
#opt_neighbors = optimal_neighbors(X_data = X_train,
#                                  y_data = y_train)


# In[403]:


##############################################################################
#K-Neighbors model             
##############################################################################

# INSTANTIATING a KNN classification model with optimal neighbors
knn_opt = KNeighborsClassifier(n_neighbors = opt_neighbors)


# FITTING the training data
knn_fit = knn_opt.fit(X_train, y_train)


# PREDICTING based on the testing set
knn_pred = knn_fit.predict(X_test)


# SCORING the results
print('Training ACCURACY:', knn_fit.score(X_train, y_train).round(4))
print('Testing  ACCURACY:', knn_fit.score(X_test, y_test).round(4))
print('AUC Score        :', roc_auc_score(y_true  = y_test,
                                          y_score = knn_pred).round(4))


# In[404]:


##############################################################################
#Tree model             
##############################################################################

# INSTANTIATING a classification tree object
tree_pruned      = DecisionTreeClassifier(max_depth = 12,
                                          min_samples_leaf = 25,
                                          min_samples_split=100,
                                          splitter='random',
                                          random_state = 802)


# FITTING the training data
tree_pruned_fit  = tree_pruned.fit(X_train, y_train)


# PREDICTING on new data
tree_pred = tree_pruned_fit.predict(X_test)


# SCORING the model
print('Training ACCURACY:', tree_pruned_fit.score(X_train, y_train).round(4))
print('Testing  ACCURACY:', tree_pruned_fit.score(X_test, y_test).round(4))
print('AUC Score        :', roc_auc_score(y_true  = y_test,
                                          y_score = tree_pred).round(4))



# In[387]:


##############################################################################
# GridSearchCV for tree
##############################################################################

# declaring a hyperparameter space
"""
split =["best", "random"]
crit = ["gini", "entropy"]
deep = pd.np.arange(2, 20, 1)
min_split = pd.np.arange(50, 200, 10)
        
# creating a hyperparameter grid
param_grid = {'criterion'          : crit,
              'splitter' : split,
             'max_depth': deep,
             'min_samples_split' : min_split}



#INSTANTIATING the model object without hyperparameters
l_tree = DecisionTreeClassifier()


# GridSearchCV object
l_tree_cv = GridSearchCV(estimator  = l_tree,
                           param_grid = param_grid,
                           cv         = 3,
                           scoring    = make_scorer(roc_auc_score,
                                                    needs_threshold = False))


# FITTING to the FULL DATASET (due to cross-validation)
l_tree_cv.fit(original_df_data, original_df_target)


# PREDICT step is not needed


# printing the optimal parameters and best score
print("Tuned Parameters  :", l_tree_cv.best_params_)
print("Tuned CV AUC      :", l_tree_cv.best_score_.round(4))
"""


# In[183]:


##############################################################################
# Feature importance plotting
##############################################################################
# plotting feature importance
"""
plot_feature_importances(tree_pruned_fit,
                         train = X_train,
                         export = False)
"""


# In[405]:


##############################################################################
#Random Forest model           
##############################################################################
# INSTANTIATING a random forest model with default values
rf_default = RandomForestClassifier(n_estimators     = 100,
                                    criterion        = 'gini',
                                    max_depth        = None,
                                    min_samples_leaf = 21,
                                    bootstrap        = True,
                                    warm_start       = False,
                                    random_state     = 802)
# FITTING the training data
rf_default_fit = rf_default.fit(X_train, y_train)


# PREDICTING based on the testing set
rf_default_fit_pred = rf_default_fit.predict(X_test)


# SCORING the results
print('Training ACCURACY:', rf_default_fit.score(X_train, y_train).round(4))
print('Testing  ACCURACY:', rf_default_fit.score(X_test, y_test).round(4))
print('AUC Score        :', roc_auc_score(y_true  = y_test,
                                          y_score = rf_default_fit_pred).round(4))


# In[384]:


##############################################################################
# GridSearchCV for random forest
##############################################################################
# declaring a hyperparameter space
"""
estimator_space  = pd.np.arange(100, 1100, 250)
leaf_space       = pd.np.arange(1, 31, 10)
criterion_space  = ['gini', 'entropy']
bootstrap_space  = [True, False]
warm_start_space = [True, False]


# creating a hyperparameter grid
param_grid = {'n_estimators'     : estimator_space,
              'min_samples_leaf' : leaf_space,
              'criterion'        : criterion_space,
              'bootstrap'        : bootstrap_space,
              'warm_start'       : warm_start_space}


# INSTANTIATING the model object without hyperparameters
full_forest_grid = RandomForestClassifier(random_state = 802)


# GridSearchCV object
full_forest_cv = GridSearchCV(estimator  = full_forest_grid,
                              param_grid = param_grid,
                              cv         = 3,
                              scoring    = make_scorer(roc_auc_score,
                                           needs_threshold = False))


# FITTING to the FULL DATASET (due to cross-validation)
full_forest_cv.fit(original_df_data, original_df_target)


# PREDICT step is not needed


# printing the optimal parameters and best score
print("Tuned Parameters  :", full_forest_cv.best_params_)
print("Tuned Training AUC:", full_forest_cv.best_score_.round(4))
"""


# In[329]:


##############################################################################
# Feature importance plotting
##############################################################################
"""
plot_feature_importances(rf_default_fit,
                         train = X_train,
                         export = True)
"""


# In[406]:


##############################################################################
#Gradient Boosting model         
##############################################################################

# INSTANTIATING the model object without hyperparameters
full_gbm_default = GradientBoostingClassifier(loss              = 'deviance',
                                              learning_rate     = 0.05,
                                              n_estimators      = 110,
                                              criterion         = 'friedman_mse',
                                              max_depth         = 3,
                                              min_samples_split = 100,
                                              warm_start        = False,
                                              random_state      = 802)


# FIT step is needed as we are not using .best_estimator
full_gbm_default_fit = full_gbm_default.fit(X_train, y_train)


# PREDICTING based on the testing set
full_gbm_default_pred = full_gbm_default_fit.predict(X_test)


# SCORING the results
print('Training ACCURACY:', full_gbm_default_fit.score(X_train, y_train).round(4))
print('Testing ACCURACY :', full_gbm_default_fit.score(X_test, y_test).round(4))
print('AUC Score        :', roc_auc_score(y_true  = y_test,
                                          y_score = full_gbm_default_pred).round(4))


# In[347]:


##############################################################################
#GridSearchCV for Gradient Boosting model         
##############################################################################
# declaring a hyperparameter space
"""
learn_space     = pd.np.arange(0.1, 1.6, 0.3)
estimator_space = pd.np.arange(50, 250, 50)
depth_space     = pd.np.arange(1, 10, 0.1)
split           = pd.np.arange(2, 100, 10)

# creating a hyperparameter grid
param_grid = { 'min_samples_split' : split    ,
              'n_estimators' : estimator_space,
              'learning_rate'     : learn_space,
              'max_depth'         : depth_space}
              
      


# INSTANTIATING the model object without hyperparameters
full_gbm_grid = GradientBoostingClassifier(random_state = 802)


# GridSearchCV object
full_gbm_cv = GridSearchCV(estimator  = full_gbm_grid,
                           param_grid = param_grid,
                           cv         = 3,
                           scoring    = make_scorer(roc_auc_score,
                                        needs_threshold = False))


# FITTING to the FULL DATASET (due to cross-validation)
full_gbm_cv.fit(original_df_data, original_df_target)


# PREDICT step is not needed


# printing the optimal parameters and best score
print("Tuned Parameters  :", full_gbm_cv.best_params_)
print("Tuned Training AUC:", full_gbm_cv.best_score_.round(4))
"""


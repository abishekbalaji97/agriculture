# Check the versions of the installed libraries before loading the libraries required for building the model

# Checking Python version
import sys
print('Python: {}'.format(sys.version))

#Checking scipy version
import scipy
print('scipy: {}'.format(scipy.__version__))

# Checking numpy version
import numpy
print('numpy: {}'.format(numpy.__version__))

# Checking matplotlib version
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))

# Checking pandas version
import pandas
print('pandas: {}'.format(pandas.__version__))

# Checking scikit-learn version
import sklearn
print('sklearn: {}'.format(sklearn.__version__))

# Checking seaborn version
import seaborn as sns
print('seaborn: {}'.format(sns.__version__))

# Loading the  libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import metrics
from scipy import stats

# Use numpy to convert to pandas dataframes to arrays
import numpy as np

#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Import train_test_split function from scikit-learn
from sklearn.model_selection import train_test_split

# Load dataset
url=open("krish.csv","r")
names = ['Temperature', 'Precipitation', 'Effective Rainfall', 'Insolation', 'Light usage efficiency','Light interception factor','Wind speed','Humidity','Days after emergence',
'Efficiency_score']
dataset = pandas.read_csv(url, names=names)

#Change the size of the pandas output screen to accomodate all the columns of a dataset for statistical analysis
pandas.set_option('display.max_rows', 90)
pandas.set_option('display.max_columns', 90)
pandas.set_option('display.width', 70)

#to find the dimensionality of the DataFrame before removing null records
print("\nThe dimensionality of the dataset before removing null records",dataset.shape,"\n")
print("Removing the null records with dropna function----")
#Data Cleaning Step
#Remove the rows with null values
dataset.dropna(subset=["Efficiency_score","Light interception factor","Days after emergence"],axis=0,inplace=True)

#to find the dimensionality of the DataFrame after removing null records
print("The dimensionality of the dataset after removing null records",dataset.shape,"\n")

#Normalizing the datasets
dataset["Temperature"]=dataset["Temperature"]/dataset["Temperature"].max()
dataset["Precipitation"]=dataset["Precipitation"]/dataset["Precipitation"].max()
dataset["Effective Rainfall"]=dataset["Effective Rainfall"]/dataset["Effective Rainfall"].max()
dataset["Insolation"]=dataset["Insolation"]/dataset["Insolation"].max()
dataset["Light usage efficiency"]=dataset["Light usage efficiency"]/dataset["Light usage efficiency"].max()
for item in dataset["Light interception factor"]:
    if item!=0:
	    dataset["Light interception factor"]=dataset["Light interception factor"]/abs(dataset["Light interception factor"]).max()
dataset["Wind speed"]=dataset["Wind speed"]/dataset["Wind speed"].max()
dataset["Humidity"]=dataset["Humidity"]/dataset["Humidity"].max()
dataset["Days after emergence"]=dataset["Days after emergence"]/dataset["Days after emergence"].max()

#Check the data types of each column
print(dataset.dtypes)
print("\n")

# head is used to peek at the first 20 records
print(dataset.head(20))
print("\n")

# description of quick statistics of dataset
print(dataset.describe(include="all"))
print("\n")

#to print the summary of the DataFrame
print(dataset.info(verbose=None, buf=None, max_cols=None, memory_usage=None, null_counts=None))
print("\n")

# class distribution ie. the number of instances of each class are shown below.
print(dataset.groupby('Efficiency_score').size())
print("\n")

#creating the dictionary to be used for mapping the efficiency scores to numeric values as most machine learning models process
# only numerical data. This process is called Integer Encoding
efficiencyscore = {'C': 2,'B': 1,'A': 0} 
  
# traversing through the dataframe's 
# efficiency score column and writing 
# values where key matches 
dataset.Efficiency_score = [efficiencyscore[item] for item in dataset.Efficiency_score]

#displaying the dataset to show the success of the changes
print(dataset[["Efficiency_score"]])
print("\n")

# Features
X=dataset[['Temperature', 'Precipitation', 'Effective Rainfall', 'Insolation', 'Light usage efficiency','Light interception factor','Wind speed','Humidity','Days after emergence']] 

# Labels
y=dataset['Efficiency_score'] 

# box and whisker plots generation to get short summary of sample and measures of data
#and to spot outliers easily
X.plot(kind='box', subplots=True, layout=(9,9), sharex=False, sharey=False)
plt.show()

# histogram generation
X.hist() 
plt.show()

#scatter plots for showing the relationship between input variables and the efficiency score along with the fitted regressions line and correlation statistics
sns.regplot(x="Temperature",y="Efficiency_score",data=dataset)
plt.show()
print(dataset[["Temperature","Efficiency_score"]].corr())
print("\n")

sns.regplot(x="Precipitation",y="Efficiency_score",data=dataset)
plt.show()
print(dataset[["Precipitation","Efficiency_score"]].corr())
print("\n")

sns.regplot(x="Effective Rainfall",y="Efficiency_score",data=dataset)
plt.show()
print(dataset[["Effective Rainfall","Efficiency_score"]].corr())
print("\n")

sns.regplot(x="Insolation",y="Efficiency_score",data=dataset)
plt.show()
print(dataset[["Insolation","Efficiency_score"]].corr())
print("\n")

sns.regplot(x="Light usage efficiency",y="Efficiency_score",data=dataset)
plt.show()
print(dataset[["Light usage efficiency","Efficiency_score"]].corr())
print("\n")

sns.regplot(x="Light interception factor",y="Efficiency_score",data=dataset)
plt.show()
print(dataset[["Light interception factor","Efficiency_score"]].corr())
print("\n")

sns.regplot(x="Wind speed",y="Efficiency_score",data=dataset)
plt.show()
print(dataset[["Wind speed","Efficiency_score"]].corr())
print("\n")

sns.regplot(x="Humidity",y="Efficiency_score",data=dataset)
plt.show()
print(dataset[["Humidity","Efficiency_score"]].corr())
print("\n")

sns.regplot(x="Days after emergence",y="Efficiency_score",data=dataset)
plt.show()
print(dataset[["Days after emergence","Efficiency_score"]].corr())
print("\n")

#Pearson coefficients and p-values of attributes
print("The Pearson coefficients and p-values of each of the attributes are shown below\n")
pearson_coef, p_value = stats.pearsonr(dataset['Temperature'], dataset['Efficiency_score'])
print("The Pearson Correlation Coefficient of Temperature is", pearson_coef, " with a P-value of P =", p_value,"\n") 

pearson_coef, p_value = stats.pearsonr(dataset['Precipitation'], dataset['Efficiency_score'])
print("The Pearson Correlation Coefficient of Precipitation is", pearson_coef, " with a P-value of P =", p_value,"\n") 

pearson_coef, p_value = stats.pearsonr(dataset['Effective Rainfall'], dataset['Efficiency_score'])
print("The Pearson Correlation Coefficient of Effective Rainfall is", pearson_coef, " with a P-value of P =", p_value,"\n") 

pearson_coef, p_value = stats.pearsonr(dataset['Insolation'], dataset['Efficiency_score'])
print("The Pearson Correlation Coefficient of Insolation is", pearson_coef, " with a P-value of P =", p_value,"\n") 

pearson_coef, p_value = stats.pearsonr(dataset['Light usage efficiency'], dataset['Efficiency_score'])
print("The Pearson Correlation Coefficient of Light usage efficiency is", pearson_coef, " with a P-value of P =", p_value,"\n") 

pearson_coef, p_value = stats.pearsonr(dataset['Light interception factor'], dataset['Efficiency_score'])
print("The Pearson Correlation Coefficient of Light interception factor is", pearson_coef, " with a P-value of P =", p_value,"\n") 

pearson_coef, p_value = stats.pearsonr(dataset['Wind speed'], dataset['Efficiency_score'])
print("The Pearson Correlation Coefficient of Wind speed is", pearson_coef, " with a P-value of P =", p_value,"\n") 

pearson_coef, p_value = stats.pearsonr(dataset['Humidity'], dataset['Efficiency_score'])
print("The Pearson Correlation Coefficient of Humidity is", pearson_coef, " with a P-value of P =", p_value,"\n") 

pearson_coef, p_value = stats.pearsonr(dataset['Days after emergence'], dataset['Efficiency_score'])
print("The Pearson Correlation Coefficient of Days after emergence is", pearson_coef, " with a P-value of P =", p_value,"\n") 
# scatter plot matrix generation 
scatter_matrix(X)
plt.show()

#Finding the correlation between the different input variables
print("The correlation between different input variables is given below")
print(dataset[['Temperature','Precipitation','Effective Rainfall','Insolation','Light usage efficiency','Light interception factor','Wind speed','Humidity','Days after emergence','Efficiency_score']].corr())
print("\n")

# Labels are the values we want to predict
labels = np.array(dataset['Efficiency_score'])
# Remove the labels from the features
# axis 1 refers to the columns
features= dataset.drop('Efficiency_score', axis = 1)
# Saving feature names for later use
feature_list = list(features.columns)
# Convert to numpy array
features = np.array(features)


# Split dataset into training set and test set
# 78% training and 22% test set splitting is achieved with the above line of code
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.22,random_state=42)
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

#Create a Gaussian Classifier using the below command
clf=RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=-1)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(train_features,train_labels)
y_pred=clf.predict(test_features)
print("\nAccuracy:",metrics.accuracy_score(test_labels, y_pred))
print("Precision:",metrics.precision_score(test_labels, y_pred))
print("Recall:",metrics.recall_score(test_labels, y_pred))
print("F-Measure:",metrics.f1_score(test_labels, y_pred))
print("\n")
grade=clf.predict([[27.2, 3.97, 3, 0.002,0,0,3.1814,50.6,7]])
a=grade.item(0)
for grade, numeric in efficiencyscore.items():   
    if numeric == a:
        print("The efficiency score for the climate modules is",grade)
print("Combining this with the efficiency score from the soil module\n")
print("For the given climatic conditions and soil characteristics the type of end yield that can be expected is given below\n")
if a==0:
    print("High yield, medium quality can be expected")
elif a==1:
    print("Medium yield, medium quality can be expected")
else:
    print("Low yield, medium quality can be expected")

"""#K-means
Clustering data points may be done using the unsupervised learning approach known as K-means. The method separates the data points into K clusters in an iterative manner, with the goal of reducing the variance in each cluster.


#scipy 

NumPy is the foundation upon which SciPy, a library for scientific computing, is built. The abbreviation for Scientific Python is "SciPy." More 
utility functions for optimisation, statistics, and signal processing are made available as a result. SciPy, like NumPy, is open source, which means 
that we are free to use it.

#curve_fit

The process of finding an ideal set of parameters for a specified function that best fits a given collection of observations is known as curve fitting.
Curve fitting is a sort of optimisation. In contrast to supervised learning, curve fitting necessitates the definition of a function that acts as a map 
between various inputs and outputs.

#stats 

Collecting data, tabulating it, and drawing conclusions from it are all aspects of statistics. The study of how to gather, analyse,
interpret, and display information is the focus of this branch of applied mathematics. Statistics demonstrates how information may be utilised
to unravel perplexing puzzles. 

"""

import pandas as pd
#importing pandas library use for data read ,cleaning
import numpy as np 
#importing numpy library 
from sklearn.cluster import KMeans  
#importing kmeans library
import matplotlib.pyplot as matplotlib
# importing matplotlib use for data visualization
from sklearn.preprocessing import LabelEncoder
#labelencoder is used to convert object data to numeric format. 
from sklearn.preprocessing import MinMaxScaler
#import minmax scaler use to data normalization
from scipy.spatial.distance import cdist 
import warnings 
warnings.filterwarnings('ignore')     



def read_dataset(new_file):
    # Read the CSV file into a pandas dataframe and skip the first 4 rows
    Agricultural_land_data = pd.read_csv(new_file, skiprows=4)
    # Remove unnecessary columns
    Agricultural_land_data1 = Agricultural_land_data.drop(['Unnamed: 66', 'Indicator Code',  'Country Code'],axis=1) 
    # Transpose the dataframe and set the index to "Country Name"
    Agricultural_land_data2 = Agricultural_land_data1.set_index("Country Name")  
    Agricultural_land_data2 = Agricultural_land_data2.T 
    
     # Reset the index to make "Year" a column
    Agricultural_land_data2.reset_index(inplace=True) 
    # Rename the "index" column to "Year"
    Agricultural_land_data2.rename(columns = {'index':'Year'}, inplace = True) 
    # Return the cleaned dataframes
    return Agricultural_land_data1, Agricultural_land_data2 
#Define the filepath of the CSV file
Agricultural_land_df = '/content/drive/MyDrive/asm/API_AG.LND.AGRI.ZS_DS2_en_csv_v2_5447782.csv' 

Agricultural_land_Data, Transpose_Agricultural_land_df = read_dataset(Agricultural_land_df)  
Agricultural_land_Data.head()

# checking null values 
Agricultural_land_Data.isnull().sum()
# showing transposed data
Transpose_Agricultural_land_df.head()
# drop null values 
def Agricultural_land_Data2(Agricultural_land_Data): 
    Agricultural_land_Data1 = Agricultural_land_Data[['Country Name', 'Indicator Name', '1961', '1962', '1963',
       '1964', '1965', '1966', '1967', '1968', '1969', '1970', '1971', '1972',
       '1973', '1974', '1975', '1976', '1977', '1978', '1979', '1980', '1981',
       '1982', '1983', '1984', '1985', '1986', '1987', '1988', '1989', '1990',
       '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999',
       '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008',
       '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017',
       '2018', '2019', '2020']] 
    Agricultural_land_Data2 = Agricultural_land_Data1.dropna() # drop null values from data.
    return Agricultural_land_Data2

Agricultural_land_Data3 = Agricultural_land_Data2(Agricultural_land_Data) 
Country_Names = Agricultural_land_Data3["Country Name"]
Agricultural_land_Data3.head() # shows starting rows from data.
#copy data set 
Final_Df=Agricultural_land_Data3.copy()
Final_Df
# check shape of Agricultural_land_Data3.
Agricultural_land_Data3.shape 
# check null values from Agricultural_land_Data3.
Agricultural_land_Data3.isnull().sum()
#labelencoder is used to convert object data to numeric format. 
lb_encder = LabelEncoder()
Agricultural_land_Data3['Country Name'] = lb_encder.fit_transform(Agricultural_land_Data3['Country Name']) 
Agricultural_land_Data3.head() 
#Select all columns except for 'Country Name' and 'Indicator Name' from DataFrame Agricultural_land_Data3
X = Agricultural_land_Data3.drop(['Country Name','Indicator Name'], axis=1)
# Select only the 'Country Name' column from DataFrame Agricultural_land_Data3
y = Agricultural_land_Data3['Country Name']  

# minmaxscaler is used to data normalization
sc = MinMaxScaler()
scaled = sc.fit_transform(X)
# finding the clusters using the elbow approach.

# finding the clusters using the elbow approach.
#Define the range of clusters to test
Cluster = range(10) 
#Initialize an empty list to store the mean distances for each cluster size
Meandist = list()

for k in Cluster:
    model = KMeans(n_clusters=k+1) 
    # Fit the model to the scaled dataset
    model.fit(scaled) 
    # Calculate the mean distance between each point and its nearest cluster center
    Meandist.append(sum(np.min(cdist(scaled, model.cluster_centers_, 'euclidean'), axis=1)) / scaled.shape[0]) 

# creating the graph and setting all the parameters.
matplotlib.rcParams.update({'font.size': 20})
matplotlib.figure(figsize=(10,7))# set the figsize.
matplotlib.plot(Cluster, Meandist, marker="o", color='LightSalmon') 
#set the x axis name 
matplotlib.xlabel('Numbers of Clusters')
# set the y axis name 
matplotlib.ylabel('Average distance') 
# set title name  for graph.
matplotlib.title('Choosing k with the Elbow Method'); 
# K Means Clustering

# specify the clustering classifier.
#Create an instance of KMeans model with 2 clusters, maximum number of iterations as 128, number of initializations as 10, and random state as 10
k_m_model = KMeans(n_clusters=2, max_iter=128, n_init=10,random_state=10) 
#Fit the model on the scaled data
k_m_model.fit(scaled) 
#Use the trained model to make predictions on the scaled data
predictions = k_m_model.predict(scaled)  

# Scatter plot for two Clusters

# set color for 2 clusters.
color_map = {0 : 'Gold', 1 : 'GreenYellow'} 
def color(x):  
    return color_map[x]  
colors = list(map(color, k_m_model.labels_))   

matplotlib.rcParams.update({'font.size': 20})
# set the figsize.
matplotlib.figure(figsize=(10,7))
# set parameter
matplotlib.scatter(x=X.iloc[:,0], y=X.iloc[:,2], c=colors)  
# set x axis name.
matplotlib.xlabel('1961')
# set y axis name.  
matplotlib.ylabel('1963') 
# set title name for graph. 
matplotlib.title('Scatter plot for two Clusters');  

# Get the Centroids, as well as the label.
ctd = k_m_model.cluster_centers_
u_lbl = np.unique(predictions) 
ctd
# Scatter plot for two Clusters with Centroids

# plot the result.
matplotlib.figure(figsize=(8,6))# set the figsize.
for i in u_lbl:
    matplotlib.scatter(scaled[predictions == i , 0] , scaled[predictions == i , 1] , label = i)  

# specify graph parameters such as colour, data.
matplotlib.scatter(ctd[:,0] , ctd[:,1] , s = 80, color = 'Maroon') 
# set x axis.
matplotlib.xlabel('1961')
# set y axis.
matplotlib.ylabel('1963')
# set title name for graphs.
matplotlib.title('Scatter plot for two Clusters with Centroids') 
# set legend for graph.
matplotlib.legend()  
matplotlib.show()  

# Creating lists to extract the whole cluster.

fst_cltr=[]
scnd_cltr=[] 

# Using the loop, find out what data is available in each cluster.
for i in range(len(predictions)):
    if predictions[i]==0:
        fst_cltr.append(Agricultural_land_Data.loc[i]['Country Name']) 
    elif predictions[i]==1:
        scnd_cltr.append(Agricultural_land_Data.loc[i]['Country Name'])
# showing first cluster.
fst_cltr = np.array(fst_cltr)
print(fst_cltr)
# showing  second cluster.
scnd_cltr = np.array(scnd_cltr)
print(scnd_cltr) 

#getting country data
fst_cltr = fst_cltr[50] 
print('Country name :', fst_cltr)
Guam_country=Country_Names[Country_Names == fst_cltr]
country_indx=Guam_country.index.values
Guam_country = Agricultural_land_Data3[Agricultural_land_Data3['Country Name']==int(country_indx)]  
Guam_country = np.array(Guam_country)  
Guam_country = np.delete(Guam_country,np.s_[:2]) 
Guam_country 

scnd_cltr = scnd_cltr[9] 
print('Country name :', scnd_cltr) 
Belgium_country=Country_Names[Country_Names == scnd_cltr]
country_indx=Belgium_country.index.values
Belgium_country = Agricultural_land_Data3[Agricultural_land_Data3['Country Name']==int(country_indx)] 
Belgium_country = np.array(Belgium_country)  
Belgium_country = np.delete(Belgium_country,np.s_[:2]) 
Belgium_country 

# plot the  graph for two cluster.
year=list(range(1961,2021)) # set year
# set the figsize.
matplotlib.figure(figsize=(26,7))

matplotlib.subplot(131)
# set x axis name 
matplotlib.xlabel('Years')
# set y axis name 
matplotlib.ylabel('Agricultural land') 
# set title name 
matplotlib.title('Guam_country') 
matplotlib.plot(year,Guam_country, color='YellowGreen');

matplotlib.subplot(132)
#set x axis name 
matplotlib.xlabel('Years')
# set y axis name 
matplotlib.ylabel('Agricultural land') 
# set title name 
matplotlib.title('Belgium_country') 
matplotlib.plot(year,Belgium_country, color='MediumAquamarine');

# Curve Fitting
x = np.array(Final_Df.columns) 
#Remove the first two elements from array x and convert the rest to integers
x = np.delete(x,0) 
x = np.delete(x,0) 
#Convert the rest of the elements in array y to integers
x = x.astype(np.int)
#Select rows from DataFrame Final_Df where Indicator Name is 'Agricultural land (% of land area)' and Country Name is 'China'
curve_fit = Final_Df[(Final_Df['Indicator Name']=='Agricultural land (% of land area)') & (Final_Df['Country Name']=='China')]   
#Convert the DataFrame to a numpy array and remove the first two elements from it
y = curve_fit.to_numpy()
y = np.delete(y,0) 
y = np.delete(y,0)
#Convert the rest of the elements in array y to integers
y = y.astype(np.int)

"""
NumPy is the foundation upon which SciPy, a library for scientific computing, is built. The abbreviation for Scientific Python is "SciPy." More 
utility functions for optimisation, statistics, and signal processing are made available as a result. SciPy, like NumPy, is open source, which means 
that we are free to use it.
"""
import scipy 
"""
The process of finding an ideal set of parameters for a specified function that best fits a given collection of observations is known as curve fitting.
Curve fitting is a sort of optimisation. In contrast to supervised learning, curve fitting necessitates the definition of a function that acts as a map 
between various inputs and outputs.
"""
from scipy.optimize import curve_fit
"""
Collecting data, tabulating it, and drawing conclusions from it are all aspects of statistics. The study of how to gather, analyse,
interpret, and display information is the focus of this branch of applied mathematics. Statistics demonstrates how information may be utilised
to unravel perplexing puzzles. 
"""
from scipy import stats 


def linear_func(x, m, c):
    return m*x + c

def create_curve_fit(x,y): 

    # Use curve_fit to fit the data to the linear function
    popt, pcov = curve_fit(linear_func, x, y) 

    # Get the values for slope and y-intercept from the fit
    m, c = popt

    # Get the errors in the fit parameters
    m_err, c_err = np.sqrt(np.diag(pcov)) 

    # Set the desired confidence interval
    conf_int = 0.95   
    alpha = 1.0 - conf_int 

    # Get the upper and lower bounds of the confidence interval for slope and y-intercept using t-distribution
    m_low, m_high = scipy.stats.t.interval(alpha, len(x)-2, loc=m, scale=m_err)
    ##############
    c_low, c_high = scipy.stats.t.interval(alpha, len(x)-2, loc=c, scale=c_err)

    # Set the figure size and font size
    matplotlib.figure(figsize=(12,6)) 
    matplotlib.rcParams.update({'font.size': 18}) 

    # Plot the data
    matplotlib.plot(x, y, 'bo', color='#00ff91', label='Data') 

    # Plot the fitted function
    matplotlib.plot(x, linear_func(x, m, c), '#3333ff', label='Fitted function')

    # Shade the area between the upper and lower bounds of the confidence interval
    matplotlib.fill_between(x, linear_func(x, m_low, c_low), linear_func(x, m_high, c_high), color='Azure', alpha=0.5, label='Confidence range')

    # Add the title and axis labels
    matplotlib.title('PLOT CURVE FITTING ')
    matplotlib.xlabel('Years') 
    matplotlib.ylabel('Agricultural land') 

    # Add a legend and show the plot
    matplotlib.legend() 
    matplotlib.show()
    create_curve_fit(x,y)
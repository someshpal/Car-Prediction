# Car-Prediction
🚗 Car Price Prediction Using features such as car brand, model, year of manufacture, and other relevant factors, we aim to predict the future prices of cars. By employing various regression models, we strive to develop an accurate and reliable system for forecasting car prices over the next few years.

# 🚙 Car Prices Prediction 

[![](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=darkgreen)](https://www.python.org)  [![](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org) [![](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/) [![](https://img.shields.io/badge/SciPy-654FF0?style=for-the-badge&logo=SciPy&logoColor=white)](https://www.scipy.org) [![](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org) [![](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)  [![](https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com) [![](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white)](https://keras.io) [![](https://img.shields.io/badge/conda-342B029.svg?&style=for-the-badge&logo=anaconda&logoColor=white)](https://www.anaconda.com)

## Introduction

There are a lot of __car manufacturers__ in the US. With the development of manufacturing units and tests, there are a lot of cars being manufactured with a lot of features. Therefore, innovators are coming up with the latest developments in the field and they are ensuring that the drivers get the best experience going on a ride in these cars.

<img src = "https://media.wired.com/photos/59547e60ce3e5e760d52d429/191:100/w_1280,c_limit/02_Bugatti-VGT_photo_ext_WEB.jpg" width = 350 height = 200/><img src = "https://github.com/suhasmaddali/Images/blob/main/Car%20Prices%20Prediction%20GitHub%20Image.jpg" width = 350 height = 200/>

<img src = "https://github.com/suhasmaddali/Images/blob/main/Car%20Prices%20Prediction%20GitHub%20Image%202.jpg" width = 350 height = 200/><img src = "https://github.com/suhasmaddali/Images/blob/main/Car%20Prices%20Prediction%20GitHub%20Image%203.jpg" width = 350 height = 200/>

## Business Constraints / Key Performance Metrics (KPIs)

However, one of the challenging aspects of running the sales for cars is to accurately give the __best price__ for cars which ensures that a lot of people buy them and there is a great demand because of this price. Factors that influence the price of cars are __mileage__, __car size__, __manufacturer__, and many others as well. But for humans to comprehensively decide the price is difficult especially when there are a lot of these features that influence the price. One of the solutions to this challenge is to use __machine learning__ and __data science__ to understand insights and make valuable predictions that generate profits for the companies. 

## Machine Learning and Deep Learning

* __Machine Learning__ and __deep learning__ have gained rapid traction in the recent decade. 
* It would be really helpful if we can predict the prices of a car based on a few sets of features such as __horsepower__, __make__ and __other features__. 
* Imagine if a company wants to set the price of a car based on some of the features such as make, horsepower, and mileage. 
* It could do so with the help of machine learning models that would help it to determine the price of a car. 
* This would ensure that the company sets the right amount so that they get the most profits while setting such a price. 
* Therefore, the machine learning models that we would be working with would ensure that the right price is set for new cars which would save a lot of money for car manufacturers respectively.
* We would be working with the car prices prediction data and looking for the predictions of different kinds of cars. 
* We would be first visualizing the data and understanding some of the information that is very important for predictions. 
* We would be using different regression techniques to get the average price of the car under consideration.

<h2> Data Source</h2>

* We would be working with quite a large data which contains about __10000__ data points where again we would be dividing that into the training set and the test set.
* Having a look at some of the cars that we are always excited to use in our daily lives, it is better to understand how these cars are being sold and their average prices. 
* Feel free to take a look at the dataset that was used in the process of predicting the prices of cars. Below is the link.

__Source:__ https://www.kaggle.com/CooperUnion/cardataset

## Metrics

Predicting car prices is a __continuous machine learning problem__. Therefore, the following metrics that are useful for regression problems are taken into account. Below are the __metrics__ that was used in the process of predicting car prices.

* [__Mean Squared Error (MSE)__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html)
* [__Mean Absolute Error (MAE)__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html)

## Exploratory Data Analysis (EDA)

In this section of the project, the data is explored to see the patterns and trends and observe interesting insights. Below are some interesting observations generated.

* A large number of cars were from the manufacturer __'Chevrolet'__ followed by __'Ford'__. 
* The total number of cars manufactured during the year __2015__ was the highest in all the years found on the data.
* There were many missing values for __'Market Category'__ feature and a few missing values for the features __'Engine HP'__ and __'Engine Cylinders'__.
* The average prices of the cars were the highest in the year __2014__ and lowest in the year __1990__ from the data. 
* The prices of __'Bugatti'__ manufacturer are extremely high compared to the other car manufacturers.  
* __'Bugatti'__ manufacturer also had an extremely high value for horsepower (HP) based on the graphs in the notebook.
* There is a __negative correlation__ between the feature __'City Mileage'__ and other features such as __'Engine Cylinders'__ and __'Engine HP'__. This is true because the higher the mileage of the car, there is higher the probability that the total number of cylinders and engine horsepower would be low. 

<h2> Visualizations</h2>

Looking at the dataset, it can be seen that there are categories such as Vehicle Size, city mpg, popularity, and transmission types. There are other features that we would explore in visualizations. 
We will be taking a look at a list of visualizations that can give us an understanding of the data. 
Exploring the count of various car companies available in the dataset, it is seen that Chevrolet has the highest number of cars followed by Ford. 

With the progress in years, there is an increase in the demand for cars sold. This is clearly demonstrated in the visualization. In addition, our ML models would perform well on the most recent cars as we have more data in this category. 

Missingno plots are useful to help us determine the total number of missing values in the dataset. There are missing values in categories such as 'Market Category' and 'Engine HP'. Based on this information, steps are taken to either impute the missing values or remove them so that they do not have an impact on the ML model performance of determining the prices of cars. 


### Model Performance

We will now focus our attention on the performance of __various models__ on the test data. Scatterplots can help us determine how much of a spread our predictions are from the actual values. Let us go over the performance of many ML models used in our problem of car price prediction. 

[__Linear Regression__](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) - After looking at the linear regression plot, it looks like the model is performing quite well. Scatterplots between the predictions and the actual test outputs closely resemble each other. If there are low latency requirements for a deployment setup, linear regression could be used. 


[__Support Vector Regressor__](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html) - Support vector regression (SVR) can be computational. In addition, the results below indicate that the predictions are far off from the actual car prices. Therefore, alternate models can be explored. 


[__K Nearest Regressor__](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html) - K-Nearest Regressor is doing a good job in predicting the car prices as highlighted in the plot below. There is less spread between the test output labels and the predictions generated by the model. Therefore, there are higher chances that the model gives a low mean absolute error and mean squared error. 


[__PLS Regression__](https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html) - This model does a good job overall when it comes to predicting car prices. However, it fails to compare trends and patterns for higher-priced cars well. This is evident due to the fact that there is a lot of spread among higher car price values as shown in the plot. K-Nearest Regressor, on the other hand, also does predictions accurately on higher priced cars. 
 

[__Decision Tree Regressor__](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html) - Based on all the models tested, the decision tree regressor was performing the best. As shown below, there is a lot of overlap between the predictions and the actual test values. 


[__Gradient Boosting Regressor__](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html) - The performance of gradient boosted decision regressor is plotted and it shows that it is quite similar to the decision tree. At prices that are extremely high, the model fails to capture the trend in the data. It does a good job overall. 


[__MLP Regressor__](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html) - It does a good job when it comes to predicting car prices. However, there are better models earlier that we can choose as their performance was better than MLP Regressor in this scenario. 


[__Final Mean Absolute Error (MAE)__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html) - After performing feature engineering and hyperparameter tuning the models, the best model that gave the least mean absolute error (lower is better) was Decision Tree Regressor. Other models such as Support Vector Regressors took a long time to train along with giving less optimum results. Along with good performance, Decision Tree Regressors are highly interpretable and they give a good understanding of how a model gave predictions and which feature was the most important for it to decide car prices. 

<img src = "https://github.com/suhasmaddali/Car-Prices-Prediction/blob/main/images/Final%20MAE.png"/>

[__Final Mean Squared Error (MSE)__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html) - The performance of the Decision Tree Regressor was also the highest when using mean squared error as the output metric. While the Gradient Boosted Regressor came close to the performance of a Decision Tree Regressor, the latter is highly interpretable and easier to deploy in real time. Therefore, we can choose this model for deployment as it is performing consistently across a large variety of metrics. 


## Machine Learning Models 

We have to be using various machine learning models to see which model reduces the __mean absolute error (MAE)__ or __mean squared error (MSE)__ on the cross-validation data respectively. Below are the various machine learning models used. 

| __Machine Learning Models__| __Mean Absolute Error__| __Mean Squared Error__|
| :-:| :-:| :-:|
| [__1. Linear Regression__](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)| 6737| 364527989|
| [__2. Support Vector Regressor__](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)|	22525|	2653742304|
|	[__3. K Nearest Regressor__](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)|	4668|	198923161|
|	[__4. PLS Regression__](https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html)|	6732|	364661296|
|	[__5. Decision Tree Regressor__](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)|	__3327__|	__135789622__|
|	[__6. Gradient Boosting Regressor__](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)|	4432|	175275369|
|	[__7. MLP Regressor__](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html)|	6467|	250908327|

## Outcomes

* The best model that was performing the best in terms of __mean absolute error (MAE)__ and __mean squared error (MSE)__ was __Decision Tree Regressor__.
* __Exploratory Data Analysis (EDA)__ revealed that __Bugatti manufacturer's prices__ were significantly higher than the other manufacturers. 
* __Scatterplots__ between the __actual prices__ and __predicted prices__ were almost __linear__, especially for the __Decision Tree Regressor__ model.

## Future Scope

* It would be great if the best __machine learning model (Decision Tree Regressor)__ is integrated in the live application where a __seller__ is able to add details such as the __manufacturer__, __year of manufacture__ and __engine cylinders__ to determine the best price for cars that generates __large profit margins__ for the seller. 
* Adding __additional data points__ and __features__ could help in also better determining the best prices for the latest cars as well. 


That's it, Thanks. 

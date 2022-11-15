# Customer-Churn
Machine learning classification project which predicts customer chrun

PROJECT TITTLE
VODAFONE CUSTOMER CHURN
PROJECT OBJECTIVE
The objective of this project is to train a machine learning model that will predict whether a customer will churn of not.

Technologies Used: Python (Jupyter Note Book)
Project Status: Completed 

DATA STRUCTURE.
Column Descriptions and Data Field Information
Gender -- Whether the customer is a male or a female
SeniorCitizen -- Whether a customer is a senior citizen or not
Partner -- Whether the customer has a partner or not (Yes, No)
Dependents -- Whether the customer has dependents or not (Yes, No)
Tenure -- Number of months the customer has stayed with the company
Phone Service -- Whether the customer has a phone service or not (Yes, No)
MultipleLines -- Whether the customer has multiple lines or not
InternetService -- Customer's internet service provider (DSL, Fiber Optic, No)
OnlineSecurity -- Whether the customer has online security or not (Yes, No, No Internet)
OnlineBackup -- Whether the customer has online backup or not (Yes, No, No Internet)
DeviceProtection -- Whether the customer has device protection or not (Yes, No, No internet service)
TechSupport -- Whether the customer has tech support or not (Yes, No, No internet)
StreamingTV -- Whether the customer has streaming TV or not (Yes, No, No internet service)
StreamingMovies -- Whether the customer has streaming movies or not (Yes, No, No Internet service)
Contract -- The contract term of the customer (Month-to-Month, One year, Two year)
PaperlessBilling -- Whether the customer has paperless billing or not (Yes, No)
Payment Method -- The customer's payment method (Electronic check, mailed check, Bank transfer(automatic), Credit card(automatic))
MonthlyCharges -- The amount charged to the customer monthly
TotalCharges -- The total amount charged to the customer
Churn -- Whether the customer churned or not (Yes or No)





QUESTIONS ASKED 
1. What is the number of male and female customers.?
2. What is the proportion on male and females who have dependents.?
3. Which internet service customers churn the most?
4. Which type of payment method is used the most?
5. Which payment method churn more?
6. Which type of billing (paperless/paper) churn the most?
7. Which contract type churn the most?

HYPOTHESIS RAISED
1. Loyal customers churn the most
2. Females are more likely to churn out than men.
3. Customers with paper billings are more likely to churn out more.
4. Customers with no Tech Support churn the most.
5. Increased charges results in customer churn.
6. Month-to-Month contract holders churn the most.

















DATA PRECESSING
To begin with the data processing, needed libraries such as numpy, pandas, matplotlip and other libraries from sklearn were imported. The data file was then loaded.
Pandas profiling was used get basic information about the data set such as the description of every column, missing values in every column, data types of every column and basic statistics of every column. Panadas profiling is a very powerful tool if you want to get basic information about your data set briefly.
Some values in the tenure column were zero which indicated that those customers were very new customers and for that matter the company had little or no information about them. It was also relized that those customers with zero tenure also figures allocated to them in the total charges column. This further confirmed the assumption that there was little information about those customers. The total number of those customers were 11, looking at the entire dataset of about 7,000 records, taking out just 11 of them wouldn’t affect the data much. After taking those entries out the total data set came to 7,032. The datatype of the Total charges column was also changed from an object to integer. 
A visualization of pair plot and correlation matrix were made to view the correlation between numeric columns

ANSWERING QUESTIONS ASKED
Q1. What is the number of male and female customers?
From the data set the number of male customers slightly outweighs the females. The table below visualizes the numbers.
MALE	3,549
FEMALES	3,483

Q2. What is the proportion of male and females who have dependents?
The table below visualizes the number of dependents in terms of male and females
 
GENDER		NO. OF DEPENDENTS
MALE	NO	2,460
	YES	1,023
FEMALES	NO	2,473
	YES	1,076

From the table above the it can be seen that in general most customers have no dependents but the number of female with no dependents slightly outweighs that of the males.  Similarly the number of females with dependents also slightly outweighs that of males.


Q3. Which internet service has the most number of customers?

 

From the diagram most customers use fiber optic more than the rest of the options which is DSL. Some customers also do not have internet service at all.

Q4. Which type of payment method is used the most by customers?

 
From the diagram most customers prefer electronic check followed by mailed check, bank transfer and credit card payment being the least preferred type of payment method.

Q5. Which payment method churn more?
This is to determine which type of customes churn more in terms of payment method. 
 
From the chart it can be seen that although most customers use the electronic check method they also churn more compared to the rest of the payment method

Q6. Which customers churn more in terms of type of internet service?
From the chart it can seen that customers who use Fiber optics churn the most than customers who use DSL. Fiber Optics has the most customers so its not good for the company if they also churn the most this means the company is loosing money and most of their customers. 
 

We dived a little deep to find out why fiber optics customers may be churning out more. We compared the types of internet service to other columns such as Monthly charges, Online service, Tech support, Device protection, Online backup and Streaming movies. 
The results is shown in the chart below
 

From the chart above it can be observed that fibre obtics has the highest monthly charges compered to the internet types whict is DSL. It can also be seen that a good number of fibre obtics customers has no online security support. Again it can also be seen that majority of fibre obtics customers has no Device protection. Majority of the fibre obtics customers also do not movie streaming capabilities. Online backup is also not available for most fibre optic customers. 

These factors maybe the reason why fibre optics customers chrun the most than other customers using DSL.

Q7. What is the number of customers with and without paperless billing?
WITH PAPERLESS BILLING	3,549
WITHOUT PAPERLESS BILLING	3,483

From the table above it can be seen that majority of the customers use paperless billing.

Q8. Which contract type churn the most?

 
From the chart above it can be seen that most customers are subscribed to 'Month-to-month' contracts which is 55% of the entire customers, followed by Two years contract with 24% and one year contract with 21%.



ANSWERING THE HYPOTHESIS STATEMENT
1.	Customers that have been with the company for a long time churn the most
To answer this, a histogram plot is made with ‘tenure’ and ‘churn’ columns. This shows which type of customers churn the most, whether the new or old customers.
 

From the above chart it can be seen that loyal customers ( thus customers that have stayed longer with the company ) churn out less than new customers. This suggest that new customers are more likely to churn.










2.	Females are more likely to churn than men
 

From the chart above, female churn slightly more than males.

3.	Customers with paper billing are more likely to churn more
To get an answer for this hypothesis, the Paperless billing was aggregated with the churn column to and sorted out in values to get the number of customers that churned in terms of paper and paperless billing.
PAPERLESS BILLING	CHURN	NO. OF CUSTOMERS
NO	NO	2,395
	YES	469
YES	NO	2,768
	YES	1,400

From the above chart the customers with paperless billing churn the more than customers without paperless billing. The business should take a critical look at their paperless billing and fix any issues if any, this will help reduce customer churn in terms of billing challenges

4.	Customers with no Tech Support churn the most.
To answer this, a histogram plot is made with ‘TechSupport’ and ‘churn’ columns. This shows which type of customers churn the most, whether those with tech support or not.
 

From the above chart it can be seen that customers with no Tech support churn the most then customers with Tech support. A suggestion to fix this issue is to make sure Tech support is extended to all customers

5.	Increased charges result in customer churn.
To answer this, a histogram plot is made with ‘Total Charges’ and ‘churn’ columns. This shows which type of customers churn the most, whether customers with more charges or less charges.

 
Surprisingly, from the above chart, High charges does not translate into high customer churn. Most customers churning out of the business pay low charges, It can also be seen that majority of the company's customers belong to the group that pay low charges. With this out come it calls for more investigation on why low-paying customers churn the most.

6.	Month to Month contract holders churn the most
To answer this, a histogram plot is made with ‘Contract’ and ‘churn’ columns. This shows which type of customers churn the most, whether customers with month-to-month contract, one year contract or two years contract.
 

From the above chart it can be seen that customers that subscribe to "Month-to-Month" contracts churn the most followed by "one-year" contract holders and lastly "two-year" contract holders churn the least.
We dive further to investigate why month-to-month contract holders churn the most.
 

From the chart it can be seen that Month-to-Month contract holders has the highest monthly charges compared to other contract holders. Most of Month-to-Month contract customers also have no Online security. Again majority of Month-to-Month contract customers have no Tech support, Device protection and no movie streaming capabilities.
These factors may be the reasons why Month-to-Month customers churn the most compared other contract holders such are one year and two years contract. The company should fix some of this things in other to reduce Month-to-Month contract customers churn.











PREPARING THE DATA SET FOR MODELING
To get the data ready for modeling, the customer ID column had to be dropped since it will not be needed for the machine learning process. The non numeric columns also need to be encoded to numeric values. 


Label Encoding
Label encoder was used for columns with only two values in them such as gender, Senior Citizen, partner, Dependents, Phone service, and paperless billing. This will transform the categorical values into zero’s and one’s.
OneHotEncoding
OneHotEncoding was used for the categorical columns with more than one values. Columns include Multiplelines, InternetService, OnlineSecurity, OnlineBackup, DevicePretection, TechSupport, StreamingTV, StreamingMovies, Contract and Paymentmethod.

MinMax scaler we also used to scale the tenure, Monthly charged and total charges columns.

Splitting the data into Train and Test data
For machine learning modeling the data set has to be split into Train and Test dataset . The Train data set is used to train the model and the test data set is used to test the model. 
Out data set was split into 80% for train and 20% for test .
Balancing of the data set
To begain the training models the set needs to be balanced. Since the number of churned customers are far lower than the customers who did not churn, we would have balance the data so that our models mimic the same patten during the training face of the modeling.
To balance off the data, the SMOTE method was used. The balancing of the data was done only of the train data set.



TRAINING THE MODELS
Random Forest
The first model to be trained was the random forest model. It came out with the following scores and a confusion matrix.
        Model.  Accuracy.  Precision.    Recall  F1 Score  F2 Score
0.  Random Forest  0.780384   0.566085  0.627072   0.59502  0.591844
 
The confusion matrix for the random forest did well predicting the true positive which is customers that actually churned . It also did fairly well in predicting the true negatives which is customers that actually did not churn. From the matrix the model did not perform well in predicting the false negative and false positive. It predicted a high number of false negative and false positive which is not good for the prediction.
Logistic Regression
The next model to trained was the logistic regression. It also came out with following scores and confusion matrix.
                Model  Accuracy  Precision    Recall  F1 Score  F2 Score
0        Random Forest  0.780384   0.566085  0.627072  0.595020  0.591844
1  Logictic Regression  0.761194   0.523810  0.790055  0.629956  0.717151
 

From the scores it can be seen that the logistic regression performed slightly better than the random forest model.
The confusion matrix did well predicting the true positive which is customers that actually churned . It also did better than the random forest in predicting the true negatives which is customers that actually did not churn. From the matrix the model did not perform well in predicting the false positive It predicted a high number of false positive, but it did well in predicting the false negative, it have very low numbers in that section.


Naïve Bayes
The naïve bayes model was also train and came out with the scores and confusion matrix as follows.
                 Model  Accuracy  Precision    Recall  F1 Score  F2 Score
0        Random Forest  0.780384   0.566085  0.627072  0.595020  0.591844
1  Logictic Regression  0.761194   0.523810  0.790055  0.629956  0.717151
2          Naive Bayes  0.682303   0.440393  0.867403  0.584186  0.726516


 
The confusion matrix for the Naïve bayes also performed well in predicting the true positive which is customers that actually churned . It also did better than the random forest  and Logistic regression in predicting the true negatives which is customers that actually did not churn. From the matrix the model performed poorly in predicting the false positive It predicted a high number of false positive, but also performed well in predicting the false negative, it have very low numbers in that section.

Catboost Model
The catboost model also had the following scores and confusion matrix.

 
The confusion matrix for the catboost model performed well in predicting the true positive. But it performed poorly in making predictions for the rest which are the true negative , false positive and false negative.

Decision Tree model
The scores for Decision tree models are as follows
     Model  Accuracy  Precision    Recall  F1 Score  F2 Score
0        Random Forest  0.780384   0.566085  0.627072  0.595020  0.591844
1  Logictic Regression  0.761194   0.523810  0.790055  0.629956  0.717151
2          Naive Bayes  0.682303   0.440393  0.867403  0.584186  0.726516
3            Cat Boost  0.777541   0.561097  0.621547  0.589777  0.608437
4        Decision Tree  0.725657   0.469849  0.516575  0.492105  0.506501
 
The confusion matrix for the Decision tree also did well predicting the true positive which is customers that actually churned . It also did fairly well in predicting the true negatives which is customers that actually did not churn. From the matrix the model did not perform well in predicting the false negative and false positive especially the false negetive. It predicted a high number of false negative and false positive which is not good for the prediction.

Gradient Boosting
Below are the scores and confusion matrix for Gradient boosting.
Model  Accuracy  Precision    Recall  F1 Score  F2 Score
0        Random Forest  0.780384   0.566085  0.627072  0.595020  0.591844
1  Logictic Regression  0.761194   0.523810  0.790055  0.629956  0.717151
2          Naive Bayes  0.682303   0.440393  0.867403  0.584186  0.726516
3            Cat Boost  0.777541   0.561097  0.621547  0.589777  0.608437
4        Decision Tree  0.725657   0.469849  0.516575  0.492105  0.506501
5       Gradient Boost  0.768301   0.546154  0.588398  0.566489  0.579434

 
The confusion matrix for the Gradient boost did well predicting the true positive which is customers that actually churned . It also did fairly well in predicting the true negatives which is customers that actually did not churn. From the matrix the model did not perform well in predicting the false negative and false positive. It predicted a high number of false negative and false positive which is not good for the prediction. However it did slightly better in predicting the false negative.

HYPERPAMETER TUNING
Performing Hyperparameter tuning with RandomSearch CV on the Random forest model.
Hyperparameter tuning consists of finding a set of optimal hyperparameter values for a learning algorithm while applying this optimized algorithm to any data set. That combination of hyperparameters maximizes the model’s performance, minimizing a predefined loss function to produce better results with fewer errors. 

There are several ways to achieve hyperparameter tuning, the method that will be used here is the Random Search CV.
After performing the hyperpameter tuning on the random forest model it had the following scores and confusion matrix.
     Model  Accuracy  Precision    Recall  F1 Score  F2 Score
0        Random Forest  0.780384   0.566085  0.627072  0.595020  0.591844
1  Logictic Regression  0.761194   0.523810  0.790055  0.629956  0.717151
2          Naive Bayes  0.682303   0.440393  0.867403  0.584186  0.726516
3            Cat Boost  0.777541   0.561097  0.621547  0.589777  0.608437
4        Decision Tree  0.725657   0.469849  0.516575  0.492105  0.506501
5       Gradient Boost  0.768301   0.546154  0.588398  0.566489  0.579434
6   Random Forest (HP)  0.768301   0.537815  0.707182  0.610979  0.665281

 


From the results , it can be seen that the model with the ideal hyperparameters performed slightly better than the first trained Random forest model. The Accuracy, Recall, F1 and F2 scores improved, with the exception of the precision score.


Performing Hyperparameter tuning with GridSearchCV on the CatBoost model.
Grid search is a sort of “brute force” hyperparameter tuning method. We create a grid of possible discrete hyperparameter values then fit the model with every possible combination. We record the model performance for each set then select the combination that has produced the best performance.
    Model  Accuracy  Precision    Recall  F1 Score  F2 Score
0        Random Forest  0.780384   0.566085  0.627072  0.595020  0.591844
1  Logictic Regression  0.761194   0.523810  0.790055  0.629956  0.717151
2          Naive Bayes  0.682303   0.440393  0.867403  0.584186  0.726516
3            Cat Boost  0.777541   0.561097  0.621547  0.589777  0.608437
4        Decision Tree  0.725657   0.469849  0.516575  0.492105  0.506501
5       Gradient Boost  0.768301   0.546154  0.588398  0.566489  0.579434
6   Random Forest (HP)  0.768301   0.537815  0.707182  0.610979  0.665281
7       Cat Boost (HP)  0.768301   0.540000  0.671271  0.598522  0.640148

 


From the results , it can be seen that the model with the ideal hyperparameters performed slightly better than the first trained Random forest model. The Accuracy, Recall, F1 and F2 scores improved, with the exception of the precision score.

Performing Hyperparameter tuning with GridSearchCV on the Decision Tree model.
After performing the hyperparameter tuning for the Decision tree model , the scores and confusion matrix are as follows.
                Model  Accuracy  Precision    Recall  F1 Score  F2 Score
0        Random Forest  0.780384   0.566085  0.627072  0.595020  0.591844
1  Logictic Regression  0.761194   0.523810  0.790055  0.629956  0.717151
2          Naive Bayes  0.682303   0.440393  0.867403  0.584186  0.726516
3            Cat Boost  0.777541   0.561097  0.621547  0.589777  0.608437
4        Decision Tree  0.725657   0.469849  0.516575  0.492105  0.506501
5       Gradient Boost  0.768301   0.546154  0.588398  0.566489  0.579434
6   Random Forest (HP)  0.768301   0.537815  0.707182  0.610979  0.665281
7       Cat Boost (HP)  0.768301   0.540000  0.671271  0.598522  0.640148
8   Decision Tree (HP)  0.755508   0.520737  0.624309  0.567839  0.600425

 

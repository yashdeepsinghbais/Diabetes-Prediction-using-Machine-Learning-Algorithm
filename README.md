@author - YashDeepSinghBais
This is the code source for prediciting diabetes using several parameter
I am using logistisc regression machine learning algorithm for this.
What is logistic regression?
Logistic regression estimates the probability of an event occurring, such as voted or didn’t vote, based on a given data set of independent variables.

This type of statistical model (also known as logit model) is often used for classification and predictive analytics. Since the outcome is a probability, the dependent variable is bounded between 0 and 1. In logistic regression, a logit transformation is applied on the odds—that is, the probability of success divided by the probability of failure. This is also commonly known as the log odds, or the natural logarithm of odds, and this logistic function is represented by the following formulas: 

Logit(pi) = 1/(1+ exp(-pi))

ln(pi/(1-pi)) = Beta_0 + Beta_1*X_1 + … + B_k*K_k

In this logistic regression equation, logit(pi) is the dependent or response variable and x is the independent variable. The beta parameter, or coefficient, in this model is commonly estimated via maximum likelihood estimation (MLE). This method tests different values of beta through multiple iterations to optimize for the best fit of log odds. All of these iterations produce the log likelihood function, and logistic regression seeks to maximize this function to find the best parameter estimate. Once the optimal coefficient (or coefficients if there is more than one independent variable) is found, the conditional probabilities for each observation can be calculated, logged, and summed together to yield a predicted probability. For binary classification, a probability less than .5 will predict 0 while a probability greater than 0 will predict 1. 

Here’s a brief overview of the libraries i am using:

1. **Pandas**: Used for data manipulation and analysis. It provides data structures like DataFrames, which are perfect for handling datasets.
   
2. **NumPy**: A library for numerical computations in Python. It supports large, multi-dimensional arrays and matrices, and comes with mathematical functions to operate on them.

3. **Matplotlib**: A plotting library used for creating static, animated, and interactive visualizations in Python.

4. **Seaborn**: Built on top of Matplotlib, Seaborn provides a high-level interface for creating attractive and informative statistical graphics.

5. **Sklearn (scikit-learn)**:
   - **train_test_split**: Splits your dataset into training and testing sets.
   - **StandardScaler**: Standardizes features by removing the mean and scaling to unit variance.
   - **LogisticRegression**: A model used for binary classification tasks.
   - **accuracy_score**: Computes the accuracy of the model.
   - **confusion_matrix**: Displays the performance of the classification model.
   - **classification_report**: Provides precision, recall, F1-score, and support for the classification model. 

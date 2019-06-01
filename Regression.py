import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn as skl
import statsmodels.api as sm
import warnings
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
from numpy import array, diag, dot, maximum, empty, repeat, ones, sum
from numpy.linalg import inv


def loadCSVData(location):
    return pd.read_csv(location, sep=',', header=[0])


def dataframeStats(inputDF):
    print("\nInput and Output Columns Statistics:")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(inputDF.describe())


def runLinearRegression(trnX, trnY, tstX, tstY):
    model = LinearRegression()
    print("\nTraining the model.")
    model.fit(trnX, trnY)
    print("\nTesting the model:")
    print('R-Squared value = ', model.score(tstX, tstY), '%')
    prdctdY = model.predict(tstX)
    print('MSE = ', mean_squared_error(tstY, prdctdY))
    print('RMSE = ', math.sqrt(mean_squared_error(tstY, prdctdY)))


def runPolynomialRegression(trnX, trnY, tstX, tstY):
    # Initialise Polynomial Regression parameters
    trainXSecondOrder = polynomials.fit_transform(trnX)
    testXSecondOrder = polynomials.fit_transform(tstX)
    runLinearRegression(trainXSecondOrder, trnY, testXSecondOrder, tstY)


def runLassoRegression(trnX, trnY, tstX, tstY, alpha):

    lasso = Lasso(alpha=alpha)
    lasso.fit(trnX, trnY)
    print('R-Squared value = ', lasso.score(tstX, tstY), '%')
    prdctdY = lasso.predict(tstX)
    print('MSE = ', mean_squared_error(tstY, prdctdY))
    print('RMSE = ', math.sqrt(mean_squared_error(tstY, prdctdY)))
    print('Coeffecients', lasso.coef_)


def runRidgeRegression(trnX, trnY, tstX, tstY, alpha):

    ridge = Ridge(alpha=alpha)
    ridge.fit(trnX, trnY)
    print('R-Squared value = ', ridge.score(tstX, tstY), '%')
    prdctdY = ridge.predict(tstX)
    print('MSE = ', mean_squared_error(tstY, prdctdY))
    print('RMSE = ', math.sqrt(mean_squared_error(tstY, prdctdY)))
    print('Coeffecients', ridge.coef_)


def forward_selected(inputdata, output):

    remaining = set(inputdata.columns)
    remaining.remove(output)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(output, ' + '.join(selected + [candidate]))
            score = smf.ols(formula, inputdata).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    formula = "{} ~ {} + 1".format(response, ' + '.join(selected))
    model = smf.ols(formula, inputdata).fit()
    return model


print("\nBeginning of Program Execution:")

# Load Data from CSV file
data = loadCSVData('data.csv')
inputColumnNames = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']
outputColumnNames = ['Y1', 'Y2']
columnNames = inputColumnNames + outputColumnNames

print("\nData file loaded.")
# Split data into train, test and validation sets
print("\nSplitting data into train, test and validation sets:")
train, test = train_test_split(data, test_size=0.3, random_state=123)
testV, validation = train_test_split(test, test_size=0.33)
print("Number of Training Sentences =   {}({}%)".format(len(train), round(len(train)/len(data)*100)))
print("Number of Testing Sentences =    {}({}%)".format(len(test), round(len(test)/len(data)*100)))
print("Number of Validation Sentences = {}({}%)".format(len(validation), round(len(validation)/len(data)*100)))

# Printing input and output column specifications
dataframeStats(train)

# Split the inputs and outputs
traininput = train.loc[:, 'X1':'X8']
trainoutput = train.loc[:, 'Y1':'Y2']
testinput = train.loc[:, 'X1':'X8']
testoutput = train.loc[:, 'Y1':'Y2']

# Scaling the inputs
print("\nScaling the inputs due to range differences in inputs:")
traininput_scaled = skl.preprocessing.normalize(traininput,)
train_scaled = np.append(traininput_scaled, np.array(trainoutput), axis=1)
train_scaled = pd.DataFrame(train_scaled, columns=columnNames)
testinput_scaled = skl.preprocessing.normalize(testinput,)
test_scaled = np.append(testinput_scaled, np.array(trainoutput), axis=1)
test_scaled = pd.DataFrame(test_scaled, columns=columnNames)

# Printing input and output column specifications
dataframeStats(train_scaled)

# Plotting Scatter plots between inputs and outputs
sns.pairplot(train_scaled, x_vars=['X1', 'X2', 'X3', 'X4'], y_vars='Y1')
sns.pairplot(train_scaled, x_vars=['X5', 'X6', 'X7', 'X8'], y_vars='Y1')
sns.pairplot(train_scaled, x_vars=['X1', 'X2', 'X3', 'X4'], y_vars='Y2')
sns.pairplot(train_scaled, x_vars=['X5', 'X6', 'X7', 'X8'], y_vars='Y2')


# Plot correlation matrix of features
size = 10
corr = train_scaled.corr()
fig, ax = plt.subplots(figsize=(size, size))
sns.heatmap(corr, vmax=1.0, center=0, fmt='.2f', square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70})
plt.show()

# Split the scaled inputs and outputs
trainX = train_scaled.loc[:, 'X1':'X8']
trainY = train_scaled.loc[:, 'Y1':'Y2']
testX = test_scaled.loc[:, 'X1':'X8']
testY = test_scaled.loc[:, 'Y1':'Y2']

# Run the Linear Regression Model
print("\nBegin the Linear Regression.")
runLinearRegression(trainX, trainY, testX, testY)

# Run the Polynomial Regression Model
print("\nBegin the Polynomial Regression.")
polynomials = skl.preprocessing.PolynomialFeatures(2)
runPolynomialRegression(trainX, trainY, testX, testY)

# Run the Lasso Regression Model
print("\nBegin the Lasso Regression.")
alpha = 0.00005
runLassoRegression(trainX, trainY, testX, testY, alpha)

# Run the Ridge Regression Model
print("\nBegin the Ridge Regression.")
alpha = 0.0000001
runRidgeRegression(trainX, trainY, testX, testY, alpha)

# Forward Selection of Features
# dataY1 = train.loc[:,['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7','X8', 'Y1']]
# model = forward_selected(dataY1,"Y1")
# print(model.model.formula)
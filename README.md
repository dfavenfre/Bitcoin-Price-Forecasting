# Bitcoin-Price-Forecasting
The Bitcoin Price Forecasting project aims to predict the price movements of Bitcoin by utilizing statistical methods and time-series modeling techniques. The project applies Ordinary Least Squares (OLS) regression and ARIMA/SARIMAX models to explore the relationships between various predictor variables such as volume, open interest (OI), and funding rate (FR) with Bitcoin price. Through the use of stationarity tests, normality checks, and autocorrelation analysis, the project evaluates the statistical significance of the variables and refines the forecasting model. The ultimate goal is to improve the accuracy of Bitcoin price predictions by identifying key factors and ensuring model stability over time.

## Normality Test
In accordance with the results of Shapiro-Wilk normality tests, it is concluded that neither
of the variables has a p value greater than 0.05. Thus, the null hypothesis is rejected and the test
concludes the variables are not normally distributed, since the null hypothesis of the
Shapiro-Wilk test suggests that the p value of the test parameter has to be greater than 0.05 for
the variable to be retained as normally distributed.
![image](https://github.com/user-attachments/assets/24080959-9e29-4d6a-9f57-45290925803d)

## Stationarity
As to stationarity assumptions, the below graph infers the following variables, such as
Price, OI and LSR, to be trending in direction with their mean and variances subject to change
with time. Contrary to that, Volume and FR variables appear to be stationary since there is no
sign of trend in any direction or an indication that mean or variances change with time.
![image](https://github.com/user-attachments/assets/1aad9d9b-bd4a-4769-afdf-6a82db921b50)

Regardless of the preliminary assumptions, Augmented Dickey-Fuller test has been
applied to conclude the stationarity test for the variables.

Augmented Dickey-Fuller test suggests that, as the null hypothesis, the p value of the
parameter should be lower than 0.05 to be retained as stationary. If otherwise, the parameter is
non-stationary.
As a consideration for the ADF test results, LSR, OI and Close variables are
non-stationary since their p value is greater than 0.05, and the null hypothesis is thus not
rejected. On the other hand, Volume and FR variables conclude a p value less than 0.05, which
suggests the null hypothesis to be rejected.
To answer the research question of whether the aforementioned predictor variables can
explain the price action of Bitcoin, a multivariate OLS model has been constructed.

![image](https://github.com/user-attachments/assets/7054a5f9-458e-4c21-b9d8-2b8bd4660d8b)

# Prediction with OLS Regression
![image](https://github.com/user-attachments/assets/0a8b63f7-78f6-43ac-9524-474c961216ed)

In accordance with the OLS results, the model puts forth that the predictor variables were
indeed adequate to explain the changes in the price of Bitcoin. However, “Volume” variable did
not yield a statistically significant result (p>0.05) and therefore it has been removed from the
model to capture the overall success of the statistically significant variables.
As to the conclusion, the null hypothesis has been rejected. The results also concluded
that the model had a success 77.6% at explaining the response variable. Further, the model also
concluded that LSR and OI variables had a negative relationship with the changes in the price of
Bitcoin.
However, the predictor variables might have failed at capturing the true potential of the
prediction due to the fact that the variables were not normally distributed and some variables
were indicating non-stationary behavior.

## Q-Q Plot
For this reason, the Q-Q plot of the residuals of the developed OLS model and the
distribution of error terms of the model were further analyzed to conclude whether the model
was a good-fit or not.

The distribution of the residual values indicate that the residuals are closely following the
theoretical line, indicating that they are distributed normally, but some values appear to be
deviating from the line.

![image](https://github.com/user-attachments/assets/ee0b59e0-3775-44e1-80f0-145d7bc1fe40)

## Error Term Distribution 
As a consideration for the distribution of error terms, addition to figure.5 Shapiro-Wilk
test had been applied to the error term values and the test concluded that the distribution had a p
value less than 0.05, meaning the distribution is not normally distributed.

![image](https://github.com/user-attachments/assets/02f18af6-b221-498b-b31d-4067ed90d095)

## Evaluations 
![image](https://github.com/user-attachments/assets/3bfbd443-a6d0-485f-85fe-d3749f09b401)

As can be seen from the figure.6, the predicted values are somehow good at capturing the
price actions whenever the volatility (standard deviation of the price of Bitcoin) flattens or slows
down. However, during high volatility the model fails to capture the price action. This is mainly
due to the fact that some of the predictor variables are non-stationary and not normally
distributed.

## Time-series Forecasting With SARIMAX
To attain a model that is somewhat better at capturing a good-fit, an ARIMA
model with the order of p=5, d=2, and q=0 has been constructed. First, a stationarity test has
been conducted and “close” variable resulted as non-stationary before applying differencing of 2.

### Price of Bitcoin (Timeseries Covering Between Jan2 - Jan3 (Hourly) 2023)
![image](https://github.com/user-attachments/assets/c8ebd403-e20c-456f-be74-8038c221e56b)

### ACF & PACF Plots (No Differencing)
![image](https://github.com/user-attachments/assets/431ca237-15f1-4314-b415-0de64a7ae3a9)

Second, the autocorrelation and partial autocorrelation functions of the “close” variable
have been graphed to understand the relationship of phi value with the variable.

### Price of Bitcoin after differencing of Dif(1) 
![image](https://github.com/user-attachments/assets/d2468c5f-db2d-4b3e-9dc5-d8d8ce88723c)

### ACF & PACF Plots (with Differencing of 1)
![image](https://github.com/user-attachments/assets/4acadda3-8a31-426c-b8bd-188b68a9149d)

The PACF suggests that after lag-2 the variable begins yielding not statistically
significant autocorrelations, with a positive sign of phi. A positive phi value indicates that Rt
values are positively affected by Rt-1 values. Therefore this preliminary PACF suggests that
there is a momentum between Rt-1 and Rt. It can also be deduced that the Rt-1 close value with
higher close value than Rt-2, is likely to open the hourly candle of Rt with an increase.

Further, to capture the best, in other words, the AIC (Akaike Information Criterion) with
less errors, a series of combinations have been conducted and an ARIMA model with order of
p=5, d=2, and q=0 has been concluded as the best, or with less AIC errors, model.

![image](https://github.com/user-attachments/assets/96f2c9d4-6610-4695-a77b-6d2c89ed5c77)

The test results for the developed ARIMA model further yield that all AR(5) are
concluded as statistically significant, along with Ljung-Box (Prob Q) parameter being higher
than 0.05, meaning the residuals of the model are not correlated with one another. Further,
Jarque-Bera parameter also suggests that the residuals are not normally distributed, with a p
value less than 0.05.

![image](https://github.com/user-attachments/assets/e5a3f553-a24e-4ca4-9680-f33549fc87e5)

The figure above (Figure.12) provides further diagnostics for the model fitness with the
residuals modeled. As concluded with Prob-Q and JB parameter results, the residuals are not
distributed normally and are not correlated with one another. Therefore, the model still has a
window for improvements. ARIMA with exogenous variables can be an improvement to begin
with.
Lastly, both trained data are populated to test out how well it fits on the observed data.
Evidently, the trained data fits well in overall, however, there appears to be some deviations at
wherever the standard deviation is high.

![image](https://github.com/user-attachments/assets/8ed1a4b2-d292-418d-98d4-b215c12bec98)

In a consideration for the conclusion, the hypothesized research question that the
aforementioned explanatory variables are adequate to explain the response variable (Price of
Bitcoin) seem to be reasonable with a room for improvements, probably with the help of
exogenous variables for ARIMAX, or with different independent variable combinations that are
somehow better at explaining the response variable.


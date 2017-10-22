# stock-forecasting
# Stock closing and opening forecasting using Deep neural network and LSTM(technical indicators included)
Details about the indicators are here https://github.com/saradindusengupta/technical_indicators_stock-market

The two py files 
```
stock-forecast-lstm.py
stock-forecast-tweet.py
```
are for forecasting stock opening and closing prices from twitter and NYtimes  using deep neural network and lstm.
The notebook directory contains the results and ipynb files.

# Results
![Alt text](![Alt text](https://github.com/BenjiKCF/Neural-Network-with-Financial-Time-Series-Data/blob/master/Photos/20170510result.png)

*Train Score: 0.00006 MSE (0.01 RMSE)

*Test Score: 0.00029 MSE (0.02 RMSE)

# Future Improvements :
1. Include more technical indicators from https://github.com/saradindusengupta/technical_indicators_stock-market
2. Use tweets for sentiment analysis more effectively 
3. More data 
4. More Indexes and better optimized hyperparameter

# References:
1. Bernal, A., Fok, S., & Pidaparthi, R. (2012). Financial Market Time Series Prediction with Recurrent Neural Networks.
2. A hybrid SOFM-SVR with a filter-based feature selection for stock market forecasting CL Huang, CY Tsai - Expert Systems with Applications, 2009 - Elsevier
3.Twitter mood predicts the stock market J Bollen, H Mao, X Zeng - Journal of computational science, 2011 - Elsevier
4.Evaluating the impact of technical indicators on stock forecasting  IEEE
5.A hybrid stock trading framework integrating technical analysis with machine learning techniques  Rajashree Dash Pradipta Kishore Dash

# Dependencies :
Language - Python 3.5

keras : https://keras.io/

tensorflow : https://www.tensorflow.org/

sklearn: http://scikit-learn.org/stable

numpy : http://www.numpy.org/

pickle: https://docs.python.org/2/library/pickle.html


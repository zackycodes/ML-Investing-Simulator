# 📈 ML Investing Market
Simulation & Predictive Analysis FrameworkWelcome to MLInvesting,
a Python-based sandbox designed to simulate market volatility and evaluate the efficacy of Machine Learning in trading strategies. The core of this project lies in dlinvesting.py, which pits a standard "buy-and-hold" logic against a LinearRegression model to see who comes out on top.

## 🚀 Key Features
Dynamic Market Engine: Simulates stochastic price movements using a configurable fluctuation range. 
### Dual-Strategy Analysis:
* __The Traditionalist__: A baseline investor that maintains a consistent market presence regardless of conditions.
* __The ML Strategist__: Powered by scikit-learn, this investor trains on historical data to predict future movements and only enters the market when the forecast is favorable.
* __Comprehensive Reporting__: Detailed daily logs and a final summary of total investment vs. accumulated profit. 

## 🛠️ Technical Architecture
### The logic within dllinvesting.py is broken down into four primary classes:
* __Market__: The environment. It generates the "noise" (fluctuations) that the investors must navigate.  
* __Investor__: The base logic. Manages the wallet, tracks investment history, and calculates P&L based on margins and loss limits.  
* __MachineLearningInvestor__: The "brains." Extends the base investor by adding a LinearRegression model to predict market direction.  
* __Simulator__: The orchestrator. Runs the loop for a specified number of days and handles the data output.  

## 💻 Getting Started
### Prerequisites
You'll need Python 3.x and the standard data science stack:  
`Bashpip install numpy scikit-learn`
### Execution
To run the default simulation (10,000 days of market data), simply execute the main script:
`Bashpython dlinvesting.py`

# Notes
This is a simulation tool for educational purposes. Don't let a LinearRegression model handle your life savings just yet!

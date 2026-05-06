import random
import numpy as np
from sklearn.linear_model import LinearRegression


class Market:
    def __init__(self, fluctuation_range=(-2, 2)):
        self.fluctuation_range = fluctuation_range

    def get_fluctuation(self):
        return random.uniform(*self.fluctuation_range)


class Investor:
    def __init__(self, initial_investment, profit_margin, loss_limit):
        self.initial_investment = initial_investment
        self.profit_margin = profit_margin
        self.loss_limit = loss_limit
        self.total_investment = 0
        self.profit = 0
        self.investment_history = []

    def invest(self, market_fluctuation):
        investment_amount = self.initial_investment
        daily_profit = 0
        daily_loss = 0

        if market_fluctuation > 0:
            # Profit day
            daily_profit = investment_amount * self.profit_margin
            self.profit += daily_profit
        else:
            # Loss day
            daily_loss = investment_amount * self.loss_limit
            self.profit -= daily_loss

        self.total_investment += investment_amount
        self.investment_history.append(investment_amount)

        return daily_profit, daily_loss, investment_amount


class Simulator:
    def __init__(self, market, investor, days):
        self.market = market
        self.investor = investor
        self.days = days

    def run_simulation(self):
        for day in range(1, self.days + 1):
            market_fluctuation = self.market.get_fluctuation()
            daily_profit, daily_loss, investment_amount = self.investor.invest(
                market_fluctuation
            )

            print(
                f"Day {day}: Market fluctuation {market_fluctuation:.2f} - Profit of ${daily_profit:.2f}, Loss of ${daily_loss:.2f}. Invested ${investment_amount:.2f}."
            )

        print("\nSimulation completed.")
        print(f"Total investment: ${self.investor.total_investment:.2f}")
        print(f"Accumulated profit: ${self.investor.profit:.2f}")


class MachineLearningInvestor(Investor):
    def __init__(
        self, initial_investment, profit_margin, loss_limit, training_days
    ):
        super().__init__(initial_investment, profit_margin, loss_limit)
        self.training_days = training_days
        self.model = None

    def train_model(self, market, days):
        profit_loss_list = []
        for _ in range(days):
            fluctuation = market.get_fluctuation()
            profit_loss = self.initial_investment * fluctuation
            profit_loss_list.append(profit_loss)

        X = np.array(range(days)).reshape(-1, 1)
        y = np.array(profit_loss_list)

        # Train a linear regression model
        model = LinearRegression()
        model.fit(X, y)
        self.model = model

    def invest_with_model(self, market_fluctuation):
        if self.model is None:
            raise Exception("Model not trained yet!")

        predicted_fluctuation = self.model.predict([[market_fluctuation]])[0]
        if predicted_fluctuation > 0:
            # Invest if model predicts profit
            profit_loss = self.initial_investment * random.uniform(0, 2)
        else:
            # Don't invest if model predicts loss
            profit_loss = 0

        return super().invest(profit_loss)


if __name__ == "__main__":
    # Investment parameters
    investment_per_day = 5
    days = 10000
    profit_margin = 0.1
    loss_limit = 0.1

    # Market simulation
    market = Market()

    # Investor without model
    investor = Investor(investment_per_day, profit_margin, loss_limit)
    simulator = Simulator(market, investor, days)
    simulator.run_simulation()

    # Investor with model
    training_days = 1000
    ml_investor = MachineLearningInvestor(
        investment
        )
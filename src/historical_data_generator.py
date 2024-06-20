import numpy as np
import pandas as pd

class HistoricalDataGenerator:
    """Class to generate random timestamp to price historical data"""

    def __init__(self):
        self._max_price = 2
        self._rng = np.random.default_rng(seed=42)

    def _generate_random_timestamps(self, start_date, end_date, n):
        """Generate random timestamps over the provided period."""

        start_u = start_date.value // 10**9
        end_u = end_date.value // 10**9
        return pd.to_datetime(self._rng.integers(start_u, end_u, n), unit='s').sort_values()

    def _generate_data(self, start_date: pd.Timestamp, end_date: pd.Timestamp, frequency: str = 'T'):
        """Generates synthetic data for the specified period."""

        # Generate timestamps for the specified period
        n = len(pd.date_range(start=start_date, end=end_date, freq=frequency))
        timestamps = self._generate_random_timestamps(start_date, end_date, n)

        # Generate synthetic prices
        sine_wave = np.sin(np.linspace(0, 100, len(timestamps)))
        noise = self._rng.normal(scale=0.5, size=len(timestamps))
        prices = sine_wave + noise + 1

        # Normalize prices to be within the range of 0 to MAX_PRICE
        price_range = prices.max() - prices.min()
        if price_range == 0:
            prices = np.full_like(prices, self._max_price / 2)
        else:
            prices = (prices - prices.min()) / price_range * self._max_price
            prices = np.round(prices, 2)

        return pd.DataFrame({'timestamp': timestamps, 'price': prices})

    def generate_historical_data(self):
        """Generates synthetic historical data for the latest 3 months."""

        # Generate timestamps for the latest 3 months
        end_date = pd.Timestamp.now() - pd.DateOffset(minutes=5)
        start_date = end_date - pd.DateOffset(months=3)

        return self._generate_data(start_date, end_date, frequency='H')

    def generate_recent_data(self):
        """Generates synthetic recent data for the last 5 minutes."""

        # Generate timestamps for the latest 5 minutes
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.DateOffset(minutes=5)

        return self._generate_data(start_date, end_date, frequency='min')

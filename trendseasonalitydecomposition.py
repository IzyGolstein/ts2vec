import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
import pandas as pd

class TrendSeasonalityDecomposition():
    """
    Инициализация класса TrendSeasonalityDecomposition.

    Параметры:
    frequency (str): Частота временного ряда (например, 'H' для часовой частоты).
    seasonality (int): Период сезонности.
    trend_type (str): Тип тренда ('add' для аддитивного тренда).
    seasonality_type (str): Тип сезонности ('add' для аддитивной сезонности).
    error_type (str): Тип ошибки ('add' для аддитивной ошибки).
    """
    def __init__(self, frequency='H', trend_type='add', seasonality_type='add', error_type='add'):
        self.frequency = frequency
        self.trend_type = trend_type
        self.seasonality_type = seasonality_type
        self.error_type = error_type

    def plot(self, data, columns, lags):
        """
        Построение графиков автокорреляции и частичной автокорреляции.

        Параметры:
        data (DataFrame): Данные временного ряда.
        columns (list): Список колонок для построения графиков.
        lags (int): Количество лагов для отображения.
        """
        for column in columns:
            fig, axes = plt.subplots(2, 1, figsize=(15, 10))

            plot_acf(data[column], ax=axes[0], lags=lags)
            axes[0].set_title(f'Autocorrelation Function (ACF) for {column}')
            axes[0].grid(True)

            plot_pacf(data[column], ax=axes[1], lags=lags)
            axes[1].set_title(f'Partial Autocorrelation Function (PACF) for {column}')
            axes[1].grid(True)

            for ax in axes:
                ax.set_xticks(np.arange(0, lags+1, step=6))

        plt.show()

    def fit(self, data, columns, seasonality = 24):
        """
        Декомпозиция тренда, сезонности и остатка для указанных колонок.

        Параметры:
        data (DataFrame): Данные временного ряда.
        columns (list): Список колонок для декомпозиции.

        Возвращает:
        DataFrame: Таблица с трендом, сезонностью и остатком для каждой колонки.
        """
        trend_dict = {}
        seasonal_dict = {}  
        residual_dict = {}

        for column in columns:
            data_column = data[column].asfreq('H')

            model = ETSModel(data_column, error = self.error_type, trend= self.trend_type, seasonal=self.seasonality_type, seasonal_periods=seasonality).fit() # надо дописать frequency, пока только h

            fitted_values = model.fittedvalues
            residuals = data_column - fitted_values

            trend = model.level
            seasonal = data_column -trend - residuals

            trend_dict[f'trend_{column}'] = trend
            seasonal_dict[f'seasonal_{column}'] = seasonal
            residual_dict[f'residual_{column}'] = residuals

        trend_df = pd.DataFrame(trend_dict, index=data.index)
        seasonal_df = pd.DataFrame(seasonal_dict, index=data.index)
        residual_df = pd.DataFrame(residual_dict, index=data.index)


        result_df = pd.concat([trend_df, seasonal_df, residual_df], axis=1)

        return result_df        

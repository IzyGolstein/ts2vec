from .ts2vec import TS2Vec
import numpy as  np
import pandas as pd
from scipy.linalg import sqrtm

class DataQualityMetrics:
    """Класс для вычисления метрик качества синтетических данных относительно реальных данных.

    Attributes:
    """
    def __init__(self):
        """Инициализация класса с заданным числом главных компонент."""

    @staticmethod
    def calculate_cross_correlation_matrix(data):
        """Вычисление кросс-корреляционной матрицы для данных.

        Args:
            data (np.array): Данные для которых нужно вычислить корреляционную матрицу.

        Returns:
            np.array: Кросс-корреляционная матрица.
        """
        return np.corrcoef(data.T)

    def correlational_score(self, real_data, synthetic_data):
        """Расчёт корреляционного скора между реальными и синтетическими данными.

        Args:
            real_data (np.array): Реальные данные.
            synthetic_data (np.array): Синтетические данные.

        Returns:
            float: Корреляционный скор, среднее абсолютное отклонение корреляционных матриц.
        """
        real_corr_matrix = self.calculate_cross_correlation_matrix(real_data)
        synth_corr_matrix = self.calculate_cross_correlation_matrix(synthetic_data)
        return np.mean(np.abs(real_corr_matrix - synth_corr_matrix))

    def calculate_fid(self, real_features, synthetic_features):
        """Вычисление Frechet Inception Distance между реальными и синтетическими признаками.

        Args:
            real_features (np.array): Признаки для реальных данных.
            synthetic_features (np.array): Признаки для синтетических данных.

        Returns:
            float: Значение FID.
        """
        mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
        mu_synthetic, sigma_synthetic = np.mean(synthetic_features, axis=0), np.cov(synthetic_features, rowvar=False)
        covmean = sqrtm(sigma_real.dot(sigma_synthetic))
        return np.linalg.norm(mu_real - mu_synthetic) + np.trace(sigma_real + sigma_synthetic - 2 * covmean)

    def extract_features(self, model, real_data):
        """Извлечение признаков из данных с помощью PCA.

        Args:
            data (np.array): Данные для извлечения признаков.

        Returns:
            np.array: Преобразованные данные.
        """

        return model.encode(real_data)
    
    def train_feature_extractor(self, real_data,n):
        """TS2Vec fit

        Args:
            data (np.array): Данные для извлечения признаков.

        Returns:
            np.array: Преобразованные данные.
        """

        model = TS2Vec(input_dims=n, device='cpu', output_dims=n)
        model.fit(real_data)
        return model


    def evaluate_data(self, real_data, synthetic_data_list, columns, metrics_to_evaluate):
        """Оценка качества списка синтетических данных по отношению к реальным данным.

        Args:
            real_data (np.array): Реальные данные.
            synthetic_data_list (list): Список синтетических данных.
            columns (list): Список колонок для выбора из данных.
            metrics_to_evaluate (list of str): Список названий метрик для вычисления.

        Returns:
            dict: Результаты по каждому набору синтетических данных и общий итог.
        """
        results = {}

        summary_results = {'Correlational Score': 0.0, 'Context-FID Score': 0.0}
        count_datasets = 0
        
        data_array = real_data[columns].to_numpy()
        new_shape = (1, data_array.shape[0], data_array.shape[1])
        real_data_selected = data_array.T.reshape(new_shape) 


        np_fakes_list = []
        for fake in synthetic_data_list:
            np_fakes_list.append(fake[columns].to_numpy())
        np_fakes = np.array(np_fakes_list)
        n_cols = len(columns)
        model = self.train_feature_extractor(real_data_selected, n_cols)

        encoded_fakes = model.encode(np_fakes)
        encoded_real_data = model.encode(real_data_selected)
        dataset_result = {}

        if 'Correlational Score' in metrics_to_evaluate:
            for i, synthetic_data in enumerate(np_fakes):
                
                corr_score = self.correlational_score(real_data, synthetic_data)
                if 'Correlational Score' not in dataset_result.keys():
                    dataset_result['Correlational Score'] = []
                dataset_result['Correlational Score'].append(corr_score)

                summary_results['Correlational Score'] += corr_score
                # print(f"Results for dataset {i}: {dataset_result}")
                count_datasets += 1

        if 'Context-FID Score' in metrics_to_evaluate:
            for i, synthetic_data in enumerate(encoded_fakes):
  
                fid_score = self.calculate_fid(encoded_real_data[0], synthetic_data)
                if 'Context-FID Score' not in dataset_result.keys():
                    dataset_result['Context-FID Score'] = []
                dataset_result['Context-FID Score'].append(fid_score)
                summary_results['Context-FID Score'] += fid_score



        if count_datasets > 0:
            for key in summary_results:
                summary_results[key] /= count_datasets

        results['Summary'] = summary_results
        print(f"Summary Results: {summary_results}")
        return results, dataset_result

import datetime
import io
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from django.shortcuts import render
from scipy.stats import t, f
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Устанавливаем высокое разрешение
plt.gcf().set_dpi(600)


def upload_file(request):
    if request.method == 'POST':
        file = request.FILES['file']
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)

            df = df.dropna()

            for index, row in df.iterrows():
                date_str = row['Date']
                value_str = row['Value']

                # Проверка соответствия формату даты
                try:
                    datetime.datetime.strptime(date_str, '%Y-%m-%d')
                except ValueError:
                    df = df.drop(index)
                    continue

                # Проверка соответствия формату числа
                try:
                    float(value_str)
                except ValueError:
                    df = df.drop(index)

            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize('Europe/Moscow')

            # Примеры 1-3
            average_level = df['Value'].mean()
            print("Средний уровень ряда:", average_level)

            df['Growth'] = df['Value'].diff()
            growth = df['Growth'].mean()
            print("Цепные приросты:", growth)

            df['Growth_1'] = df['Value'].pct_change() + 1
            growth_1 = df['Growth_1'].iloc[1:].prod() ** (1 / (len(df) - 1))
            print("Среднегодовой темп роста:", round(growth_1 * 100, 2), "%")
            print("Средний темп прироста:", round(growth_1 * 100 - 100, 2), "%")

            # Пример 4
            df['Month'] = pd.to_datetime(df['Date']).dt.month
            df_first_half = df[df['Month'].isin([1, 2, 3, 4, 5, 6])]
            df_second_half = df[df['Month'].isin([7, 8, 9, 10, 11, 12])]

            y1 = df_first_half['Value'].mean()
            y2 = df_second_half['Value'].mean()
            print("Средняя прибыль за полугодие:", y1, y2)

            # Пример 5
            sigma1 = df_first_half['Value'].var(ddof=1)
            sigma2 = df_second_half['Value'].var(ddof=1)
            print("Дисперсия за полугодие:", sigma1, sigma2)

            F = sigma1 / sigma2
            print("F-критерий Фишера:", F)

            alfa = 0.05
            n1 = len(df_first_half)
            n2 = len(df_second_half)
            v1 = n1 - 1
            v2 = n2 - 1
            F_tabl = f.ppf(1 - alfa, v1, v2)
            print("Табличное значение F-критерия:", F_tabl)

            SE = np.sqrt(sigma1 / n1 + sigma2 / n2)
            t_stat = abs(y1 - y2) / SE
            print("t-критерий Стьюдента:", t_stat)

            df_t = n1 + n2 - 2
            t_tabl = t.ppf(1 - alfa / 2, df_t)
            print("Табличное значение t-критерия:", t_tabl)
            # проверка гипотезы
            if t_stat < t_tabl:
                print("Гипотеза о стационарности ряда принимается")
            else:
                print("Гипотеза о стационарности ряда отвергается")

            # вывод датафрейма с расчетами
            print(df)

            # Пример 6
            def moving_average(data, window_size, center=False):
                if center:
                    window_size += 1
                averages = []
                for i in range(len(data) - window_size + 1):
                    window = data.iloc[i:i + window_size]["Value"].tolist()
                    average = sum(window) / len(window)
                    if center:
                        position = i + (window_size - 1) // 2
                    else:
                        position = i + window_size - 1
                    averages.append((data.iloc[position]["Date"], average))
                return pd.DataFrame(averages, columns=["Date", "Average"])

            # Скользящая средняя из трех уровней
            ma3 = moving_average(df, 3)
            # Скользящая средняя из четырёх уровней без центрирования
            ma4_no_center = moving_average(df, 4)
            # Скользящая средняя из четырёх уровней с центрированием
            ma4_center = moving_average(df, 4, center=True)

            plt.figure(figsize=(12, 6))
            # уменьшение размера маркеров
            plt.rcParams['lines.markersize'] = 0.5
            plt.plot(df["Date"], df["Value"], marker='o', label='Исходные данные', linestyle='-')
            # отображение по оси x только даты, с шагом 12
            plt.xticks(df["Date"][::12])

            # #уменьшение толщины линий
            # plt.rcParams['lines.linewidth'] = 1

            plt.plot(ma3["Date"], ma3["Average"], marker='o', linestyle='-',
                     label='Скользящая средняя (окно 3)')
            plt.plot(ma4_no_center["Date"], ma4_no_center["Average"], marker='o', linestyle='-',
                     label='Скользящая средняя (окно 4, без центрирования)')
            plt.plot(ma4_center["Date"], ma4_center["Average"], marker='o', linestyle='-',
                     label='Скользящая средняя (окно 4, с центрированием)')
            plt.xlabel('Дата')
            plt.ylabel('Значение')
            plt.title('Скользящие средние')
            plt.legend()
            # сохранение графика
            plt.savefig('plot_1.png')
            plt.show()

            # Пример 7
            df["Date_encoded"] = range(1, len(df) + 1)
            linear_model = LinearRegression()
            linear_model.fit(df[["Date_encoded"]], df["Value"])
            linear_pred = linear_model.predict(df[["Date_encoded"]])
            r2_linear = r2_score(df["Value"], linear_pred)

            poly_features = PolynomialFeatures(degree=2)
            X_poly = poly_features.fit_transform(df[["Date_encoded"]])
            poly_model = LinearRegression()
            poly_model.fit(X_poly, df["Value"])
            poly_pred = poly_model.predict(X_poly)
            r2_poly = r2_score(df["Value"], poly_pred)

            exp_model = LinearRegression()
            exp_model.fit(df[["Date_encoded"]], np.log(df["Value"]))
            exp_pred = np.exp(exp_model.predict(df[["Date_encoded"]]))
            r2_exp = r2_score(df["Value"], exp_pred)

            print(f"R² для линейной регрессии: {r2_linear}")
            print(f"R² для квадратичной регрессии: {r2_poly}")
            print(f"R² для экспоненциальной регрессии: {r2_exp}")

            best_model = None
            best_pred = None
            best_r2 = max(r2_linear, r2_poly, r2_exp)
            plt.figure(figsize=(10, 5))

            # уменьшение размера маркеров
            plt.rcParams['lines.markersize'] = 1
            # # уменьшение толщины линий
            # plt.rcParams['lines.linewidth'] = 1
            plt.plot(df["Date"], df["Value"], marker='4', linestyle='-', label='Исходные данные')
            if r2_linear == best_r2:
                best_model = linear_model
                best_pred = linear_pred
                plt.xticks(df["Date"][::12])
                plt.plot(df["Date"], linear_pred, label='Линейная регрессия', linestyle='dotted', color='r')
            elif r2_poly == best_r2:
                best_model = poly_model
                best_pred = poly_pred
                plt.xticks(df["Date"][::12])
                plt.plot(df["Date"], poly_pred, label='Квадратичная регрессия', linestyle='dotted', color='r')
            else:
                best_model = exp_model
                best_pred = exp_pred
                plt.xticks(df["Date"][::12])
                plt.plot(df["Date"], exp_pred, label='Экспоненциальная регрессия', linestyle='dotted', color='r')
            plt.xlabel('Дата')
            plt.ylabel('Значение')
            plt.title('Выбранная модель тренда')
            plt.legend()
            plt.grid()
            # сохранение графика
            plt.savefig('plot_2.png')
            plt.show()

            # Разделение данных на обучающую и тестовую выборки
            train_size = int(len(df) * 0.8)
            train_set, test_set = df[:train_size], df[train_size:]

            # Обучение модели линейной регрессии
            regressor = LinearRegression()
            regressor.fit(train_set[["Date_encoded"]], train_set["Value"])

            # Прогноз на тестовой выборке
            test_set["Predictions"] = regressor.predict(test_set[["Date_encoded"]])

            # Прогноз на будущий период
            future_dates = pd.date_range(start=df["Date"].iloc[-1], periods=12, freq="M")
            future_values = regressor.predict(pd.DataFrame({"Date_encoded": range(len(df) + 1, len(df) + 13)}))
            future_df = pd.DataFrame({"Date": future_dates, "Value": future_values})

            # Построение графика
            plt.figure(figsize=(12, 6))
            plt.plot(df["Date"], df["Value"], color="blue", label="Исходные данные")
            plt.plot(test_set["Date"], test_set["Predictions"], color="red", label="Прогноз на тестовой выборке")
            plt.plot(future_df["Date"], future_df["Value"], color="green", label="Прогноз на будущий период")
            plt.xlabel("Дата")
            plt.ylabel("Значение")
            plt.title("Прогнозирование временного ряда")
            plt.legend()
            plt.savefig('plot_3.png')
            plt.show()



            return render(request, 'success.html')
        else:
            return render(request, 'index.html')
    else:
        return render(request, 'upload.html')


def index(request):
    return render(request, 'index.html')

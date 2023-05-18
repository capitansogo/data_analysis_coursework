import datetime
import matplotlib.pyplot as plt
import numpy as np
from django.shortcuts import render
from scipy.stats import t, f
from sklearn.linear_model import LinearRegression
import pandas as pd
from django.http import HttpResponse

# Устанавливаем высокое разрешение
plt.gcf().set_dpi(600)

df = pd.DataFrame()


def index(request):
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

            data = df.copy().to_json(orient='records')
            print(data)

            # Примеры 1-3
            average_level = df['Value'].mean()
            print("Средний уровень ряда:", average_level)

            df['Growth'] = df['Value'].diff()
            growth = df['Growth'].mean()
            print("Цепные приросты:", growth)

            df['Growth_1'] = df['Value'].pct_change() + 1
            growth_1 = df['Growth_1'].iloc[1:].prod() ** (1 / (len(df) - 1))
            growth_srednegod = round(growth_1 * 100, 2)

            # Пример 4
            df['Month'] = pd.to_datetime(df['Date']).dt.month
            df_first_half = df[df['Month'].isin([1, 2, 3, 4, 5, 6])]
            df_second_half = df[df['Month'].isin([7, 8, 9, 10, 11, 12])]

            y1 = df_first_half['Value'].mean()
            y2 = df_second_half['Value'].mean()

            # Пример 5
            sigma1 = df_first_half['Value'].var(ddof=1)
            sigma2 = df_second_half['Value'].var(ddof=1)

            F = sigma1 / sigma2

            alfa = 0.05
            n1 = len(df_first_half)
            n2 = len(df_second_half)
            v1 = n1 - 1
            v2 = n2 - 1
            F_tabl = f.ppf(1 - alfa, v1, v2)

            SE = np.sqrt(sigma1 / n1 + sigma2 / n2)
            t_stat = abs(y1 - y2) / SE

            df_t = n1 + n2 - 2
            t_tabl = t.ppf(1 - alfa / 2, df_t)
            # проверка гипотезы
            if t_stat < t_tabl:
                gipoteza = "принимается"
            else:
                gipoteza = "отвергается"

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

            # соеденить ma3, ma4_no_center, ma4_center в один датафрейм
            ma3_copy = ma3.rename(columns={"Average": "MA3"})
            ma4_no_center_copy = ma4_no_center.rename(columns={"Average": "MA4_no_center"})
            ma4_center_copy = ma4_center.rename(columns={"Average": "MA4_center"})
            ma3_copy = ma3_copy.set_index("Date")
            ma4_no_center_copy = ma4_no_center_copy.set_index("Date")
            ma4_center_copy = ma4_center_copy.set_index("Date")
            ma3_copy = ma3_copy.join(ma4_no_center_copy)
            ma3_copy = ma3_copy.join(ma4_center_copy)

            ma3_data = ma3.to_json(orient='records')
            ma4_no_center_data = ma4_no_center.to_json(orient='records')
            ma4_center_data = ma4_center.to_json(orient='records')

            # Пример 7
            df["Date_encoded"] = range(1, len(df) + 1)

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

            test_set_data = test_set[["Date", "Predictions"]].copy().to_json(orient='records')
            future_df_data = future_df[["Date", "Value"]].copy().to_json(orient='records')

            df['Value'] = df['Value'].astype(float)

            # создание датасета для построения графика с полями date и value из df
            df_copy = df[['Date', 'Value']].copy()

            chart_data = df_copy.to_json(orient='records')

            # 1. Абсолютный прирост (цепной)
            df['chain_abs_growth'] = df['Value'].diff()

            # 2. Абсолютный прирост (базисный)
            df['base_abs_growth'] = df['Value'] - df['Value'].iloc[0]

            # 3. Темп роста (цепной)
            df['chain_growth_rate'] = df['Value'] / df['Value'].shift(1)

            # 4. Темп роста (базисный)
            df['base_growth_rate'] = df['Value'] / df['Value'].iloc[0]

            # 5. Темп прироста (цепной)
            df['chain_growth_tempo'] = df['chain_growth_rate'].diff()

            # 6. Темп прироста (базисный)
            df['base_growth_tempo'] = df['base_growth_rate'] - df['base_growth_rate'].iloc[0]

            # 7. Абсолютное значение 1% прироста
            df['abs_value_of_1_percent_growth'] = df['Value'] * 0.01

            # 8. Относительное ускорение темпов роста
            df['relative_growth_acceleration'] = df['chain_growth_rate'].pct_change()

            # 9. Коэффициент опережения
            df['lead_ratio'] = df['chain_abs_growth'] / df['base_abs_growth']

            # Исключаем первую строку, так как не можем вычислить прирост для первого значения
            df = df[1:]

            # Создаем модель
            model = LinearRegression()

            # Формируем X и y
            X = np.array(range(len(df))).reshape(-1, 1)
            y = df['Value'].values

            # Обучаем модель
            model.fit(X, y)

            # Предсказываем тренд
            df['linear_trend'] = model.predict(X)

            y_log = np.log1p(y)

            # Обучаем модель
            model.fit(X, y_log)

            # Предсказываем тренд и возвращаем его в исходную шкалу
            df['exponential_trend'] = np.expm1(model.predict(X))

            # Преобразуем данные с помощью обратного числа
            y_inverse = 1 / y

            # Обучаем модель
            model.fit(X, y_inverse)

            # Предсказываем тренд и возвращаем его в исходную шкалу
            df['hyperbolic_trend'] = 1 / model.predict(X)

            hyperbolic_trend = df[['Date', 'hyperbolic_trend']].copy()
            hyperbolic_trend = hyperbolic_trend.to_json(orient='records')
            exponential_trend = df[['Date', 'exponential_trend']].copy()
            exponential_trend = exponential_trend.to_json(orient='records')
            linear_trend = df[['Date', 'linear_trend']].copy()
            linear_trend = linear_trend.to_json(orient='records')
            linear_trend_data = df[['Date', 'Value', 'linear_trend']].copy()
            hyperbolic_trend_data = df[['Date', 'Value', 'hyperbolic_trend']].copy()
            exponential_trend_data = df[['Date', 'Value', 'exponential_trend']].copy()

            # Отрисовываем исходные данные
            plt.figure(figsize=(12, 8))
            plt.plot(df['Value'], label='Original data')

            df.drop(['Growth', 'Growth_1', 'Month', 'Date_encoded'], axis=1, inplace=True)

            df = df.rename(
                columns={'Date': 'Дата', 'Value': 'Значение', 'chain_abs_growth': 'Абсолютный прирост (цепной)',
                         'base_abs_growth': 'Абсолютный прирост (базисный)', 'chain_growth_rate': 'Темп роста (цепной)',
                         'base_growth_rate': 'Темп роста (базисный)', 'chain_growth_tempo': 'Темп прироста (цепной)',
                         'base_growth_tempo': 'Темп прироста (базисный)',
                         'abs_value_of_1_percent_growth': 'Абсолютное значение 1% прироста',
                         'relative_growth_acceleration': 'Относительное ускорение темпов роста',
                         'lead_ratio': 'Коэффициент опережения', 'linear_trend': 'Линейный тренд',
                         'exponential_trend': 'Показательный тренд', 'hyperbolic_trend': 'Гиперболический тренд'})

            request.session['df'] = df.to_json()

            df = df.to_html(
                classes='table table-striped table-hover table-bordered table-sm table-responsive text-center')

            ma3_copy = ma3_copy.to_html(
                classes='table table-striped table-hover table-bordered table-sm table-responsive text-center')

            hyperbolic_trend_data = hyperbolic_trend_data.rename(
                columns={'Date': 'Дата', 'Value': 'Значение', 'hyperbolic_trend': 'Гиперболический тренд'})
            hyperbolic_trend_data = hyperbolic_trend_data.to_html(
                classes='table table-striped table-hover table-bordered table-sm table-responsive text-center')

            exponential_trend_data = exponential_trend_data.rename(
                columns={'Date': 'Дата', 'Value': 'Значение', 'exponential_trend': 'Показательный тренд'})
            exponential_trend_data = exponential_trend_data.to_html(
                classes='table table-striped table-hover table-bordered table-sm table-responsive text-center')

            linear_trend_data = linear_trend_data.rename(
                columns={'Date': 'Дата', 'Value': 'Значение', 'linear_trend': 'Линейный тренд'})
            linear_trend_data = linear_trend_data.to_html(
                classes='table table-striped table-hover table-bordered table-sm table-responsive text-center')

            return render(request, 'success.html',
                          {'average_level': round(average_level, 2), 'growth': round(growth, 2),
                           'growth_srednegod': round(growth_srednegod, 2),
                           'y1': round(y1, 2), 'y2': round(y2, 2),
                           'sigma1': round(sigma1, 2), 'sigma2': round(sigma2, 2),
                           'F': round(F, 2), 'F_tabl': round(F_tabl, 2), 't_stat': round(t_stat, 2),
                           't_tabl': round(t_tabl, 2),
                           'gipoteza': gipoteza,
                           'df': df,
                           'chart_data': chart_data,
                           'ma3_data': ma3_data,
                           'ma4_no_center_data': ma4_no_center_data,
                           'ma4_center_data': ma4_center_data,
                           'data': data,
                           'hyperbolic_trend': hyperbolic_trend,
                           'exponential_trend': exponential_trend,
                           'linear_trend': linear_trend,
                           'test_set_data': test_set_data,
                           'future_df_data': future_df_data,
                           'ma3': ma3_copy,
                           'linear_trend_data': linear_trend_data,
                           'hyperbolic_trend_data': hyperbolic_trend_data,
                           'exponential_trend_data': exponential_trend_data,

                           })

        else:
            return render(request, 'upload.html')
    else:
        return render(request, 'upload.html')


def download_df(request):
    # Извлекаем DataFrame из сессии
    df = pd.read_json(request.session['df'])

    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="data.csv"'
    df.to_csv(path_or_buf=response, sep=',', float_format='%.2f', index=False, encoding='utf-8')

    return response


def test(request):
    return render(request, 'test.html')

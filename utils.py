from datetime import datetime

import numpy as np
import pandas as pd
from t_tech.invest import AsyncClient, CandleInterval
from talib import AD, ATR, CCI, EMA, MACD, OBV, RSI, SMA, STOCH, WILLR


def quotation_to_float(quotation: pd.DataFrame):
    """Преобразует объект Quotation (units+nano) в float"""
    return round(quotation.units + quotation.nano / 1000000000, 2)


async def get_stocks(TOKEN: str) -> dict:
    """Функция возвращает словарь с тикерами-ключами
    и данными в виде полных названий акций и их uid-кодов"""
    async with AsyncClient(token=TOKEN) as client:
        # выгружаем данные всех акций

        stocks = await client.instruments.shares()
        stocks = stocks.instruments

        # фильтруем акции только для московской биржи
        # и для неквалифицированных инвесторов
        # получаем тикеры, названия и коды акций
        tickers = {stock.ticker: {'figi': stock.figi,
                                  'name': stock.name,
                                  'uid': stock.asset_uid,
                                  'sector': stock.sector}
                   for stock in stocks
                   if stock.class_code == 'TQBR' and
                   not stock.for_qual_investor_flag and
                   stock.buy_available_flag == 1}

        return tickers


async def get_candles(TOKEN: str, stocks: dict, start_date: datetime, end_date=datetime) -> pd.DataFrame:
    """Функция для получения данных свечей: open, high, low, close, volume и т.д."""
    async with AsyncClient(TOKEN) as client:
        # Создание пустого списка датафреймов с данными о свечах
        all_candles_dfs = []

        print(f"Запрашиваем данные с {start_date.date()} по {end_date.date()}")
        print(f"Всего тикеров: {len(stocks)}")

        for ticker, values in stocks.items():
            # проходим по каждому тикеру и получаем его историю свечей

            candles_resp = await client.market_data.get_candles(
                figi=values['figi'],
                from_=start_date,
                to=end_date,
                interval=CandleInterval.CANDLE_INTERVAL_DAY
            )
            candles = candles_resp.candles

            # Обработка свечей и запись в список
            rows = []
            for candle in candles:
                row = {
                    'ticker': ticker,
                    'sector': values['sector'],
                    'date': candle.time.date(),
                    'open': quotation_to_float(candle.open),
                    'high': quotation_to_float(candle.high),
                    'low': quotation_to_float(candle.low),
                    'close': quotation_to_float(candle.close),
                    'volume': candle.volume,
                }
                rows.append(row)

            # преобразуем полученный список в DataFrame
            ticker_df = pd.DataFrame(rows)
            if ticker_df.empty:
                # print(f"DataFrame для {ticker} пустой после преобразования")
                continue

            # добавляем датафраму к общему списку
            all_candles_dfs.append(ticker_df)
            # print(f"Успешно добавлен {ticker} в результат")

        if not all_candles_dfs:
            print("Все тикеры были пропущены — возвращаем пустой DataFrame")
            return pd.DataFrame()

        # формируем итоговый датафрейм из списка
        result = pd.concat(all_candles_dfs, axis=0)
        result.sort_index(inplace=True)
        return result


def calc_macd(series: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
    """Расчёт показателя MACD"""
    macd, signal, hist = MACD(series.values,
                              fastperiod=fast_period,
                              slowperiod=slow_period,
                              signalperiod=signal_period)

    return macd, signal, hist


def stoch(df: pd.DataFrame):
    """Рассчитывает значения стохастического осциллятора"""
    slowk, slowd = STOCH(
        df['high'].values,
        df['low'].values,
        df['close'].values,
        fastk_period=21, slowk_period=7, slowk_matype=0,
        slowd_period=7, slowd_matype=0)

    return slowk, slowd


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создаёт технические индикаторы в датафрейме, строго по тикерам.
    Возвращает новый DataFrame (без изменения оригинала).
    """
    # Копируем, чтобы не менять исходный DF
    df = df.sort_values(['ticker', 'date']).copy()

    def _calc_group(g: pd.DataFrame) -> pd.DataFrame:
        # заменяем 0 на близкое значение, чтобы избежать деления на 0
        g.close = g.close.replace(0, 1e-9)

        g['month'] = g['date'].dt.month
        # Доходность
        g['return'] = g['close'].pct_change()
        g['range'] = (g['high'] - g['low']) / g['close'].shift(1)

        # ATR
        g['atr'] = ATR(
            g['high'].values,
            g['low'].values,
            g['close'].values,
            timeperiod=14
        )

        # OBV
        g['obv'] = OBV(
            g['close'].to_numpy(dtype=np.float64),
            g['volume'].to_numpy(dtype=np.float64)
        )

        # Stochastic
        stoch_k, stoch_d = stoch(g)
        g['stoch_k'] = stoch_k
        g['stoch_d'] = stoch_d

        # RSI
        g['rsi'] = RSI(g['close'], 14)

        # EMA
        g['ema_12'] = EMA(g['close'].values, timeperiod=12)
        g['ema_26'] = EMA(g['close'].values, timeperiod=26)

        # SMA и отношения
        g['sma_10'] = SMA(g['close'], 10)
        g['sma_40'] = SMA(g['close'], 40)

        # MACD
        macd, signal, hist = calc_macd(g['close'], 12, 26, 9)
        g['macd'] = macd
        g['signal'] = signal
        g['hist'] = hist

        # Williams R
        g['williams_r'] = WILLR(
            g['high'].values,
            g['low'].values,
            g['close'].values,
            timeperiod=14
        )

        log_ret = np.log(g['close'] / g['close'].shift(1))

        # Volatilty
        g['volatility_20'] = log_ret.rolling(20, min_periods=1).std()
        g['volatility_60'] = log_ret.rolling(60, min_periods=1).std()

        # Momentum
        g['momentum'] = g['close'] / g['close'].shift(5)

        # AD Line
        g['ad_line'] = AD(
            g['high'].to_numpy(dtype=np.float64),
            g['low'].to_numpy(dtype=np.float64),
            g['close'].to_numpy(dtype=np.float64),
            g['volume'].to_numpy(dtype=np.float64)
        )

        # Горизонт прогноза, пробовал разные, этот дал лучший результат
        horizon = 7

        # Определяем долю изменения цены между текущей и будущей
        g['return_7d'] = (g.close.shift(-horizon) - g.close) / g.close
        # Если изменение положительно, то 1. Иначе 0.
        g['target'] = (g['return_7d'] > 0).astype(float)

        return g

    # Применяем расчёт строго по тикерам
    df = df.groupby('ticker').apply(_calc_group)
    df.reset_index(level='ticker', inplace=True)

    return df


def calculate_indicators_for_prediction_modul(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создаёт технические индикаторы в датафрейме, строго по тикерам (без таргета).
    Возвращает новый DataFrame (без изменения оригинала).
    """
    # подробные описания в функции выше
    df = df.sort_values(['ticker', 'date']).copy()

    def _calc_group(g: pd.DataFrame) -> pd.DataFrame:
        g.close = g.close.replace(0, 1e-9)
        g['month'] = g['date'].dt.month

        g['return'] = g['close'].pct_change()
        g['range'] = (g['high'] - g['low']) / g['close'].shift(1)

        g['atr'] = ATR(
            g['high'].values,
            g['low'].values,
            g['close'].values,
            timeperiod=14
        )

        g['obv'] = OBV(
            g['close'].to_numpy(dtype=np.float64),
            g['volume'].to_numpy(dtype=np.float64)
        )

        stoch_k, stoch_d = stoch(g)
        g['stoch_k'] = stoch_k
        g['stoch_d'] = stoch_d

        g['rsi'] = RSI(g['close'], 14)

        g['ema_12'] = EMA(g['close'].values, timeperiod=12)
        g['ema_26'] = EMA(g['close'].values, timeperiod=26)

        g['sma_10'] = SMA(g['close'], 10)
        g['sma_40'] = SMA(g['close'], 40)

        macd, signal, hist = calc_macd(g['close'], 12, 26, 9)
        g['macd'] = macd
        g['signal'] = signal
        g['hist'] = hist

        g['williams_r'] = WILLR(
            g['high'].values,
            g['low'].values,
            g['close'].values,
            timeperiod=14
        )

        log_ret = np.log(g['close'] / g['close'].shift(1))

        g['volatility_20'] = log_ret.rolling(20, min_periods=1).std()
        g['volatility_60'] = log_ret.rolling(60, min_periods=1).std()

        g['momentum'] = g['close'] / g['close'].shift(5)

        g['ad_line'] = AD(
            g['high'].to_numpy(dtype=np.float64),
            g['low'].to_numpy(dtype=np.float64),
            g['close'].to_numpy(dtype=np.float64),
            g['volume'].to_numpy(dtype=np.float64)
        )

        return g

    df = df.groupby('ticker').apply(_calc_group)
    df.reset_index(level='ticker', inplace=True)

    return df

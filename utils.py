import pandas as pd
import numpy as np
from talib import RSI, SMA, MACD, ATR, STOCH, OBV, EMA, WILLR, CCI, AD
from datetime import datetime
from t_tech.invest import AsyncClient, CandleInterval


def quotation_to_float(quotation: pd.DataFrame):
    """Преобразует объект Quotation (units+nano) в float"""
    return round(quotation.units + quotation.nano / 1000000000, 2)


async def get_stocks(TOKEN: str) -> dict:
    """Функция возвращает словарь с тикерами-ключами
    и данными в виде полных названий акций и их uid-кодов"""
    async with AsyncClient(token=TOKEN) as client:
        # выгружаем данные всех акций
        try:
            stocks = await client.instruments.shares()
            stocks = stocks.instruments
        except Exception as e:
            print(e, "Ошибка при получении данных о бумагах")

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


async def get_candles(TOKEN: str, stocks: dict, tickers: list, start_date: datetime, end_date=datetime) -> pd.DataFrame:
    """Функция для получения данных свечей: open, high, low, close, volume и т.д."""
    async with AsyncClient(TOKEN) as client:
        # Создание пустого списка датафреймов с данными о свечах
        all_candles_dfs = []
        end_date = end_date
        start_date = start_date

        print(f"Запрашиваем данные с {start_date.date()} по {end_date.date()}")
        print(f"Всего тикеров: {len(stocks)}")

        for ticker, values in stocks.items():
            if ticker in tickers:
                # проходим по каждому тикеру и получаем его историю свечей
                try:
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

                except Exception as e:
                    # print(f"Ошибка при получении свечей для {ticker}: {e}")
                    continue

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


def get_seasons(df: pd.DataFrame):
    months = df['date'].dt.month

    season_map = {
        12: 'winter', 1: 'winter', 2: 'winter',
        3: 'spring', 4: 'spring', 5: 'spring',
        6: 'summer', 7: 'summer', 8: 'summer',
        9: 'autumn', 10: 'autumn', 11: 'autumn'
    }
    return months.map(season_map)


def calculate_indicators(df: pd.DataFrame):
    """Создаёт технические индикаторы в датафрейме"""
    df['season'] = get_seasons(df)
    df['atr'] = ATR(df['high'].values, df['low'].values,
                    df['close'].values, timeperiod=21)
    df['obv'] = OBV(df['close'].to_numpy(dtype=np.float64),
                    df['volume'].to_numpy(dtype=np.float64))
    df['stoch_k'] = stoch(df)[0]
    df['stoch_d'] = stoch(df)[1]
    vol_mean = df['volume'].rolling(21).mean()
    df['volume_ratio'] = np.where(
        vol_mean != 0, df['volume'] / vol_mean, np.nan)
    df['rsi'] = RSI(df['close'], 21)
    df['rsi_slope'] = df.groupby('ticker')['rsi'].diff()
    df['ema_12'] = EMA(df['close'].values, timeperiod=12)
    df['ema_26'] = EMA(df['close'].values, timeperiod=26)
    df['ema_slope'] = df['ema_12'].diff()
    df['sma_10'] = SMA(df['close'], 10)
    df['sma_40'] = SMA(df['close'], 40)
    df['price_vs_sma_10'] = df['close'] / df.sma_10
    df['sma10_vs_sma40'] = df.sma_10 / df.sma_40
    df['cci_20'] = CCI(df['high'].values, df['low'].values,
                       df['close'].values, timeperiod=20)
    macd, signal, hist = calc_macd(df['close'], 26, 52, 18)
    df['macd'] = macd
    df['signal'] = signal
    df['hist'] = hist
    df['hist_slope'] = df['hist'].diff()
    df['williams_r'] = WILLR(
        df['high'].values, df['low'].values, df['close'].values, timeperiod=21)
    df['volatility_20'] = df['close'].pct_change().rolling(20).std()
    df['volatility_60'] = df['close'].pct_change().rolling(60).std()
    df['vol_ratio_20_60'] = df['volatility_20'] / df['volatility_60']
    df['momentum'] = df['close'] / df['close'].shift(4)
    df['range_norm'] = (df['close'] - df['low']) / \
        (df['high'] - df['low'] + 1e-8)
    df['range_ratio'] = df['high'] / df['low']
    df['open_close_ratio'] = df['open'] / df['close']
    df['volume_spike'] = (df['volume'] > df['volume'].rolling(
        21).mean() * 1.5).astype(int)
    df['ad_line'] = AD(df['high'].to_numpy(dtype=np.float64), df['low'].to_numpy(
        dtype=np.float64), df['close'].to_numpy(dtype=np.float64), df['volume'].to_numpy(dtype=np.float64))
    df['return'] = df['close'].pct_change()
    df['return_7d'] = df['close'].pct_change(periods=6).shift(-6).values
    df['target'] = (df['return_7d'] > 0.006).astype(int)

    return df


def calculate_indicators_for_prediction_modul(df: pd.DataFrame):
    """Создаёт технические индикаторы в датафрейме"""
    df['season'] = get_seasons(df)
    df['atr'] = ATR(df['high'].values, df['low'].values,
                    df['close'].values, timeperiod=21)
    df['obv'] = OBV(df['close'].to_numpy(dtype=np.float64),
                    df['volume'].to_numpy(dtype=np.float64))
    df['stoch_k'] = stoch(df)[0]
    df['stoch_d'] = stoch(df)[1]
    vol_mean = df['volume'].rolling(21).mean()
    df['volume_ratio'] = np.where(
        vol_mean != 0, df['volume'] / vol_mean, np.nan)
    df['rsi'] = RSI(df['close'], 21)
    df['rsi_slope'] = df.groupby('ticker')['rsi'].diff()
    df['ema_12'] = EMA(df['close'].values, timeperiod=12)
    df['ema_26'] = EMA(df['close'].values, timeperiod=26)
    df['ema_slope'] = df['ema_12'].diff()
    df['sma_10'] = SMA(df['close'], 10)
    df['sma_40'] = SMA(df['close'], 40)
    df['price_vs_sma_10'] = df['close'] / df.sma_10
    df['sma10_vs_sma40'] = df.sma_10 / df.sma_40
    df['cci_20'] = CCI(df['high'].values, df['low'].values,
                       df['close'].values, timeperiod=20)
    macd, signal, hist = calc_macd(df['close'], 26, 52, 18)
    df['macd'] = macd
    df['signal'] = signal
    df['hist'] = hist
    df['hist_slope'] = df['hist'].diff()
    df['williams_r'] = WILLR(
        df['high'].values, df['low'].values, df['close'].values, timeperiod=21)
    df['volatility_20'] = df['close'].pct_change().rolling(20).std()
    df['volatility_60'] = df['close'].pct_change().rolling(60).std()
    df['vol_ratio_20_60'] = df['volatility_20'] / df['volatility_60']
    df['momentum'] = df['close'] / df['close'].shift(4)
    df['range_norm'] = (df['close'] - df['low']) / \
        (df['high'] - df['low'] + 1e-8)
    df['range_ratio'] = df['high'] / df['low']
    df['open_close_ratio'] = df['open'] / df['close']
    df['volume_spike'] = (df['volume'] > df['volume'].rolling(
        21).mean() * 1.5).astype(int)
    df['ad_line'] = AD(df['high'].to_numpy(dtype=np.float64), df['low'].to_numpy(
        dtype=np.float64), df['close'].to_numpy(dtype=np.float64), df['volume'].to_numpy(dtype=np.float64))
    df['return'] = df['close'].pct_change()

    return df

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


async def get_candles(TOKEN: str, stocks: dict, tickers: list, start_date: datetime, end_date=datetime) -> pd.DataFrame:
    """Функция для получения данных свечей: open, high, low, close, volume и т.д."""
    async with AsyncClient(TOKEN) as client:
        # Создание пустого списка датафреймов с данными о свечах
        all_candles_dfs = []

        print(f"Запрашиваем данные с {start_date.date()} по {end_date.date()}")
        print(f"Всего тикеров: {len(stocks)}")

        for ticker, values in stocks.items():
            if ticker in tickers:
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


def get_seasons(df: pd.DataFrame):
    months = df['date'].dt.month

    season_map = {
        12: 'winter', 1: 'winter', 2: 'winter',
        3: 'spring', 4: 'spring', 5: 'spring',
        6: 'summer', 7: 'summer', 8: 'summer',
        9: 'autumn', 10: 'autumn', 11: 'autumn'
    }
    return months.map(season_map)


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

        # Сезонность
        g['season'] = get_seasons(g)

        # Волатильность
        g['volatility'] = g.close.pct_change().std()

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

        # Volume ratio
        vol_mean = g['volume'].rolling(21, min_periods=1).mean()
        g['volume_ratio'] = np.where(
            vol_mean != 0, g['volume'] / vol_mean, np.nan
        )

        # RSI и его наклон
        g['rsi'] = RSI(g['close'], 14)
        g['rsi_slope'] = g['rsi'].diff()

        # EMA
        g['ema_12'] = EMA(g['close'].values, timeperiod=12)
        g['ema_26'] = EMA(g['close'].values, timeperiod=26)
        g['ema_slope'] = g['ema_12'].diff()

        # SMA и отношения
        g['sma_10'] = SMA(g['close'], 10)
        g['sma_40'] = SMA(g['close'], 40)
        g['price_vs_sma_10'] = g['close'] / g['sma_10']
        g['sma10_vs_sma40'] = g['sma_10'] / g['sma_40']

        # CCI
        g['cci_20'] = CCI(
            g['high'].values,
            g['low'].values,
            g['close'].values,
            timeperiod=14
        )

        # MACD
        macd, signal, hist = calc_macd(g['close'], 12, 26, 9)
        g['macd'] = macd
        g['signal'] = signal
        g['hist'] = hist
        g['hist_slope'] = g['hist'].diff()

        # Williams R
        g['williams_r'] = WILLR(
            g['high'].values,
            g['low'].values,
            g['close'].values,
            timeperiod=14
        )

        # --- GARCH-подобные признаки (EWMA-волатильность и асимметрия) ---
        span = 32  # аналог alpha ~ 0.94 (стандарт RiskMetrics)
        log_ret = np.log(g['close'] / g['close'].shift(1))

        # 1. EWMA-std (аналог GARCH)
        ewma_std = log_ret.ewm(span=span, min_periods=20).std()
        g['vol_ewma'] = ewma_std

        # 2. Асимметричная волатильность (аналог GJR-GARCH)
        neg_shock = log_ret.clip(upper=0)
        pos_shock = log_ret.clip(lower=0)
        weighted_sq = pos_shock**2 + 1.5 * neg_shock**2  # больший вес на падения
        g['vol_asym'] = weighted_sq.ewm(
            span=span, min_periods=20).mean() ** 0.5

        # 3. Лог-дисперсия (намёк на EGARCH)
        ewma_var = log_ret.ewm(
            span=span, min_periods=20).var().clip(lower=1e-8)
        g['log_vol'] = np.log(ewma_var)

        # --- Классические волатильности и отношения ---
        g['volatility_20'] = log_ret.rolling(20, min_periods=1).std()
        g['volatility_60'] = log_ret.rolling(60, min_periods=1).std()
        g['vol_ratio_20_60'] = np.where(
            g['volatility_60'] != 0,
            g['volatility_20'] / g['volatility_60'],
            np.nan
        )

        # Momentum
        g['momentum'] = g['close'] / g['close'].shift(5)

        # Range-нормы
        denom = (g['high'] - g['low'] + 1e-8)
        g['range_norm'] = (g['close'] - g['low']) / denom
        g['range_ratio'] = g['high'] / g['low']

        # Open/Close ratio
        g['open_close_ratio'] = g['open'] / g['close']

        # Volume spike
        g['volume_spike'] = (
            g['volume'] > g['volume'].rolling(21, min_periods=1).mean() * 1.5
        ).astype(int)

        # AD Line
        g['ad_line'] = AD(
            g['high'].to_numpy(dtype=np.float64),
            g['low'].to_numpy(dtype=np.float64),
            g['close'].to_numpy(dtype=np.float64),
            g['volume'].to_numpy(dtype=np.float64)
        )

        # Return
        g['return'] = log_ret  # лог-доходность (лучше для ML)

        # Return 7d и таргет (строго без утечки: shift(-6) делается внутри группы)
        # horizon = 6 торговых дней вперёд
        horizon = 6
        future_ret = g['return'].shift(-horizon)
        g['return_7d'] = future_ret
        g['target'] = (future_ret > 0).astype(float)

        return g

    # Применяем расчёт строго по тикерам
    df = df.groupby('ticker').apply(_calc_group)
    df.reset_index(level='ticker', inplace=True)

    return df


def calculate_indicators_for_prediction_modul(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создаёт технические индикаторы в датафрейме, строго по тикерам.
    Возвращает новый DataFrame (без изменения оригинала).
    """
    # Копируем, чтобы не менять исходный DF
    df = df.sort_values(['ticker', 'date']).copy()

    def _calc_group(g: pd.DataFrame) -> pd.DataFrame:
        g.close = g.close.replace(0, 1e-9)

        # Сезонность
        g.close = g.close.replace(0, 1e-9)
        g['season'] = get_seasons(g)
        g['volatility'] = g.close.pct_change().std()

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

        # Volume ratio
        vol_mean = g['volume'].rolling(21, min_periods=1).mean()
        g['volume_ratio'] = np.where(
            vol_mean != 0, g['volume'] / vol_mean, np.nan
        )

        # RSI и его наклон
        g['rsi'] = RSI(g['close'], 14)
        g['rsi_slope'] = g['rsi'].diff()

        # EMA
        g['ema_12'] = EMA(g['close'].values, timeperiod=12)
        g['ema_26'] = EMA(g['close'].values, timeperiod=26)
        g['ema_slope'] = g['ema_12'].diff()

        # SMA и отношения
        g['sma_10'] = SMA(g['close'], 10)
        g['sma_40'] = SMA(g['close'], 40)
        g['price_vs_sma_10'] = g['close'] / g['sma_10']
        g['sma10_vs_sma40'] = g['sma_10'] / g['sma_40']

        # CCI
        g['cci_20'] = CCI(
            g['high'].values,
            g['low'].values,
            g['close'].values,
            timeperiod=14
        )

        # MACD
        macd, signal, hist = calc_macd(g['close'], 12, 26, 9)
        g['macd'] = macd
        g['signal'] = signal
        g['hist'] = hist
        g['hist_slope'] = g['hist'].diff()

        # Williams R
        g['williams_r'] = WILLR(
            g['high'].values,
            g['low'].values,
            g['close'].values,
            timeperiod=14
        )

        # --- GARCH-подобные признаки (EWMA-волатильность и асимметрия) ---
        span = 32  # аналог alpha ~ 0.94 (стандарт RiskMetrics)
        log_ret = np.log(g['close'] / g['close'].shift(1))

        # 1. EWMA-std (аналог GARCH)
        ewma_std = log_ret.ewm(span=span, min_periods=20).std()
        g['vol_ewma'] = ewma_std

        # 2. Асимметричная волатильность (аналог GJR-GARCH)
        neg_shock = log_ret.clip(upper=0)
        pos_shock = log_ret.clip(lower=0)
        weighted_sq = pos_shock**2 + 1.5 * neg_shock**2  # больший вес на падения
        g['vol_asym'] = weighted_sq.ewm(
            span=span, min_periods=20).mean() ** 0.5

        # 3. Лог-дисперсия (намёк на EGARCH)
        ewma_var = log_ret.ewm(
            span=span, min_periods=20).var().clip(lower=1e-8)
        g['log_vol'] = np.log(ewma_var)

        # --- Классические волатильности и отношения ---
        g['volatility_20'] = log_ret.rolling(20, min_periods=1).std()
        g['volatility_60'] = log_ret.rolling(60, min_periods=1).std()
        g['vol_ratio_20_60'] = np.where(
            g['volatility_60'] != 0,
            g['volatility_20'] / g['volatility_60'],
            np.nan
        )

        # Momentum
        g['momentum'] = g['close'] / g['close'].shift(5)

        # Range-нормы
        denom = (g['high'] - g['low'] + 1e-8)
        g['range_norm'] = (g['close'] - g['low']) / denom
        g['range_ratio'] = g['high'] / g['low']

        # Open/Close ratio
        g['open_close_ratio'] = g['open'] / g['close']

        # Volume spike
        g['volume_spike'] = (
            g['volume'] > g['volume'].rolling(21, min_periods=1).mean() * 1.5
        ).astype(int)

        # AD Line
        g['ad_line'] = AD(
            g['high'].to_numpy(dtype=np.float64),
            g['low'].to_numpy(dtype=np.float64),
            g['close'].to_numpy(dtype=np.float64),
            g['volume'].to_numpy(dtype=np.float64)
        )

        # Return
        g['return'] = log_ret  # лог-доходность (лучше для ML)

        return g

    # Применяем расчёт строго по тикерам
    df = df.groupby('ticker').apply(_calc_group)
    df.reset_index(level='ticker', inplace=True)

    return df

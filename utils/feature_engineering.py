import pandas as pd
import numpy as np

def compute_volatility_features(df):
    """
    Computes volatility indicators and adds them to the DataFrame.
    Input: df (Pandas DataFrame) with columns [Open, High, Low, Close, Volume]
    Output: df with additional normalized volatility indicator columns
    """
    # 1️⃣ Average True Range (ATR) - Normalized
    df['High-Low'] = df['High'] - df['Low']
    df['High-Close'] = abs(df['High'] - df['Close'].shift(1))
    df['Low-Close'] = abs(df['Low'] - df['Close'].shift(1))
    df['True Range'] = df[['High-Low', 'High-Close', 'Low-Close']].max(axis=1)
    df['ATR'] = df['True Range'].rolling(window=14).mean()
    df['ATR_pct'] = df['ATR'] / df['Close']  # ✅ Normalized ATR

    # 2️⃣ Bollinger Band Width (BBW) - Already Normalized
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['Std_Dev'] = df['Close'].rolling(window=20).std()
    df['Upper_Band'] = df['SMA_20'] + (df['Std_Dev'] * 2)
    df['Lower_Band'] = df['SMA_20'] - (df['Std_Dev'] * 2)
    df['BB_Width'] = (df['Upper_Band'] - df['Lower_Band']) / df['SMA_20']
    
    # 3️⃣ Historical Volatility (HV) - Already Normalized
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['HV'] = df['Log_Returns'].rolling(window=21).std() * np.sqrt(252)  # Annualized HV
    
    # 4️⃣ Relative Volatility Index (RVI) - Already Normalized
    df['Price_Change'] = df['Close'].diff()
    df['Std_Up'] = np.where(df['Price_Change'] > 0, df['Price_Change'].rolling(10).std(), 0)
    df['Std_Down'] = np.where(df['Price_Change'] < 0, df['Price_Change'].rolling(10).std(), 0)
    df['RVI'] = 100 * df['Std_Up'] / (df['Std_Up'] + df['Std_Down'])
    
    # 5️⃣ ATR-EMA (Smoothed ATR) - Normalized
    df['ATR_EMA'] = df['ATR'].ewm(span=10, adjust=False).mean()
    df['ATR_EMA_pct'] = df['ATR_EMA'] / df['Close']  # ✅ Normalized ATR-EMA
    
    # 6️⃣ Choppiness Index (CHOP) - Already Normalized
    df['CHOP'] = 100 * np.log10(df['ATR'].rolling(14).sum() / (df['High'].rolling(14).max() - df['Low'].rolling(14).min())) / np.log10(14)
    
    # 7️⃣ VWAP Deviation - Already Normalized
    df['Cumulative_Typical_Price_Vol'] = (df[['High', 'Low', 'Close']].mean(axis=1) * df['Volume']).cumsum()
    df['Cumulative_Volume'] = df['Volume'].cumsum()
    df['VWAP'] = df['Cumulative_Typical_Price_Vol'] / df['Cumulative_Volume']
    df['VWAP_Deviation'] = (df['Close'] - df['VWAP']) / df['VWAP']

    # 8️⃣ Bollinger Band %B - Already Normalized
    df['BB_PercentB'] = (df['Close'] - df['Lower_Band']) / (df['Upper_Band'] - df['Lower_Band'])
    
    # 9️⃣ Relative Strength Index (RSI) - Already Normalized
    delta = df['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 🔟 Average Directional Index (ADX) - Normalized
    df['ADX'] = df['ATR'].rolling(14).mean() / df['Close']  # ✅ Normalized ADX
    
    # 3️⃣ Compute Stock-Specific ATR% Thresholds (33rd & 66th Percentiles)
    low_threshold = df['ATR_pct'].quantile(0.33)
    high_threshold = df['ATR_pct'].quantile(0.66)

    # 4️⃣ Assign Volatility Labels **Relative to Stock's Own History**
    df['Volatility_Label'] = 0  # Default: Low Volatility
    df.loc[df['ATR_pct'] > low_threshold, 'Volatility_Label'] = 1  # Moderate Volatility
    df.loc[df['ATR_pct'] > high_threshold, 'Volatility_Label'] = 2  # High Volatility

    # 🚨 Drop intermediate calculation columns
    df.drop(columns=['High-Low', 'High-Close', 'Low-Close', 'True Range', 'SMA_20', 'Std_Dev', 'Upper_Band', 'Lower_Band', 
                     'Log_Returns', 'Price_Change', 'Std_Up', 'Std_Down', 'Cumulative_Typical_Price_Vol', 'Cumulative_Volume', 'ATR', 'ATR_EMA'], 
            inplace=True)

    # 🚨 Drop NaNs caused by rolling calculations
    df.dropna(inplace=True)

    return df


def compute_trend_features(df):
    """
    Computes trend-based technical indicators and assigns trend labels.
    - Trend Classification: Uptrend, Sideways, Downtrend
    """

    # 1️⃣ Exponential Moving Averages (EMA - More Responsive)
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['EMA_55'] = df['Close'].ewm(span=55, adjust=False).mean()

    df['EMA_21_pct'] = df['EMA_21'] / df['Close']
    df['EMA_55_pct'] = df['EMA_55'] / df['Close']

    # 2️⃣ MACD (Trend Momentum)
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # 3️⃣ RSI (Relative Strength Index)
    delta = df['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 4️⃣ ADX (Average Directional Index) - Normalized ATR
    df['High-Low'] = df['High'] - df['Low']
    df['High-Close'] = abs(df['High'] - df['Close'].shift(1))
    df['Low-Close'] = abs(df['Low'] - df['Close'].shift(1))
    df['True Range'] = df[['High-Low', 'High-Close', 'Low-Close']].max(axis=1)
    df['ATR'] = df['True Range'].rolling(window=14).mean()
    df['ATR_pct'] = df['ATR'] / df['Close']  # ✅ Normalized ATR
    df['ADX'] = df['ATR_pct'].rolling(14).mean()

    # 5️⃣ Bollinger Bands - Normalized Width
    df['Std_Dev'] = df['Close'].rolling(window=20).std()
    df['Upper_Band'] = df['EMA_21'] + (df['Std_Dev'] * 2)
    df['Lower_Band'] = df['EMA_21'] - (df['Std_Dev'] * 2)
    df['BB_Width'] = (df['Upper_Band'] - df['Lower_Band']) / df['Close']
    df['BB_PercentB'] = (df['Close'] - df['Lower_Band']) / (df['Upper_Band'] - df['Lower_Band'])

    # 6️⃣ VWAP - Normalized Deviation
    df['Cumulative_Typical_Price_Vol'] = (df[['High', 'Low', 'Close']].mean(axis=1) * df['Volume']).cumsum()
    df['Cumulative_Volume'] = df['Volume'].cumsum()
    df['VWAP'] = df['Cumulative_Typical_Price_Vol'] / df['Cumulative_Volume']
    df['VWAP_Deviation'] = (df['Close'] - df['VWAP']) / df['VWAP']

    # 🔹 **Trend Classification (0 = Sideways, 1 = Uptrend, 2 = Downtrend)**
    df['Trend_Label'] = 0  # Default to Sideways

    # Define a threshold to classify sideways trends where EMA_21 is close to EMA_55
    sideways_threshold = 0.01  # 1% threshold (Tunable parameter)

    # Uptrend: EMA_21 is significantly higher than EMA_55
    df.loc[df['EMA_21_pct'] > df['EMA_55_pct'] * (1 + sideways_threshold), 'Trend_Label'] = 1  

    # Downtrend: EMA_21 is significantly lower than EMA_55
    df.loc[df['EMA_21_pct'] < df['EMA_55_pct'] * (1 - sideways_threshold), 'Trend_Label'] = 2

    # 🚨 **Drop intermediate calculation columns**
    df.drop(columns=['EMA_21', 'EMA_55', 'EMA_12', 'EMA_26', 'Signal_Line', 
                     'High-Low', 'High-Close', 'Low-Close', 'True Range', 'ATR', 'Std_Dev',
                     'Upper_Band', 'Lower_Band', 'Cumulative_Typical_Price_Vol', 'Cumulative_Volume'], inplace=True)

    # 🚨 **Drop NaNs caused by rolling calculations**
    df.dropna(inplace=True)

    return df

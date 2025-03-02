# Fixed Data Preparation & Feature Engineering
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('BTC_1min.csv')

# Calculate Returns if not already present
if 'Returns' not in df.columns:
    df['Returns'] = df['Close'].pct_change()

# Fixed Feature Engineering with proper returns handling
def create_features(df, window_sizes=[5, 20, 50]):
    # Technical Indicators
    for window in window_sizes:
        # Simple Moving Average
        df[f'SMA_{window}'] = df['Close'].rolling(window).mean()
        
        # Relative Strength Index (RSI)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        '''
        KONTOLUUUUUUU
        '''
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        
        rs = avg_gain / avg_loss
        df[f'RSI_{window}'] = 100 - (100 / (1 + rs))

    # Lag Features
    for lag in [1, 2, 3, 5, 8]:
        df[f'Returns_lag_{lag}'] = df['Returns'].shift(lag)
    
    # Volatility Features
    df['Volatility_30'] = df['Returns'].rolling(30).std() * np.sqrt(252)
    
    # Volume Features
    df['Volume_Change'] = df['Volume'].pct_change()
    df['Volume_MA_20'] = df['Volume'].rolling(20).mean()
    
    # Target Variable (Next Period Return Direction)
    df['Target'] = np.where(df['Returns'].shift(-1) > 0, 1, -1)
    
    # Clean data
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df.dropna()

df = create_features(df)

# Select Features
features = [col for col in df.columns if col not in [
    'Timestamp', 'Target', 'Returns', 'Open', 'High', 'Low', 'Close', 'Volume'
]]
target = 'Target'
#XCOxDeepSeekAI

# Time-based split
train_size = int(len(df) * 0.8)
train = df.iloc[:train_size]
test = df.iloc[train_size:]

# Scale Features
scaler = StandardScaler()
X_train = scaler.fit_transform(train[features])
X_test = scaler.transform(test[features])
y_train = train[target]
y_test = test[target]

'''
Feature engineering using LGBM Classifier, nice try Diddy
'''
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report

model = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.01,
    num_leaves=127,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    force_col_wise=true
)

model.fit(X_train, y_train,eval_set=[(X_test, y_test)])

preds = model.predict(X_test)
print(classification_report(y_test, preds))

# --- 0. ИМПОРТ БИБЛИОТЕК ---
import pandas as pd
import numpy as np
import lightgbm as lgb
import holidays
import os
import warnings
from sklearn.metrics import make_scorer

# Игнорировать предупреждения
warnings.filterwarnings('ignore')

# --- 1. ОПРЕДЕЛЕНИЕ МЕТРИКИ (SMAPE) ---
def smape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    both_zero_mask = (y_true == 0) & (y_pred == 0)
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    
    ratio = np.zeros_like(denominator, dtype=float)
    valid_mask = (denominator != 0) & (~both_zero_mask)
    
    ratio[valid_mask] = numerator[valid_mask] / denominator[valid_mask]
    ratio[both_zero_mask] = 0
    
    return 100 * np.mean(ratio)

# --- 2. ФУНКЦИИ ДЛЯ СОЗДАНИЯ ФИЧЕЙ (ИЗ ТВОЕГО 13% КОДА) ---

def create_calendar_features(df):
    """Создает календарные фичи из индекса (даты)"""
    df_new = df.copy()
    df_new['day_of_week'] = df_new.index.dayofweek
    df_new['month'] = df_new.index.month
    df_new['day_of_month'] = df_new.index.day
    df_new['day_of_year'] = df_new.index.dayofyear
    df_new['week_of_year'] = df_new.index.isocalendar().week.astype(int)
    df_new['is_weekend'] = (df_new.index.dayofweek >= 5).astype(int)
    df_new['is_month_start'] = df_new.index.is_month_start.astype(int)
    df_new['is_month_end'] = df_new.index.is_month_end.astype(int)
    # Добавим год, так как в test есть 31.12.2021
    df_new['year'] = df_new.index.year 
    return df_new

def create_holiday_features(df):
    """Создает фичи по праздникам РФ"""
    df_new = df.copy()
    # Указываем 2021 и 2022, чтобы покрыть test
    ru_holidays = holidays.Russia(years=[2021, 2022]) 
    df_new['is_holiday'] = df_new.index.map(lambda x: x in ru_holidays).astype(int)
    
    if '2021-12-31' in df_new.index:
        df_new.loc['2021-12-31', 'is_holiday'] = 1
        
    return df_new

def create_lag_features(df, target_col='target', lags=[84, 91, 98, 105]):
    """
    Создает "дальние" лаги, которые "перепрыгивают" 62-дневную дыру в данных.
    (12, 13, 14, 15 недель назад)
    """
    df_new = df.copy()
    for lag in lags:
        df_new[f'lag_{lag}'] = df_new[target_col].shift(lag)
    return df_new

# --- 3. ЗАГРУЗКА, ОБРАБОТКА И ОБУЧЕНИЕ ---

print("Запуск скрипта...")
BASE_PATH = '/kaggle/input/ai-hackaton-code-create-conquer4/'
TARGET = 'target'

try:
    train_df = pd.read_csv(f'{BASE_PATH}train.csv', parse_dates=['date'], index_col='date')
    test_df = pd.read_csv(f'{BASE_PATH}test.csv', parse_dates=['date'], index_col='date')
except FileNotFoundError as e:
    print(f"Ошибка: Файлы не найдены. {e}")
    raise

print(f"Train: {train_df.index.min().date()} по {train_df.index.max().date()}")
print(f"Test:  {test_df.index.min().date()} по {test_df.index.max().date()}")

# --- ОБРАБОТКА ДАННЫХ (Как в 13% коде) ---
test_df[TARGET] = np.nan 
df_full = pd.concat([train_df, test_df])
df_full = df_full.sort_index()

print("Создание фичей (простая версия)...")
df_full = create_calendar_features(df_full)
df_full = create_holiday_features(df_full)
df_full = create_lag_features(df_full, target_col=TARGET) # Только лаги

train_processed = df_full[~df_full[TARGET].isna()]
test_processed = df_full[df_full[TARGET].isna()]

train_processed = train_processed.dropna(subset=[col for col in train_processed.columns if 'lag_' in col])

FEATURES = [col for col in train_processed.columns if col != TARGET]
X_train = train_processed[FEATURES]
y_train = train_processed[TARGET]

# *** КЛЮЧЕВАЯ СТРОКА ***
X_test = test_processed[FEATURES].fillna(0) # Заполняем пропуски в лагах нулем

categorical_features = [
    'day_of_week', 'month', 'is_weekend', 'is_holiday', 
    'is_month_start', 'is_month_end', 'year'
]
categorical_features = [col for col in categorical_features if col in FEATURES]

print(f"Данные готовы к обучению. Форма X_train: {X_train.shape}")
print(f"Форма X_test: {X_test.shape}")

# --- 4. ОБУЧЕНИЕ МОДЕЛИ (ФИНАЛЬНЫЙ ТЮНИНГ) ---

# Параметры настроены на МАКСИМАЛЬНУЮ РОБАСТНОСТЬ (борьбу с переобучением)
model = lgb.LGBMRegressor(
    objective='mae', 
    n_estimators=5000,          # Увеличено
    learning_rate=0.005,        # Уменьшено
    n_jobs=-1,
    random_state=42,
    metric='None',
    colsample_bytree=0.7,
    subsample=0.7,
    reg_alpha=0.2,              # Увеличено
    reg_lambda=0.2,             # Увеличено
    num_leaves=8                # <-- РАДИКАЛЬНО УМЕНЬШЕНО (для борьбы с переобучением)
)

print("Начинаю обучение...")

model.fit(
    X_train, 
    y_train,
    eval_set=[(X_train, y_train)],
    eval_metric=lambda y_true, y_pred: [('custom_smape', smape(y_true, y_pred), False)],
    callbacks=[
        lgb.early_stopping(300, verbose=True), # Увеличено терпение
        lgb.log_evaluation(period=500)
    ],
    categorical_feature=categorical_features
)

print("Обучение завершено.")

# --- 5. ПРОГНОЗ И СОЗДАНИЕ ФАЙЛА ---
predictions = model.predict(X_test)
predictions[predictions < 0] = 0 

print("Прогноз готов.")

submission = pd.DataFrame({
    'date': X_test.index.strftime('%Y-%m-%d'),
    'target': predictions
})

submission.to_csv('submission3.csv', index=False)

print("Файл 'submission.csv' успешно сохранен!")
print(submission.head())
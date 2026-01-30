import numpy as np

def compute_future_return(df, horizon=30, price_col='close'):
    """
    Computes forward return over a fixed horizon
    Safe for ranking / IC models
    """

    df = df.copy()

    df['future_return'] = (
        df[price_col].shift(-horizon) / df[price_col] - 1.0
    )

    return df

def compute_forward_return(df, horizon=30):
    df = df.copy()
    df['future_return'] = np.log(
        df['close'].shift(-horizon) / df['close']
    )
    return df
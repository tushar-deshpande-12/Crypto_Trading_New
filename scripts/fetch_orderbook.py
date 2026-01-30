"""
Order Book Data Fetcher for Binance
====================================
Fetches order book snapshots from Binance API and saves them for analysis.

Based on: "Blockchain-Native Asset Direction Prediction: A Confidence-Threshold
Approach to Decentralized Financial Analytics Using Multi-Scale Feature Integration"
(MDPI Algorithms 18(12), 758, November 2025)

Key features extracted:
- Bid-ask spread
- Order book imbalance
- Depth-weighted mid-price
- Volume at best bid/ask levels
- Order book slope (price impact estimation)
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class OrderBookConfig:
    """Configuration for order book fetching."""
    symbol: str = "BTCUSDT"
    depth_limit: int = 100  # Number of price levels (5, 10, 20, 50, 100, 500, 1000, 5000)
    fetch_interval_seconds: float = 1.0  # Time between fetches
    duration_minutes: int = 1440  # 24 hours - need at least 1000+ candles for training
    output_dir: str = r"C:\crypto\ver7\dataset\BTCUSDT\orderbook"
    base_url: str = "https://api.binance.com"


class BinanceOrderBookFetcher:
    """Fetches order book data from Binance API."""

    def __init__(self, config: OrderBookConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def fetch_snapshot(self) -> Optional[Dict]:
        """Fetch a single order book snapshot."""
        url = f"{self.config.base_url}/api/v3/depth"
        params = {
            'symbol': self.config.symbol,
            'limit': self.config.depth_limit
        }

        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Add timestamp
            data['timestamp'] = datetime.utcnow().isoformat()
            data['local_timestamp'] = datetime.now().isoformat()

            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching order book: {e}")
            return None

    def calculate_features(self, snapshot: Dict) -> Dict:
        """Calculate order book features from snapshot."""
        if not snapshot or 'bids' not in snapshot or 'asks' not in snapshot:
            return {}

        bids = np.array([[float(p), float(q)] for p, q in snapshot['bids']])
        asks = np.array([[float(p), float(q)] for p, q in snapshot['asks']])

        if len(bids) == 0 or len(asks) == 0:
            return {}

        # Best bid/ask
        best_bid_price = bids[0, 0]
        best_ask_price = asks[0, 0]
        best_bid_qty = bids[0, 1]
        best_ask_qty = asks[0, 1]

        # Mid price
        mid_price = (best_bid_price + best_ask_price) / 2

        # Spread metrics
        spread_absolute = best_ask_price - best_bid_price
        spread_relative = spread_absolute / mid_price
        spread_bps = spread_relative * 10000  # Basis points

        # Volume at different depth levels
        depth_levels = [5, 10, 20, 50, 100]
        bid_volumes = {}
        ask_volumes = {}

        for depth in depth_levels:
            if depth <= len(bids):
                bid_volumes[f'bid_volume_{depth}'] = float(bids[:depth, 1].sum())
            if depth <= len(asks):
                ask_volumes[f'ask_volume_{depth}'] = float(asks[:depth, 1].sum())

        # Order book imbalance at different levels
        imbalances = {}
        for depth in depth_levels:
            bid_key = f'bid_volume_{depth}'
            ask_key = f'ask_volume_{depth}'
            if bid_key in bid_volumes and ask_key in ask_volumes:
                total = bid_volumes[bid_key] + ask_volumes[ask_key]
                if total > 0:
                    imbalance = (bid_volumes[bid_key] - ask_volumes[ask_key]) / total
                    imbalances[f'imbalance_{depth}'] = float(imbalance)

        # Depth-weighted mid-price (VWAP-style)
        # Weight by volume at each price level
        bid_weights = bids[:, 1] / bids[:, 1].sum() if bids[:, 1].sum() > 0 else np.ones(len(bids)) / len(bids)
        ask_weights = asks[:, 1] / asks[:, 1].sum() if asks[:, 1].sum() > 0 else np.ones(len(asks)) / len(asks)

        depth_weighted_bid = float(np.sum(bids[:, 0] * bid_weights))
        depth_weighted_ask = float(np.sum(asks[:, 0] * ask_weights))
        depth_weighted_mid = (depth_weighted_bid + depth_weighted_ask) / 2

        # Order book slope (price impact estimation)
        # How much the price moves per unit of volume
        if len(bids) >= 10 and bids[:10, 1].sum() > 0:
            bid_slope = (bids[0, 0] - bids[9, 0]) / bids[:10, 1].sum()
        else:
            bid_slope = 0.0

        if len(asks) >= 10 and asks[:10, 1].sum() > 0:
            ask_slope = (asks[9, 0] - asks[0, 0]) / asks[:10, 1].sum()
        else:
            ask_slope = 0.0

        # Aggregate order book pressure
        total_bid_value = float(np.sum(bids[:, 0] * bids[:, 1]))
        total_ask_value = float(np.sum(asks[:, 0] * asks[:, 1]))

        features = {
            'timestamp': snapshot.get('timestamp'),
            'local_timestamp': snapshot.get('local_timestamp'),
            'lastUpdateId': snapshot.get('lastUpdateId'),

            # Price features
            'best_bid_price': float(best_bid_price),
            'best_ask_price': float(best_ask_price),
            'mid_price': float(mid_price),
            'depth_weighted_mid': float(depth_weighted_mid),

            # Volume features
            'best_bid_qty': float(best_bid_qty),
            'best_ask_qty': float(best_ask_qty),
            **bid_volumes,
            **ask_volumes,

            # Spread features
            'spread_absolute': float(spread_absolute),
            'spread_relative': float(spread_relative),
            'spread_bps': float(spread_bps),

            # Imbalance features
            **imbalances,

            # Slope features (price impact)
            'bid_slope': float(bid_slope),
            'ask_slope': float(ask_slope),

            # Aggregate features
            'total_bid_value': total_bid_value,
            'total_ask_value': total_ask_value,
            'value_imbalance': (total_bid_value - total_ask_value) / (total_bid_value + total_ask_value) if (total_bid_value + total_ask_value) > 0 else 0.0
        }

        return features

    def collect_data(self, save_raw: bool = True) -> pd.DataFrame:
        """Collect order book data for the configured duration."""

        # Create output directory
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Session timestamp for file naming
        session_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        raw_data = []
        feature_data = []

        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=self.config.duration_minutes)

        logger.info(f"Starting order book collection for {self.config.symbol}")
        logger.info(f"Duration: {self.config.duration_minutes} minutes")
        logger.info(f"Interval: {self.config.fetch_interval_seconds} seconds")
        logger.info(f"Output: {output_dir}")

        fetch_count = 0
        error_count = 0

        try:
            while datetime.now() < end_time:
                snapshot = self.fetch_snapshot()

                if snapshot:
                    fetch_count += 1

                    if save_raw:
                        raw_data.append(snapshot)

                    features = self.calculate_features(snapshot)
                    if features:
                        feature_data.append(features)

                    if fetch_count % 60 == 0:
                        elapsed = (datetime.now() - start_time).total_seconds() / 60
                        logger.info(f"Collected {fetch_count} snapshots ({elapsed:.1f} min elapsed)")
                else:
                    error_count += 1

                time.sleep(self.config.fetch_interval_seconds)

        except KeyboardInterrupt:
            logger.info("Collection interrupted by user")

        # Save raw data
        if save_raw and raw_data:
            raw_file = output_dir / f"{session_ts}_raw.json"
            with open(raw_file, 'w') as f:
                json.dump(raw_data, f)
            logger.info(f"Raw data saved to {raw_file}")

        # Save feature data
        if feature_data:
            df = pd.DataFrame(feature_data)

            # Save as CSV
            csv_file = output_dir / f"{session_ts}_features.csv"
            df.to_csv(csv_file, index=False)
            logger.info(f"Features saved to {csv_file}")

            # Save as Parquet for efficiency
            parquet_file = output_dir / f"{session_ts}_features.parquet"
            df.to_parquet(parquet_file, index=False)
            logger.info(f"Features saved to {parquet_file}")

            return df

        logger.info(f"Collection complete. Fetches: {fetch_count}, Errors: {error_count}")
        return pd.DataFrame()


def load_orderbook_features(orderbook_dir: str = None,
                            latest_only: bool = True) -> Optional[pd.DataFrame]:
    """
    Load order book features from saved files.

    Args:
        orderbook_dir: Directory containing order book data
        latest_only: If True, load only the most recent file

    Returns:
        DataFrame with order book features
    """
    if orderbook_dir is None:
        orderbook_dir = r"C:\crypto\ver7\dataset\BTCUSDT\orderbook"

    orderbook_path = Path(orderbook_dir)
    if not orderbook_path.exists():
        logger.warning(f"Order book directory not found: {orderbook_dir}")
        return None

    # Find feature files
    parquet_files = sorted(orderbook_path.glob("*_features.parquet"))
    csv_files = sorted(orderbook_path.glob("*_features.csv"))

    files_to_load = parquet_files if parquet_files else csv_files

    if not files_to_load:
        logger.warning(f"No order book feature files found in {orderbook_dir}")
        return None

    if latest_only:
        files_to_load = [files_to_load[-1]]

    dfs = []
    for f in files_to_load:
        if f.suffix == '.parquet':
            df = pd.read_parquet(f)
        else:
            df = pd.read_csv(f)
        dfs.append(df)
        logger.info(f"Loaded {len(df)} rows from {f.name}")

    if dfs:
        result = pd.concat(dfs, ignore_index=True)
        result['timestamp'] = pd.to_datetime(result['timestamp'])
        result = result.sort_values('timestamp').reset_index(drop=True)
        return result

    return None


def get_orderbook_feature_columns() -> List[str]:
    """Get list of order book feature column names."""
    return [
        'spread_bps',
        'spread_relative',
        'imbalance_5',
        'imbalance_10',
        'imbalance_20',
        'imbalance_50',
        'imbalance_100',
        'depth_weighted_mid',
        'bid_slope',
        'ask_slope',
        'best_bid_qty',
        'best_ask_qty',
        'bid_volume_5',
        'bid_volume_10',
        'bid_volume_20',
        'ask_volume_5',
        'ask_volume_10',
        'ask_volume_20',
        'value_imbalance'
    ]


def merge_orderbook_with_ohlcv(ohlcv_df: pd.DataFrame,
                               orderbook_df: pd.DataFrame,
                               resample_rule: str = '1min') -> pd.DataFrame:
    """
    Merge order book features with OHLCV data.

    Args:
        ohlcv_df: DataFrame with OHLCV data (datetime index)
        orderbook_df: DataFrame with order book features
        resample_rule: Resampling rule for order book data

    Returns:
        Merged DataFrame
    """
    if orderbook_df is None or len(orderbook_df) == 0:
        logger.warning("No order book data to merge")
        return ohlcv_df

    # Ensure timestamp is datetime
    ob_df = orderbook_df.copy()
    ob_df['timestamp'] = pd.to_datetime(ob_df['timestamp'])
    ob_df.set_index('timestamp', inplace=True)

    # Get numeric columns only
    numeric_cols = ob_df.select_dtypes(include=[np.number]).columns.tolist()

    # Resample order book data to match OHLCV timeframe
    ob_resampled = ob_df[numeric_cols].resample(resample_rule).agg({
        col: 'mean' for col in numeric_cols
    })

    # Also add some aggregates
    ob_resampled['spread_bps_max'] = ob_df['spread_bps'].resample(resample_rule).max()
    ob_resampled['spread_bps_min'] = ob_df['spread_bps'].resample(resample_rule).min()
    ob_resampled['imbalance_5_std'] = ob_df['imbalance_5'].resample(resample_rule).std()

    # Merge with OHLCV
    merged = ohlcv_df.join(ob_resampled, how='left')

    # Forward fill missing order book data (limited to 5 periods)
    ob_cols = [c for c in merged.columns if c in numeric_cols or 'spread_bps' in c or 'imbalance' in c]
    merged[ob_cols] = merged[ob_cols].fillna(method='ffill', limit=5)

    logger.info(f"Merged order book features. Shape: {merged.shape}")

    return merged


def main():
    """Main function to run order book collection."""
    import argparse

    parser = argparse.ArgumentParser(description='Fetch Binance order book data')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading pair symbol')
    parser.add_argument('--duration', type=int, default=60, help='Collection duration in minutes')
    parser.add_argument('--interval', type=float, default=1.0, help='Fetch interval in seconds')
    parser.add_argument('--depth', type=int, default=100, help='Order book depth')
    parser.add_argument('--output', type=str, default=r"C:\crypto\ver7\dataset\BTCUSDT\orderbook",
                       help='Output directory')

    args = parser.parse_args()

    config = OrderBookConfig(
        symbol=args.symbol,
        duration_minutes=args.duration,
        fetch_interval_seconds=args.interval,
        depth_limit=args.depth,
        output_dir=args.output
    )

    fetcher = BinanceOrderBookFetcher(config)
    df = fetcher.collect_data()

    if len(df) > 0:
        print(f"\nCollection Summary:")
        print(f"  Total snapshots: {len(df)}")
        print(f"  Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"\nFeature Statistics:")
        print(df.describe())
    else:
        print("No data collected")


if __name__ == "__main__":
    main()

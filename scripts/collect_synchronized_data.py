"""
Synchronized Data Collector for Paper Implementation
=====================================================
Collects both OHLCV and order book data simultaneously from Binance.

This ensures both data sources are aligned in time for the multi-scale
feature integration approach described in the paper.

Reference: "Blockchain-Native Asset Direction Prediction" (MDPI Algorithms 18(12), 758)
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import threading
from queue import Queue

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
class CollectorConfig:
    """Configuration for synchronized data collection."""
    symbol: str = "BTCUSDT"
    timeframe: str = "1m"  # 1-minute candles
    orderbook_depth: int = 100
    orderbook_interval: float = 1.0  # Seconds between order book fetches
    duration_hours: float = 24.0  # Total collection duration
    output_dir: str = r"C:\crypto\ver7\dataset\BTCUSDT\synchronized"
    base_url: str = "https://api.binance.com"


class SynchronizedCollector:
    """Collects synchronized OHLCV and order book data."""

    def __init__(self, config: CollectorConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.running = False
        self.ohlcv_data = []
        self.orderbook_data = []

    def fetch_klines(self, limit: int = 1) -> List[Dict]:
        """Fetch the most recent klines (OHLCV) data."""
        url = f"{self.config.base_url}/api/v3/klines"
        params = {
            'symbol': self.config.symbol,
            'interval': self.config.timeframe,
            'limit': limit
        }

        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            klines = []
            for k in data:
                klines.append({
                    'open_time': k[0],
                    'open': float(k[1]),
                    'high': float(k[2]),
                    'low': float(k[3]),
                    'close': float(k[4]),
                    'volume': float(k[5]),
                    'close_time': k[6],
                    'quote_volume': float(k[7]),
                    'trades': int(k[8]),
                    'taker_buy_base': float(k[9]),
                    'taker_buy_quote': float(k[10])
                })
            return klines

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching klines: {e}")
            return []

    def fetch_orderbook(self) -> Optional[Dict]:
        """Fetch order book snapshot with features."""
        url = f"{self.config.base_url}/api/v3/depth"
        params = {
            'symbol': self.config.symbol,
            'limit': self.config.orderbook_depth
        }

        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Calculate features
            bids = np.array([[float(p), float(q)] for p, q in data['bids']])
            asks = np.array([[float(p), float(q)] for p, q in data['asks']])

            if len(bids) == 0 or len(asks) == 0:
                return None

            best_bid = bids[0, 0]
            best_ask = asks[0, 0]
            mid_price = (best_bid + best_ask) / 2
            spread_bps = (best_ask - best_bid) / mid_price * 10000

            # Imbalances at different depths
            features = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'mid_price': mid_price,
                'spread_bps': spread_bps,
                'best_bid_qty': bids[0, 1],
                'best_ask_qty': asks[0, 1],
            }

            for depth in [5, 10, 20, 50]:
                if depth <= len(bids) and depth <= len(asks):
                    bid_vol = bids[:depth, 1].sum()
                    ask_vol = asks[:depth, 1].sum()
                    total = bid_vol + ask_vol
                    features[f'imbalance_{depth}'] = (bid_vol - ask_vol) / total if total > 0 else 0
                    features[f'bid_volume_{depth}'] = bid_vol
                    features[f'ask_volume_{depth}'] = ask_vol

            # Order book slope
            if len(bids) >= 10:
                features['bid_slope'] = (bids[0, 0] - bids[9, 0]) / bids[:10, 1].sum() if bids[:10, 1].sum() > 0 else 0
            if len(asks) >= 10:
                features['ask_slope'] = (asks[9, 0] - asks[0, 0]) / asks[:10, 1].sum() if asks[:10, 1].sum() > 0 else 0

            return features

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching order book: {e}")
            return None

    def collect_orderbook_loop(self):
        """Background thread for order book collection."""
        while self.running:
            features = self.fetch_orderbook()
            if features:
                self.orderbook_data.append(features)
            time.sleep(self.config.orderbook_interval)

    def collect(self) -> Dict[str, pd.DataFrame]:
        """
        Collect synchronized OHLCV and order book data.

        Returns dict with 'ohlcv' and 'orderbook' DataFrames.
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        session_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        duration = timedelta(hours=self.config.duration_hours)
        start_time = datetime.now()
        end_time = start_time + duration

        logger.info(f"Starting synchronized collection for {self.config.symbol}")
        logger.info(f"Duration: {self.config.duration_hours} hours")
        logger.info(f"OHLCV interval: {self.config.timeframe}")
        logger.info(f"Order book interval: {self.config.orderbook_interval}s")
        logger.info(f"Output: {output_dir}")

        # Start order book collection in background thread
        self.running = True
        ob_thread = threading.Thread(target=self.collect_orderbook_loop, daemon=True)
        ob_thread.start()

        # Collect OHLCV data in main loop
        last_kline_time = 0
        kline_interval_ms = 60000 if self.config.timeframe == '1m' else 3600000

        try:
            while datetime.now() < end_time:
                # Fetch latest klines
                klines = self.fetch_klines(limit=2)

                for k in klines:
                    if k['open_time'] > last_kline_time:
                        self.ohlcv_data.append(k)
                        last_kline_time = k['open_time']

                elapsed = (datetime.now() - start_time).total_seconds() / 3600
                if len(self.ohlcv_data) % 10 == 0 and len(self.ohlcv_data) > 0:
                    logger.info(f"Collected {len(self.ohlcv_data)} candles, "
                              f"{len(self.orderbook_data)} order book snapshots "
                              f"({elapsed:.2f}h elapsed)")

                # Wait for next candle
                time.sleep(30)  # Check every 30 seconds for new candles

        except KeyboardInterrupt:
            logger.info("Collection interrupted by user")
        finally:
            self.running = False

        # Convert to DataFrames
        ohlcv_df = pd.DataFrame(self.ohlcv_data)
        if len(ohlcv_df) > 0:
            ohlcv_df['datetime'] = pd.to_datetime(ohlcv_df['open_time'], unit='ms')
            ohlcv_df = ohlcv_df.drop_duplicates(subset='open_time').sort_values('open_time')

        orderbook_df = pd.DataFrame(self.orderbook_data)
        if len(orderbook_df) > 0:
            orderbook_df['timestamp'] = pd.to_datetime(orderbook_df['timestamp'])
            orderbook_df = orderbook_df.sort_values('timestamp')

        # Save data
        if len(ohlcv_df) > 0:
            ohlcv_file = output_dir / f"{session_ts}_ohlcv.parquet"
            ohlcv_df.to_parquet(ohlcv_file, index=False)
            logger.info(f"OHLCV saved to {ohlcv_file}")

            ohlcv_csv = output_dir / f"{session_ts}_ohlcv.csv"
            ohlcv_df.to_csv(ohlcv_csv, index=False)

        if len(orderbook_df) > 0:
            ob_file = output_dir / f"{session_ts}_orderbook.parquet"
            orderbook_df.to_parquet(ob_file, index=False)
            logger.info(f"Order book saved to {ob_file}")

            ob_csv = output_dir / f"{session_ts}_orderbook.csv"
            orderbook_df.to_csv(ob_csv, index=False)

        # Create merged dataset
        merged = self.merge_datasets(ohlcv_df, orderbook_df)
        if len(merged) > 0:
            merged_file = output_dir / f"{session_ts}_merged.parquet"
            merged.to_parquet(merged_file, index=False)
            merged_csv = output_dir / f"{session_ts}_merged.csv"
            merged.to_csv(merged_csv, index=False)
            logger.info(f"Merged data saved to {merged_file}")

        logger.info(f"\nCollection complete!")
        logger.info(f"  OHLCV records: {len(ohlcv_df)}")
        logger.info(f"  Order book snapshots: {len(orderbook_df)}")
        logger.info(f"  Merged records: {len(merged)}")

        return {
            'ohlcv': ohlcv_df,
            'orderbook': orderbook_df,
            'merged': merged
        }

    def merge_datasets(self, ohlcv_df: pd.DataFrame, orderbook_df: pd.DataFrame) -> pd.DataFrame:
        """Merge OHLCV and order book data on time."""
        if len(ohlcv_df) == 0 or len(orderbook_df) == 0:
            return pd.DataFrame()

        ohlcv = ohlcv_df.copy()
        ob = orderbook_df.copy()

        # Ensure datetime columns
        ohlcv['datetime'] = pd.to_datetime(ohlcv['datetime'])
        ob['timestamp'] = pd.to_datetime(ob['timestamp'])

        # Set index
        ohlcv = ohlcv.set_index('datetime')
        ob = ob.set_index('timestamp')

        # Aggregate order book features per minute
        numeric_cols = ob.select_dtypes(include=[np.number]).columns.tolist()
        ob_agg = ob[numeric_cols].resample('1min').agg({
            col: 'mean' for col in numeric_cols
        })

        # Add std for volatility measures
        for col in ['imbalance_5', 'imbalance_10', 'spread_bps']:
            if col in ob.columns:
                ob_agg[f'{col}_std'] = ob[col].resample('1min').std()

        # Merge
        merged = ohlcv.join(ob_agg, how='left')
        merged = merged.reset_index()

        return merged


def main():
    """Main collection function."""
    import argparse

    parser = argparse.ArgumentParser(description='Collect synchronized OHLCV and order book data')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading pair')
    parser.add_argument('--duration', type=float, default=24.0, help='Duration in hours')
    parser.add_argument('--output', type=str, default=r"C:\crypto\ver7\dataset\BTCUSDT\synchronized",
                       help='Output directory')

    args = parser.parse_args()

    config = CollectorConfig(
        symbol=args.symbol,
        duration_hours=args.duration,
        output_dir=args.output
    )

    collector = SynchronizedCollector(config)
    results = collector.collect()

    if len(results['merged']) > 0:
        print("\n" + "="*60)
        print("COLLECTION COMPLETE")
        print("="*60)
        print(f"Total candles: {len(results['ohlcv'])}")
        print(f"Order book snapshots: {len(results['orderbook'])}")
        print(f"Merged records: {len(results['merged'])}")
        print(f"\nFeatures available:")
        for col in results['merged'].columns:
            print(f"  - {col}")
        print("\nTo train the model with this data:")
        print(f"  1. Update CONFIG.CSV_FILE to point to the merged.csv file")
        print(f"  2. Run: python paper_implementation.py")


if __name__ == "__main__":
    main()

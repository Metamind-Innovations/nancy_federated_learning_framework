import os
import pandas as pd
import numpy as np
import joblib
from typing import Tuple, Dict, List, Optional
from sklearn.preprocessing import StandardScaler


def preprocess_data(
    data_path: str,
    scaler_path: Optional[str] = None,
    train_file: str = "Train.csv",
    test_file: str = "Test.csv",
    create_scaler: bool = False
) -> Tuple[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series], Optional[StandardScaler]]:
    df_train = pd.read_csv(os.path.join(data_path, train_file))
    df_test = pd.read_csv(os.path.join(data_path, test_file))
    
    columns_to_drop = ['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port',
                       'Protocol', 'Timestamp']
    df_train_reduced = df_train.drop(columns_to_drop, axis=1)
    df_test_reduced = df_test.drop(columns_to_drop, axis=1)
    
    df_train_reduced.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_train_no_na = df_train_reduced.dropna(axis=0)
    
    df_test_reduced.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_test_no_na = df_test_reduced.dropna(axis=0)
    
    y_train = df_train_no_na['Label']
    X_train = df_train_no_na.drop(['Label'], axis=1)
    
    y_test = df_test_no_na['Label']
    X_test = df_test_no_na.drop(['Label'], axis=1)
    
    scaler = None
    
    if create_scaler:
        scaler = StandardScaler()
        X_scaled_train = scaler.fit_transform(X_train)
        
        if scaler_path:
            os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
            joblib.dump(scaler, scaler_path)
            print(f"StandardScaler saved to {scaler_path}")
    else:
        if not scaler_path:
            raise ValueError("Scaler path must be provided when create_scaler is False")
        
        scaler = joblib.load(scaler_path, mmap_mode=None)
        X_scaled_train = scaler.transform(X_train)
    
    X_scaled_test = scaler.transform(X_test)
    
    X_scaled_train = pd.DataFrame(X_scaled_train, columns=X_train.columns)
    X_scaled_test = pd.DataFrame(X_scaled_test, columns=X_test.columns)
    
    return (X_scaled_train, y_train), (X_scaled_test, y_test), scaler


def create_standard_scaler(
    data_paths: List[str],
    output_path: str,
    train_file: str = "Train.csv"
) -> None:
    all_features = []
    
    for path in data_paths:
        df_train = pd.read_csv(os.path.join(path, train_file))
        
        columns_to_drop = ['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port',
                           'Protocol', 'Timestamp']
        df_reduced = df_train.drop(columns_to_drop, axis=1)
        
        df_reduced.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_no_na = df_reduced.dropna(axis=0)
        
        X = df_no_na.drop(['Label'], axis=1)
        all_features.append(X)
    
    combined_features = pd.concat(all_features, axis=0)
    
    scaler = StandardScaler()
    scaler.fit(combined_features)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(scaler, output_path)
    print(f"Combined StandardScaler saved to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Data processing utilities for NANCY Federated Learning")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess data for a client")
    preprocess_parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the data directory"
    )
    preprocess_parser.add_argument(
        "--scaler-path",
        type=str,
        help="Path to the StandardScaler joblib file"
    )
    preprocess_parser.add_argument(
        "--create-scaler",
        action="store_true",
        help="Whether to create a new scaler"
    )
    
    scaler_parser = subparsers.add_parser("create-scaler", help="Create a StandardScaler from multiple clients")
    scaler_parser.add_argument(
        "--data-paths",
        type=str,
        nargs="+",
        required=True,
        help="Paths to client data directories"
    )
    scaler_parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save the scaler"
    )
    
    args = parser.parse_args()
    
    if args.command == "preprocess":
        print(f"Preprocessing data from {args.data_path}...")
        (X_train, y_train), (X_test, y_test), _ = preprocess_data(
            data_path=args.data_path,
            scaler_path=args.scaler_path,
            create_scaler=args.create_scaler
        )
        
    elif args.command == "create-scaler":
        print(f"Creating StandardScaler from {len(args.data_paths)} clients...")
        create_standard_scaler(
            data_paths=args.data_paths,
            output_path=args.output_path
        )
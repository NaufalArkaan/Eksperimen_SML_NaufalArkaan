"""
Automated Preprocessing Script for Bank Marketing Dataset
Project: Dicoding Final Project - Sistem Machine Learning MLOps (Kriteria 1 Skilled)
Author: NaufalArkaan
File: preprocessing/automate_NaufalArkaan.py
"""

import os
import pandas as pd
from sklearn.preprocessing import RobustScaler


def load_data(file_path: str) -> pd.DataFrame:
    """
    Memuat raw dataset CSV dengan delimiter ';'
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File raw dataset tidak ditemukan di: {file_path}")
    print(f"[1/5] Memuat raw dataset dari: {file_path}")
    df = pd.read_csv(file_path, sep=';')
    print(f"      Data berhasil dimuat ({df.shape[0]} baris, {df.shape[1]} kolom)")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pembersihan data:
    1. Menghapus baris duplikat.
    2. Imputasi nilai 'unknown' pada kolom 'job' dan 'education' menggunakan modus.
    3. Binary encoding untuk kolom 'default', 'housing', 'loan', dan target 'y'.
    """
    print("[2/5] Melakukan pembersihan data (Cleaning)...")
    df_clean = df.copy()

    # 1. Menghapus duplikasi
    initial_len = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    removed_duplicates = initial_len - len(df_clean)
    print(f"      - Duplikasi dihapus: {removed_duplicates} baris")

    # 2. Imputasi 'unknown' dengan modus kategori yang valid
    job_mode = df_clean[df_clean['job'] != 'unknown']['job'].mode()[0]
    edu_mode = df_clean[df_clean['education'] != 'unknown']['education'].mode()[0]

    df_clean['job'] = df_clean['job'].replace('unknown', job_mode)
    df_clean['education'] = df_clean['education'].replace('unknown', edu_mode)
    print(f"      - Imputasi unknown 'job' dengan modus: '{job_mode}'")
    print(f"      - Imputasi unknown 'education' dengan modus: '{edu_mode}'")

    # 3. Encoding fitur biner dan target
    binary_cols = ['default', 'housing', 'loan']
    for col in binary_cols:
        df_clean[col] = df_clean[col].map({'yes': 1, 'no': 0}).astype(int)

    if 'y' in df_clean.columns:
        df_clean['y'] = df_clean['y'].map({'yes': 1, 'no': 0}).astype(int)

    return df_clean


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Melakukan One-Hot Encoding untuk fitur kategorikal nominal dengan drop_first=True.
    """
    print("[3/5] Melakukan One-Hot Encoding pada fitur kategorikal nominal...")
    nominal_cols = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']

    # Pastikan kolom-kolom yang ada merupakan nominal_cols
    existing_cols = [col for col in nominal_cols if col in df.columns]
    df_encoded = pd.get_dummies(df, columns=existing_cols, drop_first=True, dtype=int)
    print(f"      - Dimensi setelah One-Hot Encoding: {df_encoded.shape[0]} baris, {df_encoded.shape[1]} kolom")
    return df_encoded


def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Penskalaan fitur numerik menggunakan RobustScaler agar tahan terhadap outlier.
    """
    print("[4/5] Penskalaan fitur numerik menggunakan RobustScaler...")
    df_scaled = df.copy()
    num_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    existing_num = [col for col in num_features if col in df_scaled.columns]

    scaler = RobustScaler()
    df_scaled[existing_num] = scaler.fit_transform(df_scaled[existing_num])
    print(f"      - Penskalaan dilakukan pada {len(existing_num)} fitur numerik")
    return df_scaled


def save_processed_data(df: pd.DataFrame, output_paths: list[str]) -> None:
    """
    Menyimpan processed dataset ke lokasi output yang ditentukan.
    """
    print("[5/5] Menyimpan dataset hasil preprocessing...")
    for out_path in output_paths:
        out_dir = os.path.dirname(out_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"      - Berhasil disimpan ke: {out_path}")


def automate_preprocessing(input_file: str, output_paths: list[str]) -> pd.DataFrame:
    """
    Pipeline automation lengkap dari raw dataset menjadi processed dataset.
    """
    df_raw = load_data(input_file)
    df_clean = clean_data(df_raw)
    df_encoded = encode_features(df_clean)
    df_processed = scale_features(df_encoded)
    save_processed_data(df_processed, output_paths)
    return df_processed


def main():
    # Menentukan lokasi file input (raw) dan file output (processed)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Cari file raw dataset
    input_file_candidates = [
        os.path.join(base_dir, "dataset", "bank-full.csv"),
        os.path.join(base_dir, "preprocessing", "bank-full.csv"),
        os.path.join(base_dir, "bank-full.csv")
    ]

    input_file = None
    for path in input_file_candidates:
        if os.path.exists(path):
            input_file = path
            break

    if not input_file:
        raise FileNotFoundError("Raw dataset 'bank-full.csv' tidak ditemukan.")

    output_paths = [
        os.path.join(base_dir, "preprocessing", "bank-full_preprocessing.csv"),
        os.path.join(base_dir, "bank-full_preprocessing.csv")
    ]

    df_processed = automate_preprocessing(input_file, output_paths)

    print("\n" + "=" * 50)
    print("AUTOMATION PREPROCESSING SELESAI")
    print("=" * 50)
    print(f"Shape final dataset : {df_processed.shape}")
    print(f"Jumlah Missing Value: {df_processed.isnull().sum().sum()}")
    if 'y' in df_processed.columns:
        print(f"Distribusi Target y :\n{df_processed['y'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()

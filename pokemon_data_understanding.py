

# %% 1. Import library
from pathlib import Path

import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# %% 2. Load dataset
# Path dibuat relatif terhadap lokasi file ini, jadi aman dijalankan dari mana saja
BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
CSV_PATH = BASE_DIR / "pokemon_complete_stats.csv"

df = pd.read_csv(CSV_PATH)

print("=" * 70)
print("GAMBARAN UMUM DATASET")
print("=" * 70)
print(f"Jumlah baris  : {df.shape[0]}")
print(f"Jumlah kolom  : {df.shape[1]}")
print("\n5 baris pertama:")
print(df.head())

# %% 3. Klasifikasi otomatis berdasarkan tipe data pandas
numeric_cols = df.select_dtypes(include="number").columns.tolist()
bool_cols = df.select_dtypes(include="bool").columns.tolist()
object_cols = df.select_dtypes(exclude=["number", "bool"]).columns.tolist()

print("\n" + "=" * 70)
print("KLASIFIKASI OTOMATIS (berdasarkan dtype)")
print("=" * 70)
print("Numerik     :", numeric_cols)
print("Boolean     :", bool_cols)
print("Teks/Object :", object_cols)

# %% 4. Klasifikasi detail (penyesuaian manual berdasarkan makna kolom)
# Kolom identitas: unik di setiap baris, tidak dipakai sebagai fitur
identifier_cols = ["id", "name"]

# Numerikal (skala rasio)
numerical_cols = [
    "height_dm", "weight_hg", "base_experience",
    "hp", "attack", "defense", "special_attack", "special_defense", "speed",
    "capture_rate",
]

# Kategorikal nominal (tanpa urutan)
nominal_cols = ["type_1", "type_2", "color", "shape", "abilities"]

# Kategorikal ordinal (ada urutan)
ordinal_cols = ["generation"]

# Kategorikal biner (hanya True/False)
binary_cols = ["is_legendary", "is_mythical"]

# Validasi: pastikan semua kolom sudah terklasifikasi
classified = identifier_cols + numerical_cols + nominal_cols + ordinal_cols + binary_cols
unclassified = [c for c in df.columns if c not in classified]
if unclassified:
    print("\nPERINGATAN - kolom belum diklasifikasi:", unclassified)

# %% 5. Tabel ringkasan jenis data
rows = []
for col in df.columns:
    if col in identifier_cols:
        kelompok, sub = "Identitas", "-"
    elif col in numerical_cols:
        kelompok, sub = "Numerikal", "Rasio"
    elif col in nominal_cols:
        kelompok, sub = "Kategorikal", "Nominal"
    elif col in ordinal_cols:
        kelompok, sub = "Kategorikal", "Ordinal"
    elif col in binary_cols:
        kelompok, sub = "Kategorikal", "Biner"
    else:
        kelompok, sub = "Belum diklasifikasi", "-"

    rows.append({
        "kolom": col,
        "dtype": str(df[col].dtype),
        "kelompok": kelompok,
        "sub_jenis": sub,
        "nilai_unik": df[col].nunique(),
        "missing": df[col].isna().sum(),
        "contoh_nilai": ", ".join(map(str, df[col].dropna().unique()[:3])),
    })

summary = pd.DataFrame(rows)

print("\n" + "=" * 70)
print("RINGKASAN JENIS DATA")
print("=" * 70)
print(summary.to_string(index=False))

# %% 6. Hitung jumlah kolom per kelompok
print("\nJumlah kolom per kelompok:")
print(summary["kelompok"].value_counts().to_string())

print("\nJumlah kolom per sub-jenis:")
print(summary.groupby(["kelompok", "sub_jenis"]).size().to_string())

# %% 7. Cek missing value
missing = df.isna().sum()
missing = missing[missing > 0].to_frame("jumlah_missing")
missing["persen"] = (missing["jumlah_missing"] / len(df) * 100).round(2)

print("\n" + "=" * 70)
print("MISSING VALUE")
print("=" * 70)
print(missing.to_string())

# %% 8. Detail masing-masing kelompok
print("\n" + "=" * 70)
print("STATISTIK DESKRIPTIF - KOLOM NUMERIKAL")
print("=" * 70)
print(df[numerical_cols].describe().round(2).T.to_string())

print("\n" + "=" * 70)
print("DISTRIBUSI - KOLOM KATEGORIKAL")
print("=" * 70)
for col in ["type_1", "color", "shape", "generation", "is_legendary", "is_mythical"]:
    print(f"\n[{col}]")
    print(df[col].value_counts().to_string())

# %% 9. Simpan ringkasan ke CSV (opsional)
OUTPUT_PATH = BASE_DIR / "ringkasan_jenis_data.csv"
summary.to_csv(OUTPUT_PATH, index=False)
print(f"\nRingkasan disimpan ke: {OUTPUT_PATH}")

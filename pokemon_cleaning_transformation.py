"""
DATA MINING - DATA CLEANING DAN TRANSFORMASI DATA
Dataset: pokemon_complete_stats.csv

Alur:
  A. Profiling data awal
  B. Data Cleaning
       B1. Duplikat
       B2. Standardisasi teks
       B3. Validasi nilai (aturan domain)
       B4. Penanganan missing value
       B5. Deteksi outlier (IQR)
       B6. Konversi tipe data
  C. Transformasi Data
       C1. Konversi satuan
       C2. Pemecahan kolom (abilities)
       C3. Pembuatan fitur baru (feature engineering)
       C4. Diskretisasi / binning
       C5. Encoding kategorikal
       C6. Perbaikan skewness (log transform)
       C7. Normalisasi (Min-Max dan Z-score)
  D. Simpan hasil + log pembersihan

"""

# %% 0. Import & konfigurasi
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
CSV_PATH = BASE_DIR / "pokemon_complete_stats.csv"
OUT_DIR = BASE_DIR / "output_cleaning"
OUT_DIR.mkdir(exist_ok=True)

BUAT_GRAFIK = True  # ubah ke False jika tidak ingin membuat grafik

# Pengelompokan kolom (hasil klasifikasi jenis data sebelumnya)
KOLOM_TEKS = ["name", "type_1", "type_2", "abilities", "generation", "color", "shape"]
KOLOM_NUMERIK = [
    "height_dm", "weight_hg", "base_experience",
    "hp", "attack", "defense", "special_attack", "special_defense", "speed",
    "capture_rate",
]
KOLOM_STAT = ["hp", "attack", "defense", "special_attack", "special_defense", "speed"]

# Log semua perubahan yang dilakukan (praktik baik: dokumentasikan setiap langkah)
log_perubahan = []


def catat(tahap, aksi, jumlah, keterangan=""):
    log_perubahan.append(
        {"tahap": tahap, "aksi": aksi, "jumlah_baris_terdampak": jumlah, "keterangan": keterangan}
    )
    print(f"  [{tahap}] {aksi}: {jumlah} baris. {keterangan}")


def judul(teks):
    print("\n" + "=" * 72)
    print(teks)
    print("=" * 72)


# %% A. Load & profiling data awal
judul("A. PROFILING DATA AWAL")

df_raw = pd.read_csv(CSV_PATH)
df = df_raw.copy()  # data mentah tidak diubah, semua proses dikerjakan di salinan

print(f"Dimensi data mentah : {df_raw.shape[0]} baris x {df_raw.shape[1]} kolom")
print("\nMissing value per kolom:")
mv = df_raw.isna().sum()
print(mv[mv > 0].to_string() if mv.sum() else "  tidak ada")
print(f"\nDuplikat baris penuh : {df_raw.duplicated().sum()}")
print(f"Duplikat id          : {df_raw['id'].duplicated().sum()}")
print(f"Duplikat name        : {df_raw['name'].duplicated().sum()}")

# %% B1. Cleaning - duplikat
judul("B1. CLEANING - DUPLIKAT")

n0 = len(df)
df = df.drop_duplicates().drop_duplicates(subset="id").reset_index(drop=True)
catat("B1", "Hapus duplikat", n0 - len(df), "duplikat baris penuh dan duplikat id")

# %% B2. Cleaning - standardisasi teks
judul("B2. CLEANING - STANDARDISASI TEKS")

n_berubah = 0
for kol in KOLOM_TEKS:
    asli = df[kol]
    baru = asli.str.strip().str.lower()
    # normalisasi spasi ganda pada kolom abilities ("a,  b" -> "a, b")
    if kol == "abilities":
        baru = baru.str.replace(r"\s*,\s*", ", ", regex=True)
    n_berubah += int((asli.fillna("") != baru.fillna("")).sum())
    df[kol] = baru
catat("B2", "Standardisasi teks (strip + lowercase)", n_berubah, "0 berarti teks sudah konsisten")

# %% B3. Cleaning - validasi nilai (aturan domain)
judul("B3. CLEANING - VALIDASI NILAI")

aturan = {
    "height_dm > 0": df["height_dm"] <= 0,
    "weight_hg > 0": df["weight_hg"] <= 0,
    "stat dasar 1-255": (df[KOLOM_STAT] < 1).any(axis=1) | (df[KOLOM_STAT] > 255).any(axis=1),
    "capture_rate 1-255": (df["capture_rate"] < 1) | (df["capture_rate"] > 255),
}
for nama_aturan, pelanggar in aturan.items():
    print(f"  Aturan '{nama_aturan}': {int(pelanggar.sum())} pelanggaran")

# weight_hg = 0 tidak masuk akal secara fisik -> dianggap missing agar bisa diimputasi
inval_w = df["weight_hg"] <= 0
if inval_w.any():
    print("  Baris dengan berat 0:", df.loc[inval_w, "name"].tolist())
df.loc[inval_w, "weight_hg"] = np.nan
catat("B3", "Berat tidak valid (<=0) diubah jadi NaN", int(inval_w.sum()))

# %% B4. Cleaning - missing value
judul("B4. CLEANING - MISSING VALUE")

# type_2 kosong = Pokemon memang hanya punya satu tipe (bukan data hilang) -> isi 'none'
n = int(df["type_2"].isna().sum())
df["type_2"] = df["type_2"].fillna("none")
catat("B4", "type_2 NaN -> 'none'", n, "kosong berarti tipe tunggal")

# abilities kosong -> 'unknown'
n = int(df["abilities"].isna().sum())
df["abilities"] = df["abilities"].fillna("unknown")
catat("B4", "abilities NaN -> 'unknown'", n)

# base_experience: buat flag dulu agar jejak imputasi tidak hilang, lalu imputasi median
df["base_experience_imputed"] = df["base_experience"].isna().astype(int)
n = int(df["base_experience"].isna().sum())
med_be = df["base_experience"].median()
df["base_experience"] = df["base_experience"].fillna(med_be)
catat("B4", "base_experience NaN -> median", n, f"median = {med_be}")

# weight_hg hasil validasi: imputasi median
n = int(df["weight_hg"].isna().sum())
med_w = df["weight_hg"].median()
df["weight_hg"] = df["weight_hg"].fillna(med_w)
catat("B4", "weight_hg NaN -> median", n, f"median = {med_w}")

print("\nSisa missing value:", int(df.isna().sum().sum()))

# %% B5. Cleaning - deteksi outlier (IQR)
judul("B5. CLEANING - DETEKSI OUTLIER (METODE IQR)")

ringkas_outlier = []
flag_outlier = pd.DataFrame(index=df.index)
for kol in KOLOM_NUMERIK:
    q1, q3 = df[kol].quantile([0.25, 0.75])
    iqr = q3 - q1
    batas_bawah, batas_atas = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    flag_outlier[kol] = (df[kol] < batas_bawah) | (df[kol] > batas_atas)
    ringkas_outlier.append(
        {
            "kolom": kol,
            "batas_bawah": round(batas_bawah, 2),
            "batas_atas": round(batas_atas, 2),
            "jumlah_outlier": int(flag_outlier[kol].sum()),
            "persen": round(flag_outlier[kol].mean() * 100, 2),
        }
    )
tabel_outlier = pd.DataFrame(ringkas_outlier)
print(tabel_outlier.to_string(index=False))

# Keputusan: outlier TIDAK dihapus. Nilai ekstrem di dataset ini valid secara domain
# (Pokemon legendary, bentuk Gigantamax, dll). Sebagai gantinya dibuat penanda
# jumlah kolom outlier per baris, dan skewness ditangani lewat transformasi log (C6).
df["n_outlier_fields"] = flag_outlier.sum(axis=1)
catat("B5", "Outlier ditandai, tidak dihapus", int((df["n_outlier_fields"] > 0).sum()),
      "nilai ekstrem valid secara domain")

# %% B6. Cleaning - konversi tipe data
judul("B6. CLEANING - KONVERSI TIPE DATA")

urutan_gen = ["generation-i", "generation-ii", "generation-iii", "generation-iv",
              "generation-v", "generation-vi", "generation-vii", "generation-viii",
              "generation-ix"]
df["generation"] = pd.Categorical(df["generation"], categories=urutan_gen, ordered=True)
for kol in ["type_1", "type_2", "color", "shape"]:
    df[kol] = df[kol].astype("category")
for kol in ["is_legendary", "is_mythical"]:
    df[kol] = df[kol].astype(bool)
df["base_experience"] = df["base_experience"].round().astype(int)
df["weight_hg"] = df["weight_hg"].round().astype(int)
catat("B6", "Konversi tipe data", len(df), "category, ordered category, bool, int")

df_clean = df.copy()

# %% C1-C3. Transformasi - satuan, pemecahan kolom, fitur baru
judul("C1-C3. TRANSFORMASI - SATUAN, PEMECAHAN KOLOM, FITUR BARU")

dft = df_clean.copy()

# C1. Konversi satuan ke satuan yang lazim
dft["height_m"] = dft["height_dm"] / 10   # desimeter -> meter
dft["weight_kg"] = dft["weight_hg"] / 10  # hektogram -> kilogram

# C2. Pecah kolom abilities menjadi ability_1..ability_3 + jumlah ability
daftar_ability = dft["abilities"].str.split(", ")
for i in range(3):
    dft[f"ability_{i + 1}"] = daftar_ability.str[i].fillna("none")
dft["n_abilities"] = np.where(
    dft["abilities"] == "unknown", 0, daftar_ability.str.len()
)

# C3. Fitur baru
dft["total_stats"] = dft[KOLOM_STAT].sum(axis=1)
dft["offense"] = dft["attack"] + dft["special_attack"]
dft["defensive"] = dft["hp"] + dft["defense"] + dft["special_defense"]
dft["bmi"] = (dft["weight_kg"] / (dft["height_m"] ** 2)).round(2)
dft["is_dual_type"] = (dft["type_2"].astype(str) != "none").astype(int)
dft["generation_num"] = dft["generation"].cat.codes + 1  # 1..9

# id > 10000 pada PokeAPI = bentuk alternatif (mega, gmax, regional, dll.)
dft["is_alt_form"] = (dft["id"] > 10000).astype(int)
nama = dft["name"]
dft["form_type"] = np.select(
    [
        nama.str.contains("-mega"),
        nama.str.contains("-gmax|-eternamax"),
        nama.str.contains("-alola|-galar|-hisui|-paldea"),
        dft["is_alt_form"] == 1,
    ],
    ["mega", "gigantamax", "regional", "other_form"],
    default="base",
)
print("Fitur baru ditambahkan: height_m, weight_kg, ability_1..3, n_abilities, total_stats,")
print("  offense, defensive, bmi, is_dual_type, generation_num, is_alt_form, form_type")
print("\nDistribusi form_type:")
print(dft["form_type"].value_counts().to_string())

# %% C4. Transformasi - binning
judul("C4. TRANSFORMASI - DISKRETISASI (BINNING)")

# Equal-frequency binning: total_stats dibagi 4 kelompok dengan jumlah data seimbang
dft["stat_tier"] = pd.qcut(
    dft["total_stats"], q=4, labels=["rendah", "sedang", "tinggi", "sangat_tinggi"]
)
# Equal-width binning manual untuk ukuran tubuh
dft["size_class"] = pd.cut(
    dft["height_m"], bins=[0, 0.5, 1.0, 2.0, np.inf],
    labels=["kecil", "sedang", "besar", "raksasa"],
)
print(dft["stat_tier"].value_counts().sort_index().to_string())
print()
print(dft["size_class"].value_counts().sort_index().to_string())

# %% C5-C7. Transformasi - skewness, normalisasi, encoding
judul("C5-C7. TRANSFORMASI - LOG, NORMALISASI, ENCODING")

# C6. Skewness: data height dan weight sangat miring ke kanan -> log1p
skew_sebelum = dft[["height_m", "weight_kg"]].skew()
dft["height_log"] = np.log1p(dft["height_m"])
dft["weight_log"] = np.log1p(dft["weight_kg"])
skew_sesudah = dft[["height_log", "weight_log"]].skew()
print("Skewness sebelum log1p :", skew_sebelum.round(2).to_dict())
print("Skewness sesudah log1p :", skew_sesudah.round(2).to_dict())

# C7. Normalisasi
fitur_skala = [
    "height_log", "weight_log", "base_experience",
    "hp", "attack", "defense", "special_attack", "special_defense", "speed",
    "capture_rate", "total_stats", "bmi",
]
minmax = (dft[fitur_skala] - dft[fitur_skala].min()) / (dft[fitur_skala].max() - dft[fitur_skala].min())
minmax.columns = [f"{c}_minmax" for c in fitur_skala]

zscore = (dft[fitur_skala] - dft[fitur_skala].mean()) / dft[fitur_skala].std(ddof=0)
zscore.columns = [f"{c}_z" for c in fitur_skala]

print("\nContoh hasil Min-Max (rentang 0-1):")
print(minmax.describe().loc[["min", "max"]].round(2).iloc[:, :4].to_string())
print("\nContoh hasil Z-score (mean ~0, std ~1):")
print(zscore.describe().loc[["mean", "std"]].round(2).iloc[:, :4].to_string())

# C5. Encoding kategorikal
# - One-hot untuk nominal (type_1, type_2, color, shape, form_type)
# - Ordinal sudah diwakili generation_num
# - Biner dikonversi ke 0/1
onehot = pd.get_dummies(
    dft[["type_1", "type_2", "color", "shape", "form_type"]].astype(str),
    prefix=["t1", "t2", "color", "shape", "form"],
    dtype=int,
)
biner = dft[["is_legendary", "is_mythical"]].astype(int)

# Dataset siap-model: hanya angka (cocok untuk clustering / klasifikasi / korelasi)
df_model = pd.concat(
    [
        dft[["id", "name"]],
        minmax,
        dft[["generation_num", "is_dual_type", "is_alt_form", "n_abilities"]],
        biner,
        onehot,
    ],
    axis=1,
)
print(f"\nDataset siap-model: {df_model.shape[0]} baris x {df_model.shape[1]} kolom")
print(f"Kolom hasil one-hot : {onehot.shape[1]}")

# %% D. Perbandingan sebelum vs sesudah
judul("D. PERBANDINGAN SEBELUM VS SESUDAH")

perbandingan = pd.DataFrame(
    {
        "sebelum": [df_raw.shape[0], df_raw.shape[1], int(df_raw.isna().sum().sum())],
        "setelah_cleaning": [df_clean.shape[0], df_clean.shape[1], int(df_clean.isna().sum().sum())],
        "setelah_transformasi": [dft.shape[0], dft.shape[1], int(dft.isna().sum().sum())],
        "siap_model": [df_model.shape[0], df_model.shape[1], int(df_model.isna().sum().sum())],
    },
    index=["jumlah_baris", "jumlah_kolom", "total_missing"],
)
print(perbandingan.to_string())

# %% E. Simpan hasil
judul("E. MENYIMPAN HASIL")

df_clean.to_csv(OUT_DIR / "pokemon_cleaned.csv", index=False)
dft.to_csv(OUT_DIR / "pokemon_transformed.csv", index=False)
df_model.to_csv(OUT_DIR / "pokemon_model_ready.csv", index=False)
tabel_outlier.to_csv(OUT_DIR / "ringkasan_outlier.csv", index=False)
pd.DataFrame(log_perubahan).to_csv(OUT_DIR / "cleaning_log.csv", index=False)

for f in sorted(OUT_DIR.glob("*.csv")):
    print(f"  tersimpan: {f.relative_to(BASE_DIR)}")

# %% F. Grafik (opsional)
if BUAT_GRAFIK:
    try:
        import matplotlib

        matplotlib.use("Agg")  # simpan ke file, tidak perlu jendela
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(2, 2, figsize=(11, 8))
        ax[0, 0].hist(dft["weight_kg"], bins=40, color="#4C78A8")
        ax[0, 0].set_title("weight_kg (sebelum log)")
        ax[0, 1].hist(dft["weight_log"], bins=40, color="#54A24B")
        ax[0, 1].set_title("weight_log (sesudah log1p)")
        ax[1, 0].boxplot([dft[c] for c in KOLOM_STAT], tick_labels=KOLOM_STAT)
        ax[1, 0].set_title("Boxplot base stats (outlier)")
        ax[1, 0].tick_params(axis="x", rotation=45)
        dft["stat_tier"].value_counts().sort_index().plot(kind="bar", ax=ax[1, 1], color="#F58518")
        ax[1, 1].set_title("Distribusi stat_tier (binning)")
        ax[1, 1].tick_params(axis="x", rotation=0)
        plt.tight_layout()
        fig.savefig(OUT_DIR / "grafik_cleaning_transformasi.png", dpi=120)
        print("  tersimpan: output_cleaning/grafik_cleaning_transformasi.png")
    except ImportError:
        print("  matplotlib belum terpasang, grafik dilewati (pip install matplotlib)")

print("\nSelesai.")

# KLASIFIKASI POKEMON LEGENDARY DENGAN DECISION TREE

# TUJUAN:
#   Membuat model yang menebak apakah sebuah Pokemon berstatus Legendary
#   (is_legendary = True) atau bukan (False) berdasarkan data angkanya,
#   lalu mencari fitur apa yang paling menentukan status tersebut.


# 1. IMPORT LIBRARY
import pandas as pd                 # membaca dan mengolah data berbentuk tabel
import matplotlib.pyplot as plt     # membuat grafik dan gambar pohon

from sklearn.model_selection import train_test_split
# train_test_split: memisahkan data menjadi data latih
# dan data uji (untuk menguji model pada data yang belum pernah dilihat).

from sklearn.tree import DecisionTreeClassifier, plot_tree
# DecisionTreeClassifier: algoritma Decision Tree untuk klasifikasi.
# plot_tree: fungsi untuk menggambar pohon keputusan yang sudah dilatih.

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# accuracy_score        : persentase tebakan yang benar dari seluruh data uji.
# confusion_matrix      : tabel jumlah tebakan benar dan salah per kelas.
# classification_report : ringkasan precision, recall, dan f1-score per kelas.


# 2. PENGATURAN
NAMA_FILE = "pokemon_complete_stats.csv"   # nama file dataset

PAKAI_CAPTURE_RATE = True
# True  : capture_rate ikut dipakai sebagai fitur untuk akurasi lebih tinggi (sekitar 96%),
#         tetapi capture_rate hampir identik dengan status Legendary
# False : hanya memakai 6 stat tempur -> lebih "adil" untuk menguji apakah stat
#         saja cukup membedakan Legendary (akurasi sekitar 88%).

KEDALAMAN_POHON = 4    # batas kedalaman pohon; makin dalam makin rumit dan rawan overfitting
UKURAN_UJI = 0.2       # 20% data untuk pengujian, 80% untuk pelatihan
ACAK = 42              # angka acak tetap agar hasil sama setiap kali dijalankan


# 3. MEMBACA DATA DAN MELAKUKAN BREAKDOWN AWAL
df = pd.read_csv(NAMA_FILE)    # membaca CSV menjadi DataFrame bernama df

print("=" * 60)
print("BREAKDOWN DATASET")
print("=" * 60)

# shape menghasilkan (jumlah baris, jumlah kolom)
print("Jumlah baris dan kolom :", df.shape)

# 5 baris pertama, untuk memastikan data terbaca dengan benar
print("\n5 baris pertama:")
print(df.head())

# info() menampilkan nama kolom, tipe data, dan jumlah nilai yang terisi.
# Berguna untuk mengecek apakah ada kolom yang tipenya tidak sesuai.
print("\nInformasi kolom:")
df.info()

# Menghitung nilai kosong (missing value) di tiap kolom.
print("\nJumlah nilai kosong per kolom:")
print(df.isnull().sum())

# Melihat sebaran kelas target. Hasilnya menunjukkan data tidak seimbang:
# False (bukan Legendary) jauh lebih banyak daripada True (Legendary).
print("\nSebaran kelas is_legendary:")
print(df["is_legendary"].value_counts())
print("\nPersentase:")
print((df["is_legendary"].value_counts(normalize=True) * 100).round(1))

# Membandingkan rata-rata tiap stat antara Pokemon biasa dan Legendary.
kolom_stat = ["hp", "attack", "defense", "special_attack",
              "special_defense", "speed", "capture_rate"]
print("\nRata-rata stat berdasarkan status Legendary:")
print(df.groupby("is_legendary")[kolom_stat].mean().round(1).T)


# 4. MEMILIH FITUR (X) DAN TARGET (y)
# Fitur  : kolom yang dipakai model untuk belajar (input).
# Target : kolom yang ingin diprediksi (output).

# Kolom yang SENGAJA tidak dipakai:
#   - id, name              : hanya identitas, tidak punya makna untuk prediksi
#   - is_mythical           : status khusus yang berkaitan dengan target
#   - type_1, type_2, color, shape, abilities, generation : berupa teks/kategori,
#     perlu encoding tambahan dan tidak diperlukan untuk contoh ini
fitur = ["hp", "attack", "defense", "special_attack", "special_defense", "speed"]

if PAKAI_CAPTURE_RATE:
    fitur.append("capture_rate")   # tambahkan capture_rate bila pengaturan di atas True

X = df[fitur]               # X = tabel fitur
y = df["is_legendary"]      # y = label yang diprediksi (True / False)

print("\nFitur yang dipakai:", fitur)


# 5. MEMBAGI DATA LATIH DAN DATA UJI
# Model dilatih dengan data latih lalu diuji dengan data uji yang belum pernah
# dilihat. Dengan begitu kita tahu kemampuan model pada data baru, bukan
# sekadar menghafal data latih.
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=UKURAN_UJI,   # proporsi data uji
    random_state=ACAK,      # membuat pembagian acak yang bisa diulang
    stratify=y,             # PENTING: menjaga proporsi Legendary (~9%) tetap sama
                            # di data latih dan data uji. Tanpa ini, data uji bisa
                            # kebetulan hampir tidak berisi Legendary sama sekali.
)

print("\nJumlah data latih :", len(X_train))
print("Jumlah data uji   :", len(X_test))
print("Legendary di data uji:", int(y_test.sum()))


# 6. MEMBUAT DAN MELATIH MODEL
model = DecisionTreeClassifier(
    criterion="gini",
    # Ukuran untuk memilih pemisahan terbaik di tiap node.
    # Gini mengukur seberapa "campur" kelas di sebuah node (0 = murni satu kelas).
    # Alternatifnya "entropy".

    max_depth=KEDALAMAN_POHON,
    # Membatasi kedalaman pohon. Tanpa batas, pohon cenderung menghafal data
    # latih (overfitting) dan sulit dibaca.

    class_weight="balanced",
    # Karena Legendary jarang, kesalahan pada kelas Legendary diberi bobot lebih
    # besar secara otomatis. Tujuannya agar model tidak mengabaikan kelas minoritas.

    random_state=ACAK,
    # Agar pohon yang dihasilkan konsisten setiap dijalankan.
)

# fit() adalah proses belajar: model mencari aturan pemisah terbaik
# dari pasangan fitur (X_train) dan label (y_train).
model.fit(X_train, y_train)


# 7. PREDIKSI DAN EVALUASI MODEL
# predict() membuat model menebak label untuk data uji.
y_pred = model.predict(X_test)

print("\n" + "=" * 60)
print("HASIL EVALUASI")
print("=" * 60)

# Akurasi: persentase tebakan benar.
print("Akurasi:", round(accuracy_score(y_test, y_pred), 4))

# Confusion matrix (urutan kelas: False lalu True):
#   [[TN  FP]      TN = bukan Legendary, ditebak bukan Legendary (benar)
#    [FN  TP]]     FP = bukan Legendary, ditebak Legendary (salah)
#                  FN = Legendary, ditebak bukan Legendary (terlewat)
#                  TP = Legendary, ditebak Legendary (benar)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification report, fokus pada baris "True" (kelas Legendary):
#   precision : dari semua yang ditebak Legendary, berapa yang benar-benar Legendary
#   recall    : dari semua Legendary asli, berapa yang berhasil terdeteksi
#   f1-score  : rata-rata harmonis precision dan recall
#   support   : jumlah data asli per kelas di data uji
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Biasa", "Legendary"]))


# 8. FITUR PALING BERPENGARUH
# feature_importances_ menunjukkan seberapa besar kontribusi tiap fitur dalam
# mengurangi ketidakmurnian (gini) di seluruh pohon. Total semuanya = 1.
pentingnya = pd.Series(model.feature_importances_, index=fitur)

print("=" * 60)
print("FITUR PALING BERPENGARUH")
print("=" * 60)
print(pentingnya.sort_values(ascending=False).round(4))


# 9. VISUALISASI POHON KEPUTUSAN
plt.figure(figsize=(18, 9))          # ukuran gambar (lebar, tinggi) dalam inci
plot_tree(
    model,
    feature_names=fitur,                    # nama fitur ditampilkan di tiap node
    class_names=["Biasa", "Legendary"],     # urutan sesuai False lalu True
    filled=True,                            # beri warna sesuai kelas mayoritas
    rounded=True,                           # sudut kotak dibuat membulat
    fontsize=9,
)
plt.title("Decision Tree - Klasifikasi Pokemon Legendary")
plt.savefig("decision_tree_legendary.png", dpi=150, bbox_inches="tight")
print("Gambar pohon disimpan sebagai decision_tree_legendary.png")
plt.show()

# Cara membaca kotak (node) pada gambar:
#   - baris pertama : kondisi pemisah. Jika benar -> cabang kiri, jika salah -> kanan
#   - gini          : tingkat campuran kelas (0 = murni)
#   - samples       : jumlah data yang sampai ke node itu
#   - value         : jumlah (berbobot) data per kelas [Biasa, Legendary]
#   - class         : kelas prediksi di node itu (kelas mayoritas)
# Node paling bawah (daun) adalah keputusan akhir.


# 10. MENCOBA PREDIKSI POKEMON BARU (CONTOH)
# Kita buat satu Pokemon rekaan dengan stat tinggi untuk melihat tebakan model.
# Kolom harus sama persis dengan fitur yang dipakai saat melatih.
contoh = {
    "hp": 100, "attack": 110, "defense": 100,
    "special_attack": 110, "special_defense": 100, "speed": 95,
    "capture_rate": 3,
}
contoh_df = pd.DataFrame([contoh])[fitur]   # [fitur] memastikan urutan kolom sama

hasil = model.predict(contoh_df)[0]
peluang = model.predict_proba(contoh_df)[0]  # peluang [Biasa, Legendary]

print("\n" + "=" * 60)
print("PREDIKSI POKEMON CONTOH")
print("=" * 60)
print("Hasil tebakan      :", "Legendary" if hasil else "Biasa")
print("Peluang Legendary  :", round(peluang[1] * 100, 1), "%")
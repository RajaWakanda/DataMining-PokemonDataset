"Tujuan klasifikasi adalah memprediksi status Legendary suatu Pokemon berdasarkan atribut statistiknya, serta mengidentifikasi atribut yang paling berpengaruh terhadap status tersebut."

Tujuannya: membangun model yang bisa menebak apakah sebuah Pokemon berstatus Legendary atau bukan, hanya dari data angkanya (stat tempur, capture rate, dan sebagainya), tanpa perlu diberi tahu statusnya.

Secara praktis ada tiga hal yang bisa dicari dengan target ini:

Mencari pola pembeda. Apa yang membuat Pokemon Legendary "berbeda" dari yang biasa? Dari hasil tadi, ciri utamanya: stat lebih tinggi di semua aspek (terutama HP dan Speed di model stat saja) dan capture rate sangat rendah. Pohon keputusan menampilkannya sebagai aturan yang mudah dibaca, misalnya "jika capture_rate <= 5 maka Legendary".
Mengetahui fitur paling berpengaruh. Lewat feature importance kamu bisa menjawab: fitur mana yang paling menentukan status Legendary? Di datasetmu, capture_rate paling dominan, lalu HP.
Menguji kemampuan prediksi pada data tidak seimbang. Legendary hanya sekitar 9% dari data, jadi ini latihan bagus untuk memahami kenapa akurasi saja menyesatkan. Model yang selalu menebak "bukan Legendary" sudah akurat 91%, tapi tidak berguna. Karena itu kita melihat recall dan precision kelas Legendary.

Contoh penerapannya: kalau ada Pokemon baru yang statusnya belum diketahui, model bisa memperkirakan apakah ia kemungkinan Legendary. Di dunia nyata, konsepnya sama dengan mendeteksi hal langka seperti penipuan transaksi, penyakit langka, atau pelanggan yang akan berhenti berlangganan, yaitu kasus di mana kelas yang dicari jumlahnya sedikit.

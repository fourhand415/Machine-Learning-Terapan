# Laporan Proyek Machine Learning - Muhammad Farhan Lucky Putra

## Project Overview

![image-11-1024x768](https://github.com/user-attachments/assets/91a3e4a7-c1d6-4259-b9c8-329c5af73344)

Pariwisata merupakan sektor strategis yang memiliki peran penting dalam menunjang pertumbuhan ekonomi nasional, pelestarian budaya, dan pembangunan berkelanjutan di tingkat daerah. Di Indonesia, potensi wisata tersebar luas di berbagai kota, mulai dari wisata alam, sejarah, kuliner, hingga edukasi. Namun, dengan begitu banyaknya pilihan destinasi, wisatawan kerap mengalami kesulitan dalam memilih tempat yang sesuai dengan preferensi dan kebutuhan mereka. Di sinilah peran sistem rekomendasi menjadi penting—membantu menyaring informasi dan memberikan saran yang relevan secara personal.

Proyek ini bertujuan untuk mengembangkan sistem rekomendasi tempat wisata yang mencakup lima kota besar di Indonesia dan enam kategori wisata, yaitu wisata alam, sejarah, hiburan, kuliner, belanja, dan edukasi. Sistem ini akan memanfaatkan data historis, ulasan pengguna, rating, serta fitur tempat wisata untuk memberikan rekomendasi yang sesuai dengan karakteristik dan minat pengguna. Pendekatan yang digunakan menggabungkan teknik content-based filtering dan collaborative filtering, yang terbukti efektif dalam meningkatkan akurasi dan relevansi rekomendasi dalam berbagai studi sebelumnya.

Menurut Lu et al., sistem rekomendasi yang dikembangkan dalam konteks pariwisata dapat memberikan nilai tambah signifikan dalam proses pengambilan keputusan wisatawan, terutama ketika dipersonalisasi berdasarkan preferensi pengguna [[1]](https://doi.org/10.1016/j.dss.2015.03.008). Lebih lanjut, Ricci et al. menyatakan bahwa sistem rekomendasi yang mempertimbangkan konteks perjalanan, seperti lokasi pengguna, tujuan kunjungan, dan waktu kunjungan, memiliki potensi besar dalam meningkatkan pengalaman wisatawan secara keseluruhan [[2]](https://dx.doi.org/10.1007/978-0-387-85820-3_1). Oleh karena itu, pengembangan sistem rekomendasi dalam proyek ini tidak hanya bermanfaat bagi wisatawan, tetapi juga bagi pelaku industri pariwisata dan pemerintah daerah untuk menyusun strategi promosi yang lebih tepat sasaran.

Dengan adanya sistem ini, diharapkan wisatawan dapat merencanakan perjalanan dengan lebih efisien dan menyenangkan, sementara destinasi-destinasi yang kurang terekspos juga berpeluang untuk mendapatkan perhatian lebih besar dari publik.

## Business Understanding

Sistem rekomendasi dalam bidang pariwisata berperan penting dalam membantu wisatawan menemukan destinasi yang sesuai dengan preferensi pribadi mereka. Saat ini, banyak wisatawan kesulitan memilih tempat wisata karena banyaknya pilihan dan kurangnya sistem yang memberikan saran yang relevan dan personal. Oleh karena itu, perlu dilakukan klarifikasi masalah agar sistem yang dikembangkan benar-benar mampu menjawab kebutuhan pengguna dan pemangku kepentingan lainnya.

### Problem Statements

Berdasarkan uraian yang telah dipaparkan pada latar belakang diatas, maka dapat diambil sebuah rumusan masalah yang dirumuskan sebagai berikut.
- Bagaimana cara merekomendasikan tempat wisata yang sesuai dengan preferensi pengguna berdasarkan riwayat pencarian, ulasan, dan kategori wisata yang diminati?
- Bagaimana cara meningkatkan visibilitas destinasi wisata yang kurang populer tetapi potensial?

### Goals

Berdasarkan rumusan masalah yang telah dipaparkan, maka tujuan penelitian sebagai berikut. 
- Mengembangkan sistem rekomendasi wisata berbasis preferensi pengguna dengan memanfaatkan data historis, interaksi pengguna, ulasan, dan fitur dari destinasi. Ini bertujuan untuk memberikan rekomendasi yang lebih akurat dan personal.
- Meningkatkan eksposur destinasi wisata kurang populer melalui strategi rekomendasi yang mempertimbangkan keseimbangan antara popularitas dan kesesuaian preferensi pengguna, sehingga mendukung pemerataan kunjungan wisata.

### Solution Statements
Berdasarkan tujuan yang telah dipaparkan, maka penelitian ini memiliki solusi sebagai berikut.
- Content-Based Filtering
    - Sistem akan menganalisis atribut dari tempat wisata (misalnya kategori, lokasi, rating, dan kata kunci dari ulasan) dan mencocokkannya dengan preferensi pengguna berdasarkan riwayat interaksi. Pendekatan ini cocok untuk pengguna baru atau yang sudah memiliki preferensi tertentu.
- Collaborative Filtering (User-Based atau Item-Based)
    - Sistem akan merekomendasikan tempat wisata berdasarkan kemiripan antar pengguna atau antar destinasi, berdasarkan pola interaksi. Metode ini memungkinkan sistem menemukan tempat yang disukai pengguna lain dengan profil yang mirip.


## Data Understanding
Paragraf awal bagian ini menjelaskan informasi mengenai jumlah data, kondisi data, dan informasi mengenai data yang digunakan. Sertakan juga sumber atau tautan untuk mengunduh dataset. Contoh: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Restaurant+%26+consumer+data).

Selanjutnya, uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
- accepts : merupakan jenis pembayaran yang diterima pada restoran tertentu.
- cuisine : merupakan jenis masakan yang disajikan pada restoran.
- dst

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data beserta insight atau exploratory data analysis.

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model sisten rekomendasi yang Anda buat untuk menyelesaikan permasalahan. Sajikan top-N recommendation sebagai output.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menyajikan dua solusi rekomendasi dengan algoritma yang berbeda.
- Menjelaskan kelebihan dan kekurangan dari solusi/pendekatan yang dipilih.

## Evaluation
Pada bagian ini Anda perlu menyebutkan metrik evaluasi yang digunakan. Kemudian, jelaskan hasil proyek berdasarkan metrik evaluasi tersebut.

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.


## Refrensi
**[1]** J. Lu, D. Wu, M. Mao, W. Wang, and G. Zhang, "Recommender system application developments: A survey," Decision Support Systems, vol. 74, pp. 12–32, 2015. [Online]. Available: https://doi.org/10.1016/j.dss.2015.03.008 [Accessed: May 14, 2025]. [Accessed: May 28, 2025].

**[2]** F. Ricci, L. Rokach, and B. Shapira, "Introduction to Recommender Systems Handbook," in Recommender Systems Handbook, 1st ed., Springer, 2011, pp. 1–35. [Online]. Available: http://dx.doi.org/10.1007/978-0-387-85820-3_1. [Accessed: May 28, 2025].

**---Ini adalah bagian akhir laporan---**

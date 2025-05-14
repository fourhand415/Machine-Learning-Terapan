# Laporan Proyek Machine Learning - Muhammad Farhan Lucky Putra

## Domain Proyek : Kesehatan
![Gambar Kanker Payudara](https://github.com/user-attachments/assets/d94969a5-e74b-403e-a7bc-e561fdec1165)

Kanker payudara merupakan salah satu jenis kanker yang paling umum dan menjadi penyebab utama kematian akibat kanker pada perempuan di seluruh dunia. Menurut *World Health Organization* (WHO), pada tahun 2022 tercatat lebih dari 2,3 juta kasus kanker baru secara global, dengan jumlah kematian sebanyak 670 ribu kematian [[1]](https://www.who.int/news-room/fact-sheets/detail/breast-cancer). Kanker ini ditandai oleh pertumbuhan sel abnormal yang tidak terkendali pada jaringan payudara, yang dapat menyebar ke bagian tubuh lainnya jika tidak terdeteksi dan ditangani sejak dini. 

Deteksi dini dan prediksi akurat sangatlah krusial untuk meningkatkan peluang sembuh dan menekan angka kematian. Pemanfaatan teknologi machine learning dalam membantu diagnosis kanker payudara menjadi bidang penelitian yang sangat penting. Model prediktif berbasis data medis dapat meningkatkan akurasi diagnosis dan mendukung pengambilan keputusan klinis secara lebih cepat dan efisien.

Salah satu dataset yang paling banyak digunakan dalam penelitian klasifikasi kanker payudara adalah Breast Cancer Wisconsin (Diagnostic) Data Set. Berbagai studi telah berhasil mengembangkan model prediksi kanker payudara berbasis machine learning dengan tingkat akurasi tinggi menggunakan dataset ini [[2]](https://doi.org/10.36040/jati.v8i5.10752)[[3]](https://doi.org/10.1016/j.procs.2016.04.224).

## Business Understanding
### Problem Statements
Berdasarkan uraian yang telah dipaparkan pada latar belakang diatas, maka dapat diambil sebuah rumusan masalah yang dirumuskan sebagai berikut.
1. Bagaimana membangun model machine learning yang dapat mendeteksi kasus kanker payudara berdasarkan hasil diagnosanya?
2. Model machine learning apa yang memiliki akurasi tertinggi dan bagaimana cara meningkatkan akurasinya dalam kasus mendeteksi kanker payudara?

### Goals
Berdasarkan rumusan masalah yang telah dipaparkan, maka tujuan penelitian sebagai berikut. 
1. Membangun model machine learning untuk memprediksi tingkat tumor berdasarkan rekam medis diagnosanya.
2. Membandingkan beberapa model machine learning untuk menentukan model dengan performa terbaik dengan menggunakan metrik evaluasi serta melakukan hyperparameter tuning.

### Solution Statements
Berdasarkan tujuan yang telah dipaparkan, maka penelitian ini memiliki solusi sebagai berikut.
1. Menggunakan berbagai algoritma machine learning untuk membandingkan performa model, dengan tujuan mendapatkan model atau algoritma yang memiliki prediksi tinggi dalam memperkirakan tingkat tumor pada kanker payudara.
2. Menerapkan hyperparameter tuning dengan menggunakan teknik Bayes Search untuk memilih kombinasi parameter terbaik pada masing-masing algoritma. Evaluasi model dilakukan menggunakan mengukur metrik evaluasi sepeti akurasi, precision, recall, dan F1-score.

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah _Breast Cancer Wisconsin (Diagnostic) Data Set_ yang tersedia di [Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data). Dataset ini awalnya berasal dari UCI Machine Learning Repository dan digunakan secara luas dalam berbagai studi dan eksperimen klasifikasi medis, khususnya dalam mendeteksi tumor payudara.

Dataset ini memiliki 569 data dengan 30 fitur numerik yang berasal dari informasi diagnostik citologi aspirasi jarum halus (Fine Needle Aspiration, FNA) pada massa payudara. Dataset ini memiliki label target yang terdiri dari dua kelas yaitu M (Malignant) dan B (Benign), yang menunjukkan apakah tumor bersifat ganas atau jinak.

### Informasi Datasets

| Jenis | Keterangan |
| ------ | ------ |
| Title | Breast Cancer Wisconsin (Diagnostic) Data Set |
| Source | [Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) |
| Owner | [UCI Machine Learning](https://www.kaggle.com/organizations/uciml)|
| License | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)|
| Visibility | Publik |
| Tags | Cancer, Healthcare |
| Usability | 8.53 |

### Variabel-variabel pada _Breast Cancer Wisconsin (Diagnostic) Data Set_
Dataset ini berisi 30 fitur numerik yang dihasilkan dari citra inti sel kanker payudara, di mana setiap fitur merupakan hasil perhitungan terhadap karakteristik bentuk dan tekstur inti sel. Sepuluh jenis fitur dasar yang dihitung untuk setiap inti sel meliputi.
1. **Radius** : rata-rata jarak dari pusat inti ke titik-titik pada perimeter.
2. **Texture** : simpangan baku dari nilai intensitas skala abu-abu.
3. **Perimeter** : panjang keliling inti sel.
4. **Area** : luas permukaan inti sel.
5. **Smoothness** : variasi lokal panjang radius, menunjukkan seberapa halus bentuk tepi.
6. **Compactness** : dihitung sebagai (perimeter² / area − 1.0), menggambarkan seberapa padat bentuk inti.
7. **Concavity** : tingkat lekukan cekung pada kontur inti.
8. **Concave Points** : jumlah titik-titik cekung di sepanjang kontur.
9. **Symmetry** : tingkat simetri bentuk inti.
10. **Fractal Dimension** : pendekatan dimensi fraktal dengan metode “coastline approximation”, mengukur kompleksitas kontur.

Untuk masing-masing dari sepuluh fitur dasar tersebut, dihitung tiga jenis nilai statistik yaitu.
1. **Mean** : rata-rata nilai dari seluruh piksel atau elemen pengukur.
2. **Standard Error (SE)** : simpangan baku dari nilai-nilai dalam pengukuran, menunjukkan seberapa stabil nilai tersebut.
3. **Worst** : rata-rata dari tiga nilai tertinggi pada masing-masing fitur.

Dengan demikian, total terdapat 30 fitur numerik per sampel (10 fitur × 3 jenis pengukuran), yang seluruhnya telah dinormalisasi dengan presisi empat angka signifikan. Fitur-fitur ini menjadi masukan utama bagi model machine learning dalam melakukan klasifikasi antara tumor jinak dan ganas.

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis.

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

## Refrensi
**[1]** World Health Organization, “Breast cancer,” *WHO*, 2024. [Online]. Available: https://www.who.int/news-room/fact-sheets/detail/breast-cancer. [Accessed: May 14, 2025].

**[2]** M. T. T. B. Sirait, N. S. Fathonah, and M. N. Fauzan, “Pemanfaatan Algoritma ADASYN dan Support Vector Machine dalam Meningkatkan Akurasi Prediksi Kanker Paru-Paru,” JATI (Jurnal Mahasiswa Teknik Informatika), vol. 8, no. 5, pp. 8773–8778, 2024, doi: 10.36040/jati.v8i5.10752.

**[3]** H. Asri, H. Mousannif, H. A. Moatassime, and T. Noel, “Using Machine Learning Algorithms for Breast Cancer Risk Prediction and Diagnosis,” *Procedia Computer Science*, vol. 83, pp. 1064–1069, 2015, doi: 10.1016/j.procs.2016.04.224.

**---Ini adalah bagian akhir laporan---**

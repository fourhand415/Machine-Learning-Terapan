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
- Dengan menggunakan rating tempat wisata yang diberikan pengguna, bagaimana cara merekomendasikan ke pengguna?
- Bagaimana cara membangun model sistem rekomendasi menggunakan pendekatan (content-based filtering) dengan Cosine Similiarity dan (Collaborative Filtering) model based Deep learning?
- Bagaimana cara mengukur performa model sistem rekomendasi yang telah dibangun?

### Goals

Berdasarkan rumusan masalah yang telah dipaparkan, maka tujuan penelitian sebagai berikut. 
- Menghasilkan rekomendasi tempat wisata sebanyak Top-N Rekomendasi kepada pengguna berdasarkan tempat yang dicari.
- Menghasilkan rekomendasi tempat wisata yang sesuai dengan prefrensi pengguna sebelumnya.
- Membangun model sistem rekomendasi menggunakan pendekatan (content-based filtering) dengan Cosine Similiarity dan (Collaborative Filtering) model based Deep learning berdasarkan fitur yang telah dipilih dari dataset.
- Mengukur performa model sistem rekomendasi menggunakan metrik evaluasi yang sesuai.

### Solution Statements
Berdasarkan tujuan yang telah dipaparkan, maka penelitian ini memiliki solusi sebagai berikut.
- Content-Based Filtering
    - Sistem akan menganalisis atribut dari tempat wisata (misalnya kategori, lokasi, rating, dan kata kunci dari ulasan) dan mencocokkannya dengan preferensi pengguna berdasarkan riwayat interaksi. Pendekatan ini cocok untuk pengguna baru atau yang sudah memiliki preferensi tertentu.
- Collaborative Filtering
    - Sistem akan merekomendasikan tempat wisata berdasarkan kemiripan antar pengguna atau antar destinasi, berdasarkan pola interaksi. Metode ini memungkinkan sistem menemukan tempat yang disukai pengguna lain dengan profil yang mirip.
- Pembangunan Model
    - Model sistem rekomendasi dibangun dengan Content-Based Filtering dengan Cosine Similarity dan Colaborative Filtering dengan menggunakan model based Deep Learning.
- Evaluasi Performa Model
    - Evaluasi performa menggunakan Root Mean Squared Error yang akan memberikan wawasan tentang efektiviras model dalam merekomendasikan tempat wisata kepada pengguna.

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah _Indonesia Tourism Destination_ yang tersedia di [Kaggle](https://www.kaggle.com/datasets/aprabowo/indonesia-tourism-destination). Dataset ini merupakan dataset yang berisi beberapa tempat wisata di 5 kota besar di Indonesia yaitu Jakarta, Yogyakarta, Semarang, Bandung, Surabaya.

### Informasi Datasets

| Jenis      | Keterangan                                                                       |
|------------|----------------------------------------------------------------------------------|
| Title      | Indonesia Tourism Destination                                                    |
| Source     | [Kaggle](https://www.kaggle.com/datasets/aprabowo/indonesia-tourism-destination) |
| Owner      | [A_Prabowo](https://www.kaggle.com/aprabowo)                                     |
| License    | Data files © Original Authors                                                    |
| Visibility | Publik                                                                           |
| Tags       | Beginner, Travel, Asia, Recommender Systems                                      |
| Usability  | 8.24                                                                             |

Dataset yang digunakan dalam melakukan sistem rekomendasi adalah `tourism_rating.csv`, `tourism_with_id.csv`, dan `user.csv`. Informasi terkait ketiga dataset yang digunakan akan dibahas dibawah ini.

Ketiga Variable yang digunakan diberikan nama agar mempermudah dengan kode sebagai berikut.
```python
# Menentukan Variable
rating = pd.read_csv('tourism_rating.csv')
info_tourism = pd.read_csv('tourism_with_id.csv')
user = pd.read_csv('user.csv')
```

kemudian dilihat 5 data pertama dari masing-masing variable yang digunakan dengan kode sebagai berikut.

1. `tourism_rating.csv` atau `rating`
```python
# Melihat 5 Data Pertama
rating.head()
```

![image](https://github.com/user-attachments/assets/6c5b9341-c9f7-481d-a934-0b956d1359a3)

dilihat dari hasil yang ada terdapat 3 fitur yaitu `User_Id`, `Place_Id`, dan `Place_Ratings`.

2. `tourism_with_id.csv` atau `info_tourism`

```python
# Melihat 5 Data Pertama
info_tourism.head()
```

![image](https://github.com/user-attachments/assets/8dfc4424-72e6-4d8e-8de7-a3832928b3c8)

dilihat dari hasil yang ada terdapat 13 fitur yaitu `Place_Id`, `Place_Name`, `Description`, `Category`, `City`, `Price`, `Rating`, `Time_Minutes`, `Coordinate`, `Lat`, `Long`, `Unnamed: 11`, dan `Unnamed: 12`.

3. `user.csv` atau `user`
```python
# Melihat 5 Data Pertama
user.head()
```

![image](https://github.com/user-attachments/assets/d4d3d1e1-1893-4663-b36a-11663612b9ff)

dilihat dari hasil yang ada terdapat 3 fitur yaitu `User_Id`, `Location`, dan `Age`.

Dari ketiga dataset yang digunakan didapatkan Informasi Terkait Dataset dengan kode dan hasil sebagai berikut/.
```python
# Informasi Terkait Dataset
print('Jumlah data penilaian tempat: ', len(rating.Place_Id.unique()))
print('Jumlah data kota tempat: ', len(info_tourism.City.unique()))
print('Jumlah data kategori tempat: ', len(info_tourism.Category.unique()))
print('Jumlah data pengguna: ', len(user.User_Id.unique()))
print('Jumlah data user yang memberikan rating penilaian tempat: ', len(rating.User_Id))
```

![image](https://github.com/user-attachments/assets/27a2f0b8-f621-4d27-8fd1-67ece8aee4f0)

dilihat dari hasil yang ada didapatkan bahwa ada **437 penilaian tempat** yang terdapat di **5 kota** dengan **6 kategori**. Kemudian ada **300 pengguna** dengan **total 10000 rating penilaian tempat**.

### _Exploratory Data Analysis_ (EDA)

Kemudian dilakukan _Exploratory Data Analysis_ (EDA) yang merupakan proses untuk menganalisis dari variable yang akan digunakan.

1. `tourism_rating.csv` atau `rating`

Dengan menggunakan kode python sebagai berikut.
```python
# Melihat Informasi Variable
rating.info()
```
didapatkan informasi yanng ditunjukkan pada tabel sebagai berikut.

|#   |Column         |Non-Null Count  |Dtype  |
|--- | ------        | -------------- | ----- | 
| 0  | User_Id       | 10000 non-null | int64 | 
| 1  | Place_Id      | 10000 non-null | int64 | 
| 2  | Place_Ratings | 10000 non-null | int64 | 

Variable `Rating` terdapat 10000 Baris dan 3 Kolom sebagai berikut.

- User_Id = Kolom yang menunjukkan id dari setiap pengguna.
- Place_Id = Kolom yang menunjukkan id dari setiap tempat wisata.
- Place_Ratings = Kolom yang menunjukkan rating yang diberikan oleh pengguna pada tempat wisata tertentu.

Kemudian dilakukan pengecekan Deskripsi dari Variable Rating yang menunjukkan hasil sebagai berikut.

![image](https://github.com/user-attachments/assets/2a5f0115-791c-4882-b9bc-d45222f74d21)

dilihat dari hasil yang ada didapatkan bahwa.

- User_Id memiliki tepat 10000 id yang menunjukkan tidak adanya missing value dari fitur ini, kemudian ada 300 user yang ditunjukkan dari nilai MAX.
- Place_Id memiliki tepat 10000 id yang menunjukkan tidak ada missing value dari fitur ini, kemudian  ada 437 user yang ditunjukkan dari nilai MAX.
- Place_Ratings memiliki rating dimulai dari 1 hingga 5 dengan rata" rating dari 10000 data pada angka 3,066.

Kemudian dilakukan Visualisasi Dari Variable Rating yang ditunjukkan sebagai berikut.

```python
# Distribusi Nilai Rating Tempat Wisata
plt.figure(figsize=(8, 5))
sns.histplot(rating['Place_Ratings'], bins=10)
plt.title('Distribusi Nilai Rating Tempat Wisata')
plt.xlabel('Rating')
plt.ylabel('Jumlah')
plt.show()
```

Dari Kode diatas didapatkan Visualisasi dari Distribusi Nilai Rating Tempat Wisata sebagai berikut.

![image](https://github.com/user-attachments/assets/f70c50eb-7c4e-4cb0-833a-28f86e252f56)

Dari visualisasi yang ada ditunjukkan bahwa **sebaran yang cukup merata** dengan Nilai rating 2 hingga 4 mendominasi distribusi dengan jumlah yang hampir setara. Kemudian **tidak terjadi bias signifikan terhadap rating tinggi atau rendah** dengan gambar histogram yang seimbang dari distribusinya.

2. `tourism_with_id.csv` atau `info_tourism`

Dengan menggunakan kode python sebagai berikut.
```python
# Melihat Informasi Variable
info_tourism.info()
```
didapatkan informasi yanng ditunjukkan pada tabel sebagai berikut.

|#   |Column        |Non-Null Count  |Dtype   |
|--- | ------       | -------------- | -----  | 
| 0  | Place_Id     | 437 non-null   | int64  | 
| 1  | Place_Name   | 437 non-null   | object | 
| 2  | Description  | 437 non-null   | object | 
| 3  | Category     | 437 non-null   | object | 
| 4  | City         | 437 non-null   | object | 
| 5  | Price        | 437 non-null   | int64  | 
| 6  | Rating       | 437 non-null   | float64|
| 7  | Time_Minutes | 205 non-null   | float64|
| 8  | Coordinate   | 437 non-null   | object |
| 9  | Lat          | 437 non-null   | float64|
| 10 | Long         | 437 non-null   | float64|
| 11 | Unnamed: 11  | 0 non-null     | float64|
| 12 | Unnamed: 12  | 437 non-null   | int64  |

Variable `info_tourism` terdapat 437 Baris dan 13 Kolom sebagai berikut.

- Place_Id = Kolom yang menunjukkan id dari setiap tempat wisata.
- Place_Name = Kolom yang menunjukkan nama dari setiap tempat wisata.
- Description = Kolom yang menunjukkan deskripsi dari setiap tempat wisata.
- Category = Kolom yang menunjukkan kategori dari tempat wisata yang ada.
- City = Kolom yang menunjukkan kota dari tempat wisata yang ada.
- Price = Kolom yang menunjukkan harga masuk dari tempat wisata yang ada.
- Rating = Kolom yang menunjukkan rating yang diberikan oleh pengguna pada tempat wisata tertentu.
- Time_Minutes = Kolom yang menunjukkan waktu yang biasa dihabiskan untuk menelusuri tempat wisata.
- Coordinate = Kolom yang menunjukkan koordinat dari setiap tempat wisata.
- Lat = Kolom yang menunjukkan Latitude dari setiap tempat wisata.
- Long = Kolom yang menunjukkan Longitude dari setiap tempat wisata.
- Unnamed: 11 = Kolom kosong yang tidak digunakan pada proyek ini.
- Unnamed: 12 = Kolom yang berisi angka 1 - 437 yang tidak digunakan pada proyek ini.

kemudian dilakukan pengecekan macam kategori tempat dan macam kota yang terdapat pada dataset dengan kode dan hasil sebagai berikut.

```python
# Melihat macam kategori tempat
print('Macam Kategori Tempat: ', info_tourism.Category.unique())
```

![image](https://github.com/user-attachments/assets/28a60ab9-c8b5-4b59-8e0b-b4aa6fdd7a0e)

Dataset ini memiliki 6 Kategori Tempat yaitu : Budaya, Taman Hiburan, Cagar Alam, Bahari, Pusat Perbelanjaan, dan Tempat Ibadah.

```python
# Melihat kota pada dataset
print('Kota: ', info_tourism.City.unique())
```

![image](https://github.com/user-attachments/assets/6bbec43f-67a0-47c0-9597-1606a4e9f5c4)

Dataset ini memili 5 Macam Kota yaitu : Jakarta, Yogyakarta, Bandung, Semarang, dan Surabaya.

Kemudian dilakukan Visualisasi Dari Variable info_tourism yang ditunjukkan sebagai berikut.

- Visualisasi Jumlah Tempat Wisata per Kategori

```python
# Jumlah Tempat Wisata per Kategori
plt.figure(figsize=(10, 6))
sns.countplot(data=info_tourism, y='Category', order=info_tourism['Category'].value_counts().index)
plt.title('Jumlah Tempat Wisata per Kategori')
plt.tight_layout()
plt.show()
```
![image](https://github.com/user-attachments/assets/05cb9883-155a-4833-94fd-18c72d1a97d9)

Dari visualisasi diatas didapatkan bahwa Kategori Taman Hiburan memiliki jumlah paling banyak dari dataset ini kemudian disusul oleh Kategori Budaya dan cagar alam. Pusat perbelanjaan menjadi tempat wisata paling sedikit.

- Visualisasi Jumlah Tempat Wisata per Kota

```python
# Jumlah Tempat Wisata per Kota
plt.figure(figsize=(10, 6))
sns.countplot(data=info_tourism, y='City', order=info_tourism['City'].value_counts().index)
plt.title('Jumlah Tempat Wisata per Kota')
plt.tight_layout()
plt.show()
```
![image](https://github.com/user-attachments/assets/7a2366ae-c89f-45d7-b4bb-d93767641a5a)

Dari visualisasi diatas didapatkan bahwa Kota Yogyakarta dan Bandung memiliki jumlah paling banyak dan tidak terlalu jauh jumlahnya kemudian disusul oleh jakarta dan semarang. kota surabaya menjadi yang paling sedikit untuk tempat wisatanya.

3. `user.csv` atau `user`

Dengan menggunakan kode python sebagai berikut.
```python
# Melihat Informasi Variable
user.info()
```
didapatkan informasi yanng ditunjukkan pada tabel sebagai berikut.

|#   |Column         |Non-Null Count  |Dtype   |
|--- | ------        | -------------- | -----  | 
| 0  | User_Id       | 300 non-null   | int64  | 
| 1  | Locatiom      | 300 non-null   | object | 
| 2  | Age           | 300 non-null   | int64  | 

Variable `user` terdapat 300 Baris dan 3 Kolom sebagai berikut.

- User_Id = Kolom yang menunjukkan id dari setiap pengguna.
- Location = Kolom yang menunjukkan lokasi dari setiap tempat wisata.
- Age = Kolom yang menunjukkan umur dari setiap pengguna.

Variable user memiliki shape (300,3) yang didapatkan dari kode sebagai berikut.

```python
# Informasi Shape Variable
print(user.shape)
```

Kemudian dilakukan Visualisasi Dari Variable user yang ditunjukkan sebagai berikut.

- Visualisasi Distribusi Umur User

```python
# Distribusi Umur User
plt.figure(figsize=(8, 5))
sns.histplot(user['Age'], bins=20)
plt.title('Distribusi Umur User')
plt.xlabel('Umur')
plt.ylabel('Jumlah')
plt.show()
```
![image](https://github.com/user-attachments/assets/16e5c695-0d20-4fa1-8114-956197a34bc8)

Dari visualisasi diatas didapatkan bahwa umur paling banyak berada pada umur 30 kemudian umur yang lain hampir sama rata.

- Visualisasi 10 Lokasi Terbanyak

 ```python
# Visualisasi 10 Lokasi Terbanyak
top_locations = user['Location'].value_counts().nlargest(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_locations.values, y=top_locations.index)
plt.title('10 Lokasi Terbanyak')
plt.xlabel('Jumlah User')
plt.ylabel('Lokasi')
plt.tight_layout()
plt.show()
```

![image](https://github.com/user-attachments/assets/ff6ff8cd-6408-41a9-8d5c-f25b4a776c12)

Dari visualisasi diatas didapatkan bahwa lokasi terbanyak dari user adalah Bekasi, Jawa barat. Kemudian hampir sama rata untuk variable lainnya dan yang paling sedikit adalah Ponorogo, Jawa Timur.

- Visualisasi Boxplot Umur User

```python
# Boxplot Umur User
plt.figure(figsize=(6, 2))
sns.boxplot(x=user['Age'])
plt.title('Boxplot Umur User')
plt.show()
```
![image](https://github.com/user-attachments/assets/2e912adc-79c1-45f0-ac4f-34751ac7eea8)

Dari visualisasi diatas didapatkan bahwa umur rata-rata berada di kurang dari 30 tahun, dengan batas bawah sekitar 24 tahun dan batas atas sekitar 34 tahun.

## Data Preparation

Data Preparation dilakukan karena kita menggunakan beberapa dataset yang akan digunakan secara langsung dalam satu waktu, oleh karena itu kita perlu untuk menggabungkan dan memilih fitur-fitur yang akan digunakan sebelum masuk ke permodelan.

Tahapan yang akan dilakukan adalah.

- Data Preprocessing

Data Preprocessing ini dilakukan dengan awalnya menggabungkan variable `info_tourism` dan `rating` yang akhirnya menjadi `tourism_all` dengan kode sebagai berikut.

```python
# Menggabungkan Info_Tourism dengan Rating
tourism_all = np.concatenate((
    info_tourism.Place_Id.unique(),
    rating.Place_Id.unique()
))

tourism_all = np.sort(np.unique(tourism_all))

print('Total jumlah turis: ', len(tourism_all))
```
kemudian mendapatkan hasil sebagai berikut.

![image](https://github.com/user-attachments/assets/f2fca362-b148-4d47-ba45-dc7dad774e14)

Kemudian merubah nama `rating` menjadi `all_tourism_rate` dengan kode sebagai berikut.

```python
# Mengubah Nama Rating menjadi all_tourism_rate
all_tourism_rate = rating
all_tourism_rate
```
kemudian menggabungkan beberapa informasi (fitur) dari `info_tourism` dengan `all_tourism_rate` menjadi `all_tourism` dengan code dan hasil sebagai berikut.

```python
# Menggabungkan all_tourism_rate dengan beberapa informasi dari tourism
all_tourism = pd.merge(all_tourism_rate, info_tourism[['Place_Id','Place_Name','Description','City','Category']],on='Place_Id', how='left')
all_tourism
```

![image](https://github.com/user-attachments/assets/294164c4-7ba2-4dd6-9781-6b8dc088998b)

kemudian menambahkan fitur city_category yang merupakan gabungan dari kota dan kategori ke `all_tourism` dengan kode dan hasil sebagai berikut sebagai berikut.

```python
# Menambahkan fitur city_category ke all_tourism
all_tourism['city_category'] = all_tourism[['City','Category']].agg(' '.join,axis=1)
```

![image](https://github.com/user-attachments/assets/bc10501a-45db-4f3c-a21f-0d3bde18ba3d)

untuk sementara `all_tourism` yang akan dilakukan pengecekan missing value dan duplikasi.

- Cek Missing Value
  
- Cek Duplikasi Data
- TF-IDF Vectorizer

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

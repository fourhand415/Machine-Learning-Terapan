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
    - Evaluasi performa menggunakan Root Mean Squared Error, Precission, Recall, dan F1-Score yang akan memberikan wawasan tentang efektiviras model dalam merekomendasikan tempat wisata kepada pengguna.

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

Dilakukan pengecekan missing value untuk mengetahui apakah ada fitur yang hilang atau tidak pada dataset yang digunakan, jika ada yang hilang akan dilakukan imputasi dengan mean atau median atau jika sedikit variable yang hilang akan dilakukan penghapusan pada baris yang memiliki missing value. Pengecekan missing value akan dilakukan dengan kode sebagai berikut.

```python
# Mengecek Missing Value
all_tourism.isnull().sum()
```
![image](https://github.com/user-attachments/assets/0c2cbb79-ad4b-4f9e-a657-b89f0ce3480d)

dari hasil tersebut terlihat tidak ada missing value pada `all_tourism` kemudian akan dilanjutkan dengan pengecekan duplikasi.

- Cek Duplikasi Data
  
Dilakukan pengecekan duplikasi bertujuan agar dataset yang digunakan tidak ada data duplikat. pengecekan duplikat dilakukan dengan kode sebagai berikut.

```python
# Mengecek Duplicated Data
all_tourism.duplicated().sum()
```
![image](https://github.com/user-attachments/assets/2c515f50-9ec9-45ad-8667-6232c86ee9d8)

terlihat masi ada duplikasi dan akan dilakukan penghapusan data duplikat berdasarkan kolom place_id dengan kode sebagai berikut.
```python
# Menghapus data duplikat berdasarkan kolom Place_Id
Preparasi = all_tourism.drop_duplicates('Place_Id')
Preparasi
```
dari kode tersebut telah menghapus semua duplikat berdasarkan place_id dari 10000 baris menjadi 437 baris.

Kemudian untuk Content Based Filtering diambil beberapa fitur lagi dengan menerapkan hasil duplikasi dengan code sebagai berikut.

```python
# Menerapkan Preparasi
Id = Preparasi.Place_Id.tolist()

Nama = Preparasi.Place_Name.tolist()

Kategori = Preparasi.Category.tolist()

Deskripsi = Preparasi.Description.tolist()

Kota = Preparasi.City.tolist()

Kota_Kategori = Preparasi.city_category.tolist()
```

```python
# Dataset yang digunakan
tourism_fix = pd.DataFrame({
    'Id': Id,
    'Nama': Nama,
    'Kategori': Kategori,
    'Deskripsi': Deskripsi,
    'Kota': Kota,
    'Kota_Kategori': Kota_Kategori
})

tourism_fix
```

![image](https://github.com/user-attachments/assets/f60c3608-e4b8-43fc-b2ad-9b5f62acd6ab)

dari kode diatas didapatkan `tourism_fix` sebagai dataset untuk content based filtering.

- TF-IDF Vectorizer
Melakukan perubahan data kedalam representasi numerik sebelum tahap modeling dengan cosine numerik. perubahan ini diperlukan karena content based filtering mengukur dari kemiripan antar item yang dimana bisa dilakukan dengan TF-IDF Vectorizer.

Tahapan TF-IDF Vectorizer adalah sebagai berikut.

```python
# Merubah nama Variable
data = tourism_fix
data.sample(5)
```
```python
# Inisialisasi TfidfVectorizer
tf = TfidfVectorizer()
# Melakukan perhitungan idf pada data kota_kategori
tf.fit(data['Kota_Kategori'])
# Mengeluarkan Nama Fitur
print('Nama Fitur: ', list(tf.vocabulary_.keys()))
```
![image](https://github.com/user-attachments/assets/47ece293-88a3-4940-8d0f-eccf4aa004ed)

```python
# Melakukan fit lalu ditransformasikan ke bentuk matrix
tfidf_matrix = tf.transform(data['Kota_Kategori'])
# Melihat ukuran matrix
tfidf_matrix.shape
```
![image](https://github.com/user-attachments/assets/c4db7dc6-21d3-486e-a63c-73bc144664f9)

```python
# Mengubah vektor tf-idf dalam bentuk matriks dengan fungsi todense()
tfidf_matrix.todense()
```
![image](https://github.com/user-attachments/assets/b4c93a7e-c80e-465f-aa8d-c8ff7e7d13c5)

```python
# Membuat DataFrame
pd.DataFrame(
    tfidf_matrix.todense(),
    columns=list(tf.vocabulary_.keys()),
    index = data.Nama
).sample(5)
```
![image](https://github.com/user-attachments/assets/c702a1db-260b-416d-acc3-1030b5057086)

Tahapan TF-IDF Vectorizer dimulai dari melakukan perhitungan kemiripan yang akan diberikan pada kota_kategori, kemudian akan memecah dari setiap fitur untuk diberikan kemiripannya, kemudian hasilnya akan dibentuk matrik dan akhirnya dibentuk dalam dataframe. Hasil TF-IDF Vectorizer ini memberikan nilai yang merupakan nilai kemiripan dari setiap fitur untuk tempat wisata.

- Encode

Proses encode ini berguna pada model collaborative filtering dimana akan merubah isi setiap fitur dengan nilai yang bisa dibaca oleh komputer. proses encode ini dilakukan dengan code sebagai berikut.

```python
# Membaca dataset
df = rating
df
```
pada collaborative filtering menggunakan data dari rating
```python
# Mengubah User_ID menjadi list tanpa nilai yang sama
user_ids = df.User_Id.unique().tolist()
print('list User_ID: ', user_ids)

# Melakukan encoding User_ID
user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}
print('encoded User_ID: ', user_to_user_encoded)

# Melakukan proses encoding angka ke ke User_ID
user_encoded_to_user = {i: x for i, x in enumerate(user_ids)}
print('encoded angka ke User_ID: ', user_encoded_to_user)
```
![image](https://github.com/user-attachments/assets/e7cb82c4-db44-4b83-a7b9-65514c6b2b42)

```python
# Mengubah Place_ID menjadi list tanpa nilai yang sama
place_ids = df.Place_Id.unique().tolist()
print('list Place_ID: ', place_ids)

# Melakukan encoding Place_ID
place_to_place_encoded = {x: i for i, x in enumerate(place_ids)}
print('encoded Place_ID: ', place_to_place_encoded)

# Melakukan proses encoding angka ke ke Place_ID
place_encoded_to_place = {x: i for x, i in enumerate(place_ids)}
print('encoded angka ke Place_ID: ', place_to_place_encoded)
```
![image](https://github.com/user-attachments/assets/f373b1fb-82db-4b2e-aa84-4efbb62ffad1)

```python
# Mapping User_ID ke dataframe user
df['user'] = df.User_Id.map(user_to_user_encoded)

# Mapping Place_ID ke dataframe place
df['place'] = df.Place_Id.map(place_to_place_encoded)
```
```python
# Mendapatkan jumlah user
num_users = len(user_to_user_encoded)
print(num_users)

# Mendapatkan jumlah place
num_place = len(place_encoded_to_place)
print(num_place)

# Mengubah rating menjadi nilai float
df['Place_Ratings'] = df['Place_Ratings'].values.astype(np.float32)

# Nilai minimum rating
min_rating = min(df['Place_Ratings'])

# Nilai maksimal rating
max_rating= max(df['Place_Ratings'])

print('Number of User: {}, Number of Place: {}, Min Rating: {}, Max Rating: {}'.format(
    num_users, num_place, min_rating, max_rating
))
```
![image](https://github.com/user-attachments/assets/4aa2c1ee-7ff6-408f-96ed-102c105a8e3c)

Dari proses diatas data dari rating telah di encode dan siap digunakan dalam collaborarive filtering.

- Membagi Data untuk Training Testing dan Validasi

Karena pada collaborative filtering menggunakan algoritma deep learning maka akan dilakukan pembagian dataset menjadi training testing dan validasi. Menggunakan kode sebagai berikut.

```python
# Mengacak dataset
df = df.sample(frac=1, random_state=42)
df
```
```python
# Membuat variabel x untuk mencocokkan data user dan place menjadi satu value
x = df[['user', 'place']].values

# Membuat variabel y untuk membuat rating dari hasil
y = df['Place_Ratings'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values

# Membagi menjadi 80% data train dan 20% data validasi
train_indices = int(0.8 * df.shape[0])
x_train, x_val, y_train, y_val = (
    x[:train_indices],
    x[train_indices:],
    y[:train_indices],
    y[train_indices:]
)

print(x, y)
```
![image](https://github.com/user-attachments/assets/249e319f-3a9d-4f76-8f50-3270fc2e6eb9)

dengan itu telah dibagi dan bisa dilakukan proses training pada tahap modeling.


## Modeling
Tahap modeling yang digunakan adalah.
- Cosine Similarity
- RecommenderNet

1. Cosine Similarity - Content Based Filtering
Tahapan Cosine Similarity adalah dengan menghitung kemiripan berdasarkan matriks tfidf. dengan code sebagai  berikut akan dijelaskan terkait cosine similarity.

- Membuat model dan melakukan rekomendasi
  
```python
# Menghitung cosine similarity pada matrix tf-idf
cosine_sim = cosine_similarity(tfidf_matrix)
cosine_sim
```

![image](https://github.com/user-attachments/assets/b12d9f64-4a48-4565-9459-1b934db43807)

hasil dari perhitungan kemiripan akan dimasukkan kedalam dataframe yang ada.

```python
# Membuat dataframe dari variabel cosine_sim dengan baris dan kolom berupa nama
cosine_sim_df = pd.DataFrame(cosine_sim, index=data['Nama'], columns=data['Nama'])
print('Shape:', cosine_sim_df.shape)

# Melihat similarity matrix
cosine_sim_df.sample(5, axis=1).sample(10, axis=0)
```
![image](https://github.com/user-attachments/assets/bb2ccbd4-86e1-4d27-b317-e7b92958bbc0)

Dari hasil ini akan di cari Top N Rekomendasi dari tempat yang akan dipilih (misal dalam kasus ini Taman Hiburan Rakyat).
```python
# Mendapatkan Rekomendasi
def Rekomendasi(place_name,similarity_data=cosine_sim_df,items=data[['Nama','Kategori','Deskripsi','Kota']],k=5):
    index = similarity_data.loc[:,place_name].to_numpy().argpartition(range(-1,-k,-1))

    closest = similarity_data.columns[index[-1:-(k+2):-1]]

    closest = closest.drop(place_name,errors='ignore')

    return pd.DataFrame(closest).merge(items).head(k)
```
```python
# Melakukan Rekomendasi
Rekomendasi("Taman Hiburan Rakyat")
```
![image](https://github.com/user-attachments/assets/c1735986-1963-4590-9667-dfad190b4450)

hasil rekomendasi dari taman hiburan rakyat berasal dari kesamaan kategori dan kota pada taman hiburan rakyat itu sendiri. nilai nya berasal dari cosine similarity.

- Cara kerja Content Based Filtering serta kekurangan dan kelebihan
    - Cara Kerja Content-based filtering adalah pendekatan sistem rekomendasi yang berfokus pada karakteristik item dan preferensi individu pengguna. Sistem ini membangun profil pengguna berdasarkan atribut dari item yang telah disukai sebelumnya sehingga memberikan rekomendasi yang lebih personal. Proses ini tidak memerlukan informasi atau perilaku dari pengguna lain, melainkan murni bergantung pada hubungan antara item dan minat pengguna.
    - Kelebihan Content Based Filtering adalah model ini tidak bergantung pada data pengguna lain sehingga tingkat personalisasi sangat tinggi.
    - Kekurangan Content Based Filtering adalah kecenderungan untuk memberikan rekomendasi item yang sangat mirip pada hal yang disukai dari pengguna, hal ini membuat tidak akan ada rekomendasi yang baru diluar dari prefrensi pengguna.

2. RecommenderNet - Collaborative Filtering

Recommender Net adalah jenis sistem rekomendasi berbasis neural network (jaringan saraf tiruan) yang mempelajari representasi pengguna dan item dalam bentuk vektor numerik (embedding). Recommender Net umumnya digunakan untuk memprediksi preferensi pengguna terhadap item tertentu, misalnya prediksi rating film, rekomendasi produk, atau saran konten yang relevan. Sistem ini merupakan pendekatan pembelajaran mendalam (deep learning) yang mampu menangkap hubungan kompleks antara pengguna dan item.

- Membuat model dan melakukan rekomendasi
  
```python
# Proses Training
class RecommenderNet(tf.keras.Model):

  # Insialisasi fungsi
  def __init__(self, num_users, num_place, embedding_size, **kwargs):
    super(RecommenderNet, self).__init__(**kwargs)
    self.num_users = num_users
    self.num_place = num_place
    self.embedding_size = embedding_size
    self.user_embedding = layers.Embedding(
        num_users,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = keras.regularizers.l2(1e-6)
    )
    self.user_bias = layers.Embedding(num_users, 1)
    self.place_embedding = layers.Embedding(
        num_place,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = keras.regularizers.l2(1e-6)
    )
    self.place_bias = layers.Embedding(num_place, 1)

  def call(self, inputs):
    user_vector = self.user_embedding(inputs[:,0]) # memanggil layer embedding 1
    user_bias = self.user_bias(inputs[:, 0]) # memanggil layer embedding 2
    place_vector = self.place_embedding(inputs[:, 1]) # memanggil layer embedding 3
    place_bias = self.place_bias(inputs[:, 1]) # memanggil layer embedding 4

    dot_user_place = tf.tensordot(user_vector, place_vector, 2)

    x = dot_user_place + user_bias + place_bias

    return tf.nn.sigmoid(x) # activation sigmoid
```
```python
# Compile Model
model = RecommenderNet(num_users, num_place, 50)

# model compile
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = keras.optimizers.Adam(learning_rate=0.001),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)
```
```python
# Run Model
history = model.fit(
    x = x_train,
    y = y_train,
    batch_size = 8,
    epochs = 100,
    validation_data = (x_val, y_val),
)
```
```python
# Mendapatkan Rekomendasi
place_df = tourism_fix
df = pd.read_csv(f'tourism_rating.csv')

user_id = df.User_Id.sample(1).iloc[0]
place_visited_by_user = df[df.User_Id == user_id]

place_not_visited = place_df[~place_df['Id'].isin(place_visited_by_user['Place_Id'].values)]['Id']
place_not_visited = list(
    set(place_not_visited)
    .intersection(set(place_to_place_encoded.keys()))
)

place_not_visited = [[place_to_place_encoded.get(x)] for x in place_not_visited]
user_encoder = user_to_user_encoded.get(user_id)
user_place_array = np.hstack(
    ([[user_encoder]] * len(place_not_visited), place_not_visited)
)
```
```python
# Melakukan Rekomendasi
ratings = model.predict(user_place_array).flatten()

top_ratings_indices = ratings.argsort()[-10:][::-1]
recommended_place_ids = [
    place_encoded_to_place.get(place_not_visited[x][0]) for x in top_ratings_indices
]

print('Showing recommendations for users: {}'.format(user_id))
print('===' * 9)
print('Place with high ratings from user')
print('----' * 8)

top_place_user = (
    place_visited_by_user.sort_values(
        by = 'Place_Ratings',
        ascending=False
    )
    .head(5)
    .Place_Id.values
)

place_df_rows = place_df[place_df['Id'].isin(top_place_user)]
for row in place_df_rows.itertuples():
    print(row.Nama, ':', row.Kategori, ',', row.Kota)

print('----' * 8)
print('Top 10 place recommendation')
print('----' * 8)

recommended_place = place_df[place_df['Id'].isin(recommended_place_ids)]
for row in recommended_place.itertuples():
    print(row.Nama, ':', row.Kategori, ',', row.Kota)
```
![image](https://github.com/user-attachments/assets/97ae93c8-5d87-4bd2-81b5-ae11e9c3f3d7)

pada RecomenderNet akan merepresentasikan Pengguna dan Item pada Embedding Layer kemudian menggabungkannya yang akhirnya akan menghasilkan prediksi.

- Cara kerja Collaborative Filtering serta kekurangan dan kelebihan
    - Collaborative Filtering adalah metode rekomendasi yang menganalisis interaksi antara pengguna dan item—seperti penilaian, klik, atau pembelian—untuk memprediksi item apa yang kemungkinan akan disukai oleh seorang pengguna. Algoritma ini tidak melihat isi dari item, melainkan mencari kesamaan pola perilaku antar pengguna atau antar item. Misalnya, jika dua pengguna memiliki riwayat penilaian yang mirip, maka item yang disukai oleh salah satu pengguna dapat direkomendasikan kepada pengguna lainnya. Terdapat dua pendekatan utama dalam collaborative filtering. User-based filtering mencari pengguna lain yang memiliki pola preferensi serupa, kemudian merekomendasikan item yang mereka sukai. Sementara itu, item-based filtering melihat item yang serupa berdasarkan pengguna yang sama-sama menyukai atau menilai tinggi item tersebut. Semakin banyak data interaksi yang tersedia, semakin akurat model dalam menemukan pola-pola yang berguna untuk rekomendasi.
    - Kelebihan Collaborative Filtering adalah kemampuannya dalam memberikan rekomendasi yang sangat personal. Karena berdasarkan perilaku nyata pengguna, sistem ini dapat menyesuaikan rekomendasi dengan preferensi spesifik pengguna tanpa perlu tahu isi atau atribut item tersebut. collaborative filtering juga memungkinkan penemuan item baru yang mungkin belum pernah dijelajahi oleh pengguna. Hal ini karena sistem bisa merekomendasikan item yang disukai oleh pengguna lain yang mirip, meskipun item tersebut belum pernah dilihat atau diketahui oleh pengguna target.
    - Kekurangan Collaborative Filtering  juga memiliki beberapa kelemahan penting. Salah satunya adalah masalah cold start, yaitu kesulitan memberikan rekomendasi yang akurat ketika pengguna atau item masih baru dan belum memiliki cukup interaksi. Selain itu, metode ini rentan terhadap data sparsity, di mana interaksi pengguna-item sangat sedikit dibandingkan total kombinasi yang mungkin, sehingga menyulitkan pencarian pola yang akurat.
  
## Evaluation

Pada tahap ini, metrik evaluasi yang digunakan untuk mengukur performa model meliputi **RMSE**, **Precision**, **Recall**, dan **F1-Score**. Metrik-metrik ini dipilih karena relevansi mereka dalam konteks sistem rekomendasi, baik dari sisi akurasi prediksi maupun kualitas hasil rekomendasi.

### 1. RMSE (Root Mean Squared Error)
RMSE digunakan untuk mengukur seberapa jauh prediksi sistem dari nilai sebenarnya, terutama dalam sistem rekomendasi berbasis rating (nilai numerik). Semakin rendah nilai RMSE, semakin baik model dalam memprediksi rating.

$$
\text{RMSE} = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 }
$$

- `y_i`: rating sebenarnya  
- `\hat{y}_i`: rating yang diprediksi oleh model  
- Cocok digunakan untuk model berbasis regresi.

### 2. Precision
Precision mengukur proporsi item yang direkomendasikan dan benar-benar relevan dari seluruh item yang direkomendasikan. Cocok digunakan ketika kita ingin meminimalkan kesalahan dalam rekomendasi yang diberikan.

$$
\text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}
$$

- Semakin tinggi nilai precision, semakin akurat rekomendasi yang diberikan.

### 3. Recall
Recall mengukur seberapa banyak item relevan yang berhasil direkomendasikan oleh sistem dari seluruh item relevan yang tersedia.

$$
\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}
$$

- Recall tinggi menunjukkan bahwa sistem berhasil mencakup sebagian besar item relevan.

### 4. F1-Score
F1-Score adalah harmonic mean dari precision dan recall. Metrik ini berguna ketika kita ingin menjaga keseimbangan antara keduanya.

$$
\text{F1-Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

- Digunakan untuk mengevaluasi kinerja keseluruhan dalam klasifikasi relevan/tidak relevan dari hasil rekomendasi.

Pada Collaborative Filtering digunakan RMSE dengan code sebagai berikut.

```python
# Visualisasi Metriks
plt.plot(history.history['root_mean_squared_error'])
plt.plot(history.history['val_root_mean_squared_error'])
plt.title('model_metrics')
plt.ylabel('root_mean_squared_error')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
```
dengan hasil sebagai berikut.

![image](https://github.com/user-attachments/assets/fa6aa12d-7ed6-433f-b5fa-2745d8cac580)

Penjelasan dari hasil matriks evaluasi pada Collaborative Filtering :

1. Pola RMSE Data Training
RMSE pada data train turun tajam dari awal hingga sekitar epoch ke-20.

Setelah itu, nilai RMSE cenderung stabil di sekitar 0.31.

Ini menunjukkan bahwa model berhasil mempelajari pola dari data training dengan baik dan mengalami konvergensi.

2. Pola RMSE Data Testing
RMSE pada data test justru naik perlahan seiring bertambahnya epoch, mulai dari sekitar 0.34 hingga lebih dari 0.36.

Pada Content Based Filtering digunakan Precision, Recall, dan F1-Score dengan code sebagai berikut.

```python
# Fungsi Evaluasi
def evaluate_recommendation(recommendations, test_item_name, df, k=5):
    test_kategori = df[df['Nama'] == test_item_name]['Kategori'].values[0]

    recommended_kategori = recommendations['Kategori'].tolist()
    
    relevant_recommended = sum([1 for c in recommended_kategori if c == test_kategori])
    
    total_relevant = len(df[(df['Kategori'] == test_kategori) & (df['Nama'] != test_item_name)])
    
    precision = relevant_recommended / k
    recall = relevant_recommended / total_relevant if total_relevant > 0 else 0
    
    f1 = 0
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1
```
```python
# Evaluasi hasil rekomendasi
test_item_name = "Taman Hiburan Rakyat"
recs = Rekomendasi(test_item_name, k=5)

precision, recall, f1 = evaluate_recommendation(recs, test_item_name, data, k=5)

# Cetak hasil evaluasinya
print(f"Evaluasi untuk '{test_item_name}'")
print(f"Precision@5 : {precision:.4f}")
print(f"Recall@5    : {recall:.4f}")
print(f"F1-Score    : {f1:.4f}")
```
dengan hasil sebagai berikut.

![image](https://github.com/user-attachments/assets/6844c845-25f0-45ef-99e8-9bd983b8dff8)

Penjelasan dari hasil matriks evaluasi pada Content Based Filtering :
1. Precision@5 = 1.0000
Artinya, dari 5 item yang direkomendasikan, semuanya relevan (misalnya, benar-benar disukai atau sesuai dengan preferensi pengguna). Ini menunjukkan bahwa sistem sangat akurat dalam memilih item yang ditampilkan di daftar rekomendasi teratas. Namun, precision tinggi belum tentu berarti sistem bekerja optimal secara keseluruhan.

2. Recall@5 = 0.0373
Recall sangat rendah (hanya 3.73%), yang berarti dari seluruh item relevan yang mungkin ada untuk pengguna, hanya sebagian kecil (3.73%) yang berhasil ditampilkan oleh sistem di 5 teratas. Hal ini bisa terjadi jika: Total item relevan sangat banyak, tapi hanya 5 yang diambil untuk evaluasi. Sistem hanya "menebak dengan sangat hati-hati" dan tidak mencakup variasi preferensi pengguna.

3. F1-Score = 0.0719
F1-Score yang rendah mencerminkan bahwa meskipun precision tinggi, recall yang sangat rendah menurunkan performa keseluruhan dari segi keseimbangan antara akurasi dan cakupan rekomendasi.

## Kesimpulan

Berdasarkan hasil evaluasi, pendekatan sistem rekomendasi yang menggabungkan Content-Based Filtering dengan model Cosine Similarity serta Collaborative Filtering menggunakan RecommenderNet terbukti efektif dalam mencapai tujuan proyek, yaitu memberikan rekomendasi tempat wisata yang relevan. Metode Content-Based Filtering mampu menyarankan destinasi dengan karakteristik serupa dari tempat-tempat yang sebelumnya disukai pengguna, seperti berdasarkan kategori atau jenis wisata, sehingga menghasilkan rekomendasi Top-5 yang akurat dan sesuai. Sementara itu, Collaborative Filtering melalui RecommenderNet berhasil menganalisis pola interaksi dan rating pengguna terhadap tempat wisata, sehingga dapat memprediksi preferensi pengguna dengan tingkat akurasi yang baik. Evaluasi menunjukkan bahwa model memiliki nilai RMSE sekitar 0.31 pada data training dan 0.36 pada data testing, yang menandakan sedikit overfitting namun secara umum model tetap memberikan performa prediktif yang cukup baik. Secara keseluruhan, kedua metode ini saling melengkapi dan berhasil menyajikan rekomendasi Top-5 tempat wisata yang sesuai dengan preferensi pengguna berdasarkan tempat dan kategorinya.

## Refrensi
**[1]** J. Lu, D. Wu, M. Mao, W. Wang, and G. Zhang, "Recommender system application developments: A survey," Decision Support Systems, vol. 74, pp. 12–32, 2015. [Online]. Available: https://doi.org/10.1016/j.dss.2015.03.008 [Accessed: May 14, 2025]. [Accessed: May 28, 2025].

**[2]** F. Ricci, L. Rokach, and B. Shapira, "Introduction to Recommender Systems Handbook," in Recommender Systems Handbook, 1st ed., Springer, 2011, pp. 1–35. [Online]. Available: http://dx.doi.org/10.1007/978-0-387-85820-3_1. [Accessed: May 28, 2025].

**---Ini adalah bagian akhir laporan---**

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

| Jenis      | Keterangan                                                                   |
|------------|------------------------------------------------------------------------------|
| Title      | Breast Cancer Wisconsin (Diagnostic) Data Set                                |
| Source     | [Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) |
| Owner      | [UCI Machine Learning](https://www.kaggle.com/organizations/uciml)           |
| License    | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)        |
| Visibility | Publik                                                                       |
| Tags       | Cancer, Healthcare                                                           |
| Usability  | 8.53                                                                         |

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

### Exploratory Data Analysis - Deskripsi Variabel
| #  | Column                  |Non-Null Count| Dtype    |
|----|-------------------------|--------------|----------|
| 0  | Diagnosis               | 569 non-null | object   |
| 1  | Radius_mean             | 569 non-null | float64  |
| 2  | Texture_mean            | 569 non-null | float64  |
| 3  | Perimeter_mean          | 569 non-null | float64  |
| 4  | Area_mean               | 569 non-null | float64  |
| 5  | Smoothness_mean         | 569 non-null | float64  |
| 6  | Compactness_mean        | 569 non-null | float64  |
| 7  | Concavity_mean          | 569 non-null | float64  |
| 8  | Concave points_mean     | 569 non-null | float64  |
| 9  | Symmetry_mean           | 569 non-null | float64  |
| 10 | Fractal_dimension_mean  | 569 non-null | float64  |
| 11 | Radius_se               | 569 non-null | float64  |
| 12 | Texture_se              | 569 non-null | float64  |
| 13 | Perimeter_se            | 569 non-null | float64  |
| 14 | Area_se                 | 569 non-null | float64  |
| 15 | Smoothness_se           | 569 non-null | float64  |
| 16 | Compactness_se          | 569 non-null | float64  |
| 17 | Concavity_se            | 569 non-null | float64  |
| 18 | Concave points_se       | 569 non-null | float64  |
| 19 | Symmetry_se             | 569 non-null | float64  |
| 20 | Fractal_dimension_se    | 569 non-null | float64  |
| 21 | Radius_worst            | 569 non-null | float64  |
| 22 | Texture_worst           | 569 non-null | float64  |
| 23 | Perimeter_worst         | 569 non-null | float64  |
| 24 | Area_worst              | 569 non-null | float64  |
| 25 | Smoothness_worst        | 569 non-null | float64  |
| 26 | Compactness_worst       | 569 non-null | float64  |
| 27 | Concavity_worst         | 569 non-null | float64  |
| 28 | Concave points_worst    | 569 non-null | float64  |
| 29 | Symmetry_worst          | 569 non-null | float64  |
| 30 | Fractal_dimension_worst | 569 non-null | float64  |

Dari tabel tersebut didapatkan informasi bahwa :
- Terdapat 1 fitur dengan kategori object yaitu Diagnosis, fitur ini juga menjadi target dalam proyek penelitian ini.
- Terdapat 30 fitur dengan tipe numeric dengan 30 fiturnya bertipe float64.

|         |radius_mean|texture_mean|perimeter_mean|area_mean  |smoothness_mean|compactness_mean|concavity_mean|concave points_mean|symmetry_mean|fractal_dimension_mean |radius_se |texture_se|perimeter_se|area_se    |smoothness_se|compactness_se|concavity_se|concave points_se|symmetry_se|fractal_dimension_se|radius_worst|texture_worst|perimeter_worst|area_worst  |smoothness_worst|compactness_worst|concavity_worst|concave points_worst|symmetry_worst|fractal_dimension_worst|
|:-------:|:---------:|:----------:|:------------:|:---------:|:-------------:|:--------------:|:------------:|:-----------------:|:-----------:|:---------------------:|:--------:|:--------:|:----------:|:---------:|:-----------:|:------------:|:----------:|:---------------:|:---------:|:------------------:|:----------:|:-----------:|:-------------:|:----------:|:--------------:|:---------------:|:-------------:|:------------------:|:------------:|:---------------------:|
| Count   |569.000000 |569.000000	 |569.000000	  |569.000000 |569.000000     |569.000000	     |569.000000    |569.000000         |569.000000   |569.000000         	  |569.000000|569.000000|569.000000  |569.000000 |569.000000   |569.000000    |569.000000  |569.000000    	 |569.000000 |569.000000          |569.000000  |569.000000	 |569.000000	   |569.000000  |569.000000      |569.000000	     |569.000000	   |569.000000          |569.000000    |569.000000	           |
| Mean    |14.127292  |19.289649   |91.969033	    |654.889104 |0.096360	      |0.104341	       |0.088799	    |0.048919	          |0.181162	    |0.062798	              |0.405172	 |1.216853	|2.866059	   |40.337079	 |0.007041	   |0.025478	    |0.031894	   |0.011796	       |0.020542	 |0.003795	          |16.269190	 |25.677223	   |107.261213	   |880.583128  |0.132369	       |0.254265	       |0.272188	     |0.114606	          |0.290076	     |0.083946               |
| Std     |3.524049   |4.301036    |24.298981	    |351.914129 |0.014064	      |0.052813	       |0.079720	    |0.038803	          |0.027414	    |0.007060	              |0.277313	 |0.551648	|2.021855	   |45.491006	 |0.003003	   |0.017908	    |0.030186	   |0.006170	       |0.008266	 |0.002646	          |4.833242	   |6.146258	   |33.602542	     |569.356993	|0.022832	       |0.157336	       |0.208624	     |0.065732	          |0.061867	     |0.018061               |
| Min     |6.981000   |9.710000    |43.790000	    |143.500000 |0.052630	      |0.019380	       |0.000000	    |0.000000	          |0.106000	    |0.049960	              |0.111500	 |0.360200	|0.757000	   |6.802000	 |0.001713	   |0.002252	    |0.000000	   |0.000000	       |0.007882	 |0.000895	          |7.930000	   |12.020000	   |50.410000	     |185.200000	|0.071170	       |0.027290	       |0.000000	     |0.000000	          |0.156500	     |0.055040               |
| 25%     |11.700000  |16.170000	 |75.170000	    |420.300000 |0.086370	      |0.064920	       |0.029560	    |0.020310	          |0.161900	    |0.057700	              |0.232400	 |0.833900	|1.606000    |17.850000	 |0.005169	   |0.013080	    |0.015090	   |0.007638	       |0.015160	 |0.002248	          |13.010000   |21.080000	   |84.110000	     |515.300000	|0.116600	       |0.147200	       |0.114500	     |0.064930	          |0.250400	     |0.071460               |
| 50%     |13.370000  |18.840000   |86.240000	    |551.100000 |0.095870	      |0.092630	       |0.061540	    |0.033500	          |0.179200	    |0.061540	              |0.324200	 |1.108000	|2.287000	   |24.530000	 |0.006380	   |0.020450	    |0.025890	   |0.010930       	 |0.018730	 |0.003187	          |14.970000	 |25.410000	   |97.660000	     |686.500000	|0.131300	       |0.211900	       |0.226700	     |0.099930	          |0.282200	     |0.080040               |
| 75%     |15.780000  |21.800000   |104.100000	  |782.700000 |0.105300	      |0.130400	       |0.130700	    |0.074000	          |0.195700	    |0.066120	              |0.478900	 |1.474000	|3.357000	   |45.190000	 |0.008146	   |0.032450	    |0.042050	   |0.014710	       |0.023480	 |0.004558	          |18.790000	 |29.720000	   |125.400000	   |1084.000000	|0.146000	       |0.339100	       |0.382900	     |0.161400	          |0.317900	     |0.092080               |
| Max     |28.110000  |39.280000   |188.500000	  |2501.000000|0.163400	      |0.345400	       |0.426800	    |0.201200	          |0.304000	    |0.097440	              |2.873000	 |4.885000	|21.980000   |542.200000 |0.031130	   |0.135400	    |0.396000	   |0.052790	       |0.078950	 |0.029840	          |36.040000	 |49.540000	   |251.200000	   |4254.000000	|0.222600	       |1.058000	       |1.252000	     |0.291000	          |0.663800	     |0.207500               |

Table diatas memberikan informasi statistik pada masing-masing kolom, antara lain:

- Count adalah jumlah sampel pada data.
- Mean adalah nilai rata-rata.
- Std adalah standar deviasi.
- Min yaitu nilai minimum setiap kolom.
- 25% adalah kuartil pertama. Kuartil adalah nilai yang menandai batas interval - dalam empat bagian sebaran yang sama.
- 50% adalah kuartil kedua, atau biasa juga disebut median (nilai tengah).
- 75% adalah kuartil ketiga.
- Max adalah nilai maksimum.

| Jumlah Baris | Jumlah Kolom |
|--------------|--------------|
| 569          |31            |

Dengan 569 Baris data dengan 31 Jumlah Kolom

### Exploratory Data Analysis - Missing Value, Duplikasi Data, dan Outlier

#### Penanganan Missing Value

Langkah pertama adalah memeriksa data yang hilang pada setiap fitur. Jika ada nilai yang hilang pada fitur yang penting akan dilakukan imputasi dengan menggunakan mean atau median, namun jika yang hilang hanya sedikit maka akan dilakukan penghapusan pada data yang memiliki missing value.

Pengecekan missing value menggunakan kode sebagai berikut.
  ```python
  # Cek missing value
  df.isnull().sum()
  ```
Hasil dari kode tersebut adalah sebagai berikut.

| Column                  |Value|
|-------------------------|-----|
| Diagnosis               |  0  |
| Radius_mean             |  0  |
| Texture_mean            |  0  |
| Perimeter_mean          |  0  |
| Area_mean               |  0  |
| Smoothness_mean         |  0  |
| Compactness_mean        |  0  |
| Concavity_mean          |  0  |
| Concave points_mean     |  0  |
| Symmetry_mean           |  0  |
| Fractal_dimension_mean  |  0  |
| Radius_se               |  0  |
| Texture_se              |  0  |
| Perimeter_se            |  0  |
| Area_se                 |  0  |
| Smoothness_se           |  0  |
| Compactness_se          |  0  |
| Concavity_se            |  0  |
| Concave points_se       |  0  |
| Symmetry_se             |  0  |
| Fractal_dimension_se    |  0  |
| Radius_worst            |  0  |
| Texture_worst           |  0  |
| Perimeter_worst         |  0  |
| Area_worst              |  0  |
| Smoothness_worst        |  0  |
| Compactness_worst       |  0  |
| Concavity_worst         |  0  |
| Concave points_worst    |  0  |
| Symmetry_worst          |  0  |
| Fractal_dimension_worst |  0  |

Setelah dilakukan pengecekan missing value, pada dataset ini tidak memiliki missing value sehingga tidak perlu penanganan missing value.

#### Penanganan Duplikasi Data

Langkah kedua adalah melakukan pengecekan apakah ada data duplikasi di dalam dataset. Data duplikasi bisa terjadi akibat dari kesalahan saat input data atau pengumpulan data.

Pengecekan duplikasi data menggunakan kode sebagai berikut.
  ```python
  # Cek duplikasi data
  jumlah_duplikasi = df.duplicated().sum()
  print(f"Jumlah duplikasi data: {jumlah_duplikasi}")
  ```
![jumlah_duplikasi](https://github.com/user-attachments/assets/db5c2c45-d5e8-4c04-aca0-3a078cf4ebcd)

Dataset ini  memiliki 0 baris duplikasi. Karena tidak memiliki duplikasi data, maka tidak dilakukan penanganan duplikasi data.

#### Penanganan Outlier

Langkah terakhir adalah melakukan pengecekan outlier dengan menggunakan IQR, sebelum dilakukan IQR dilakukan pengecekan data melalui visualisasi boxplot dengan kode sebagai berikut.

  ```python
  # Cek Outlier
  data_numerik = df.select_dtypes(include=['float'])
  kolom_numerik = data_numerik.columns

  for feature in kolom_numerik:
    plt.figure(figsize=(10,6))
    sns.boxplot(x=data_numerik[feature])
    plt.title(f'Box Plot {feature}')
    plt.show()
  ```
Karena ada 30 fitur numerik, diberikan 2 contoh Hasil Box Plot sebagai berikut.
![Box Plot radius_mean](https://github.com/user-attachments/assets/48e0a98f-66b9-43c6-bb09-dfaa499bd5bb)
![Box Plot texture_mean](https://github.com/user-attachments/assets/6bd7462c-be6c-44a2-b888-245bc6d5bb4d)
Karena terlihat ada outlier yang ditunjukkan adanya titik data di luar box plot yang ada dilakukan dengan menggunakan IQR dengan code sebagai berikut.

  ```python
# Atasi Outlier
Q1 = data_numerik[kolom_numerik].quantile(0.25)
Q3 = data_numerik[kolom_numerik].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Mengganti nilai outlier dengan batas atas/bawah
outlier_bersih = data_numerik
for feature in kolom_numerik:
    outlier_bersih[feature] = np.where(outlier_bersih[feature] < lower_bound[feature], lower_bound[feature], outlier_bersih[feature])
    outlier_bersih[feature] = np.where(outlier_bersih[feature] > upper_bound[feature], upper_bound[feature], outlier_bersih[feature])


for feature in outlier_bersih.columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=outlier_bersih[feature])
    plt.title(f'Box Plot {feature}')
    plt.show()
  ```
Hasil penanganan outlier dengan menggunakan metode IQR ditunjukkan sebagai berikut.
![Hasil penanganan outlier radius_mean](https://github.com/user-attachments/assets/ff3fb137-1daa-4490-b6ed-1a41de6b5c26)
![Hasil penanganan outlier texture_mean](https://github.com/user-attachments/assets/f51fda7c-e4dd-4494-ab4d-c23b5a84ba7c)
Hasil tersebut menunjukkan bahwa outlier telah teratasi. kemudian data yang telah dilakukan pengecekan missing value, duplikasi data, dan outlier akan bisa digunakan dengan baik.

### Exploratory Data Analysis - Univariate Analysis
Univariate Analysis ini bertujuan untuk memvisualisasikan setiap fitur secara individual dalam dataset. Sehingga dapat mengetahui informasi lebih mendalam pada masing-masing fitur.

#### Univariate Analysis - Fitur Numerik
Dilakukan visualisasi histogram untuk melihat bentuk distribusi dari fitur numerik, kode dan hasil visualisasi adalah sebagai berikut.
```python
# Histogram Variabel Numerik

# Kolom Numerik
kolom_numerik = df.select_dtypes(include=['float']).columns
 
# Menentukan jumlah baris dan kolom untuk grid subplot
n_cols = 3  # Jumlah kolom yang diinginkan
n_rows = -(-len(kolom_numerik) // n_cols)  # Ceiling division untuk menentukan jumlah baris
 
# Membuat subplot
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
 
# Flatten axes array untuk memudahkan iterasi jika diperlukan
axes = axes.flatten()
 
# Plot setiap variabel
for i, column in enumerate(kolom_numerik):
    sns.histplot(df[column], ax=axes[i], bins=20, kde = True, edgecolor='black')
    axes[i].set_title(column)
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Frekuensi')
 
# Menghapus subplot yang tidak terpakai (jika ada)
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
 
# Menyesuaikan layout agar lebih rapi
plt.tight_layout()
plt.show()
```
![Histogram](https://github.com/user-attachments/assets/5d8118e6-c1ca-475c-9a8c-3e6cdb5cc611)

Histogram digunakan untuk menampilkan frekuensi data dalam rentang nilai tertentu, ini juga memberikan gambaran bagaimana data tersebar, apakah simetris, miring ke kiri atau kanan, dan ada outlier atau tidak.

Dari hasil histogram menunjukkan beberapa fitur seperti radius_mean, texture_mean, perimeter_mean, dan lain-lain cenderung memiliki distribusi yang miring ke kanan (positively skewed), artinya banyak nilai terkonsentrasi di bagian rendah dengan ekor panjang ke kanan. Kemudian juga ada fitur yang distribusinya lebih simetris dan mendekati normal, misalnya smoothness_mean dan symmetry_mean.

#### Univariate Analysis - Fitur Kategorik
Dilakukan visualisasi count plot untuk menghitung jumlah tiap variabel, kode dan hasil visualisasi adalah sebagai berikut.
```python
# Visualisasi Data Target
df['diagnosis'] = df['diagnosis'].astype('category',copy=False)
df['diagnosis'].value_counts().plot(kind='bar')
```
![Countplot kategori](https://github.com/user-attachments/assets/bd77ca50-5f44-4ef7-b57d-a3738aa6ddf0)
Count plot digunakan untuk menghitung banyaknya setiap kategori dari suatu fitur, dari count plot yang ada menunjukkan bahwa Benign (B) menunjukkan lebih banyak daripada Malignant (M).


### Exploratory Data Analysis - Multivariate Analysis
Karena pada dataset ini hanya fitur numerik yang memiliki banyak fitur (30 fitur numerik), maka akan dilakukan korelasi untuk melihat hubungan antar fitur numerik, kode dan hasil visualisasi adalah sebagai berikut.
```python
# Korelasi Variabel Numerik
data_numerik = df.select_dtypes(include=['float'])
plt.figure(figsize=(12, 8))
sns.heatmap(data_numerik.corr(), annot=True, cmap='coolwarm', fmt=".1f")
plt.title('Korelasi Variabel')
plt.show()
```
![Korelasi](https://github.com/user-attachments/assets/aeaafac9-66bb-447e-bb28-dfb0ff6af455)
Heatmap Korelasi diatas menunjukkan hubungan antar fitur yang ada pada dataset. Jika memiliki nilai mendekati 1 fitur akan memiliki korelasi yang sangat kuat, jika semakin menjauhi 1 fitur akan memiliki korelasi yang lemah.

Dari heatmap tersebut ada beberapa fitur seperti radius_mean, perimeter_mean, area_mean yang memiliki korelasi tinggi. dan fitur seperti fractal_dimension dan smoothness cenderung memiliki korelasi lebih rendah dengan fitur lainnya.


## Data Preparation
Pada bagian ini akan ada 3 tahap persiapan data yaitu.
1. Encoding Fitur Kategori
2. Memisahkan Target dan Fitur & Normalisasi Data
3. Train Test Split

### Encoding Fitur Kategori
Encoding dilakukan untuk mempermudah dalam perhitungan karena akan merubah dari kategorikal ke numerikal dan algoritma machine learning tidak dapat langsung bekerja dengan data non-numerik. Pada dataset ini dilakukan mapping secara manual dengan kode sebagai berikut.
```python
# Mapping kolom diagnosis
# M = malignant, B = benign
df.diagnosis = df.diagnosis.map({'M': 1, 'B': 0})
df.head()
```
Dari hasil ini akan merubah dari Malignant (M) menjadi 1 dan Benign menjadi 0.

### Memisahkan Target dan Fitur & Normalisasi Data
```python
# Pisahkan Target dan Fitur
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']
```

```python
# Normalisasi Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled
```

![image](https://github.com/user-attachments/assets/2afadd27-e5c7-4022-a9be-083e987e49ab)
Dilakukan pemisahan antara Target (y) yang merupakan diagnosis, dan Fitur (X) yaitu 30 fitur numerik lainnya.

Setelah itu dilakukan normalisasi atau standarisasi dengan StandardScaler() pada 30 fitur numerik. Penggunaan StandardScaler() dalam machine learning sangat penting pada banyak model, terutama yang berbasis jarak atau yang mengandalkan asumsi distribusi data.

### Train Test Split
```python
# Split Data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
```

```python
print(f"Total sample di semua data: {len(X_scaled)}")
print(f"Total sample di data train: {len(X_train)}")
print(f"Total sample di data test: {len(X_test)}")
```

![image](https://github.com/user-attachments/assets/d571a991-9b91-4f53-a6b4-73ad8ce10672)

Selanjutnya, dilakukan train-test-split dengan pembagian data sebesar 80:20 antara data latih (train) dan data uji (test). Dari total 569 data, setelah dilakukan pembagian, data terbagi menjadi 455 untuk data latih dan 114 untuk data uji.

**Jadi mengapa perlu dilakukan data prepration?**
1. Dilakukan encoding fitur kategorik karena algoritma tidak akan bisa bekerja langsung dengan data non-numerik.
2. Proses normalisasi/standarisasi karena dalam machine learning sangat penting terutama yang berbasis jarak atau yang mengandalkan asumsi distribusi data.
3. Memisahkan data menjadi set pelatihan dan pengujian memungkinkan kita untuk mengevaluasi kinerja model pada data yang tidak pernah dilihat sebelumnya. 




## Modeling
Pada tahap ini, beberapa algoritma machine learning digunakan untuk memecahkan masalah masalah ini, yaitu K-Nearest Neighbors (KNN), Support Vector Machine (SVM), dan Random Forest (RF).

Pada tahap pertama digunakan model dasar pada 3 algoritma yang digunakan kemudian dilatih menggunakan parameter default dari masing-masing model. Kemudian data dilatih tanpa merubah apapun pada parameter model, ini berguna untuk mendapatkan performa dasar tanpa optimasi parameter pada setiap model.

Seletah model dilatih, tahap kedua yang dilakukan adalah melakukan hyperparameter tuning untuk menemukan kombinasi dari parameter terbaik yang dapat meningkatkan kinerja model. Hyperparameter tuning yang digunakan adalah BayesSearchCV. BayesSearchCV dipilih karena menggunakan model probabilistik untuk memprediksi area yang lebih pasti, kemudian juga space nya lebih cepat dibandingkan GridSearchCV atau RandomizedSearchCV.

Setelah kombinasi parameter terbaik telah ditemukan, model akan dibangun kembali dengan kombinasi yang baru dengan parameter terbaiknya, kemudian akan di uji kembali dengan mambandingkan apakah ada peningkatan performa dibandingkan dengan parameter default.

Berikut merupakan penjelasan dari 3 model yang akan digunakan.

1. K-Nearest Neighbors (KNN)
   
![KNN](https://github.com/user-attachments/assets/9687c920-2612-4175-96a1-00bef12e71cd)

K-Nearest Neighbors adalah algoritma supervised learning yang digunakan untuk klasifikasi dan regresi. Untuk klasifikasi, model ini memprediksi kelas sebuah data uji berdasarkan kelas mayoritas dari K tetangga terdekatnya di data pelatihan [[4]](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html).

Kode Pelatihan Model KNN adalah sebagai berikut.
```python
knn = KNeighborsClassifier().fit(X_train, y_train)
```

Cara Kerja pada K-Nearest Neighbors

Tentukan nilai K (jumlah tetangga terdekat). Hitung jarak (misalnya Euclidean) antara data uji dan seluruh data pelatihan. Ambil K data pelatihan terdekat. Prediksi label berdasarkan kelas yang paling sering muncul di antara K tetangga tersebut.

Kelebihan K-Nearest Neighbors
- Sederhana dan intuitif.
- Tidak memerlukan pelatihan eksplisit (lazy learning).
- Dapat digunakan untuk klasifikasi dan regresi.

Kekurangan K-Nearest Neighbors
- Sensitif terhadap skala fitur (perlu normalisasi/standardisasi).
- Tidak cocok untuk dataset besar (komputasi jarak bisa mahal).
- Pemilihan K sangat memengaruhi performa.

Parameter yang digunakan pada K-Nearest Neighbors
| Parameter   | Penjelasan                                                                                     | Default                                                    |
|-------------|------------------------------------------------------------------------------------------------|------------------------------------------------------------|
| n_neighbors | Jumlah tetangga terdekat yang dipertimbangkan (nilai K).                                       |int, default=5                                              |
| weights     |	Bobot kontribusi tetangga. 'uniform' = setara, 'distance' = semakin dekat bobotnya lebih besar.|{‘uniform’, ‘distance’}, callable or None, default=’uniform’|
| metric      |Fungsi jarak: default 'minkowski'. Bisa juga 'euclidean', 'manhattan', dll.                     |str or callable, default=’minkowski’                        |
   
2. Support Vector Machine (SVM)

![SVM](https://github.com/user-attachments/assets/2b2da9dd-1f6d-48de-b3bb-3a963ca4e098)

Support Vector Machine (SVM) adalah metode pembelajaran mesin yang digunakan untuk klasifikasi, regresi, dan deteksi outlier. Tujuan utama SVM dalam klasifikasi adalah menemukan hyperplane optimal yang memisahkan data dari dua kelas dengan margin maksimum [[5]](https://scikit-learn.org/stable/modules/svm.html).

Kode Pelatihan Model SVM adalah sebagai berikut.
```python
svm = SVC().fit(X_train, y_train)
```
Cara Kerja pada Support Vector Machine

SVM mencoba memisahkan kelas dengan hyperplane optimal dan margin maksimum. Untuk data non-linear, kernel digunakan untuk memetakan ke ruang fitur berdimensi lebih tinggi. 

Kelebihan Support Vector Machine

- Efektif di ruang dimensi tinggi – bekerja baik ketika jumlah fitur lebih banyak dari jumlah sampel.
- Bekerja baik untuk margin yang jelas antara kelas.
- Fleksibel dengan kernel trick – bisa digunakan untuk data non-linear menggunakan kernel (RBF, polynomial, dll).
- Hemat memori – hanya menggunakan subset data training (support vectors).

Kekurangan Support Vector Machine

- Kurang efisien untuk dataset besar – karena waktu komputasi meningkat signifikan.
- Pemilihan kernel dan parameter butuh tuning yang hati-hati.
- Tidak cocok untuk data yang memiliki banyak noise atau overlap antar kelas.
- Sulit diinterpretasikan – terutama dengan kernel non-linear.

Parameter yang digunakan pada Support Vector Machine


3. Random Forest (RF)

![Random Forest](https://github.com/user-attachments/assets/1eb67822-57d4-445d-92b4-d6e20cf94403)

Random Forest adalah metode ensemble learning berbasis pohon keputusan (decision tree) yang digunakan untuk klasifikasi dan regresi. Algoritma ini membangun banyak pohon keputusan selama proses pelatihan dan mengeluarkan kelas sebagai hasil prediksi berdasarkan mayoritas voting dari semua pohon (untuk klasifikasi) atau rata-rata (untuk regresi) [[6]](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).

Kode Pelatihan Model Random Forest adalah sebagai berikut.
```python
rf = RandomForestClassifier().fit(X_train, y_train)
```
Cara Kerja pada Random Forest

Membuat banyak decision tree (default-nya 100). Kemudian setiap pohon dilatih menggunakan bootstrap sampling (subset acak dari data pelatihan dengan pengembalian). Pada setiap split, hanya subset acak dari fitur yang dipertimbangkan untuk memecah node — ini mengurangi korelasi antar pohon. Untuk klasifikasi, hasil akhir ditentukan berdasarkan voting terbanyak dari semua pohon.

Kelebihan Random Forest

- Robust terhadap overfitting, dibanding single decision tree.
- Bisa menangani data besar dengan banyak fitur.
- Dapat mengukur pentingnya fitur.
- Cocok untuk data dengan kombinasi fitur numerik dan kategorikal.

Kekurangan Random Forest

- Kurang interpretatif – hasilnya seperti “black box”, sulit dipahami dibanding single decision tree.
- Lambat saat prediksi pada data besar, karena harus menghitung hasil banyak pohon.
- Model besar dan berat – konsumsi memori dan waktu tinggi untuk banyak data dan banyak pohon.

Parameter yang digunakan pada Random Forest



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

**[4]** Scikit-learn Developers, “sklearn.neighbors.KNeighborsClassifier,” Scikit-learn, 2024. [Online]. Available: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html. [Accessed: 17-May-2025].

**[5]** Scikit-learn Developers, “Support Vector Machines,” Scikit-learn, 2024. [Online]. Available: https://scikit-learn.org/stable/modules/svm.html. [Accessed: 17-May-2025].

**[6]** Scikit-learn Developers, “sklearn.ensemble.RandomForestClassifier,” Scikit-learn, 2024. [Online]. Available: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html. [Accessed: 17-May-2025].

**---Ini adalah bagian akhir laporan---**

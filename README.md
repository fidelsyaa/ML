# Laporan Proyek Machine Learning - Fidel Lusiana Putri

## Domain Proyek
Diabetes melitus adalah penyakit metabolik kronis yang ditandai dengan meningkatnya kadar gula darah secara abnormal. Menurut World Health Organization (WHO), sekitar 422 juta orang di dunia mengidap diabetes dan angka ini terus meningkat setiap tahunnya, khususnya di negara berkembang. Deteksi dini dan prediksi terhadap potensi seseorang terkena diabetes sangat penting untuk mencegah komplikasi serius seperti penyakit jantung, kebutaan, dan gagal ginjal.

Kelompok masyarakat suku Pima di Amerika Serikat menunjukkan prevalensi diabetes tipe 2 yang jauh lebih tinggi dibandingkan populasi umum. Oleh karena itu, dataset Pima Indians Diabetes sering digunakan sebagai basis analisis prediktif dalam bidang kesehatan.

Pendeteksian diabetes secara konvensional biasanya memerlukan prosedur medis yang kompleks, mahal, dan memakan waktu. Masalah prediksi diabetes pada wanita suku Pima penting diselesaikan karena prevalensinya yang tinggi, risiko komplikasi serius, serta beban ekonomi dan sosial yang besar. Banyak penderita tidak menyadari kondisinya hingga terlambat, sementara akses layanan kesehatan sering terbatas pada kelompok ini. Untuk mengatasi hal tersebut, pendekatan prediktif berbasis data digunakan dengan memanfaatkan algoritma machine learning seperti Logistic Regression, SVM, KNN, dan Random Forest. Model dilatih menggunakan data kesehatan sederhana, dievaluasi dengan metrik akurasi dan ROC-AUC, serta ditingkatkan melalui tuning. Solusi ini berpotensi menjadi sistem pendukung keputusan bagi deteksi dini dan penanganan diabetes yang lebih efektif, dengan memanfaatkan data riwayat medis sederhana serta model machine learning, prediksi diabetes dapat dilakukan secara cepat dan murah untuk mendukung skrining awal atau sistem peringatan dini.

- [ Diabetic prediction based on machine learning using PIMA indian dataset](https://www.researchgate.net/profile/Subhi-Zeebaree/publication/382399289_Diabetic_Prediction_based_on_Machine_Learning_Using_PIMA_Indian_Dataset/links/669ba3d9cb7fbf12a45fc538/Diabetic-Prediction-based-on-Machine-Learning-Using-PIMA-Indian-Dataset.pdf)
- [Application of Data Mining Methods and Techniques for Diabetes Prediction Using Pima Indian Dataset](https://link.springer.com/chapter/10.1007/978-981-97-0573-3_47)
- [Advancing Precision Healthcare Analytics: Machine Learning Approach-es for Diabetes Prognosis using the PIMA Indian Dataset](https://journals.flvc.org/FLAIRS/article/view/135329)
- [Techniques of Machine Learning for the Purpose of Predicting Diabetes Risk in PIMA Indians](https://www.e3s-conferences.org/articles/e3sconf/abs/2023/67/e3sconf_icmpc2023_01151/e3sconf_icmpc2023_01151.html)
- [Using the ADAP learning algorithm to forecast the onset of diabetes mellitus](https://pmc.ncbi.nlm.nih.gov/articles/PMC2245318/)
 

## Business Understanding
Diabetes merupakan penyakit kronis yang mempengaruhi jutaan orang di seluruh dunia. Wanita dari suku Pima memiliki prevalensi diabetes yang lebih tinggi dibandingkan populasi umum. Oleh karena itu, penting untuk membangun sistem prediktif guna mendeteksi potensi diabetes secara dini berdasarkan data kesehatan.

### Problem Statements
- Tingginya prevalensi diabetes pada wanita suku Pima dan keterbatasan dalam deteksi dini menjadikan banyak kasus tidak terdiagnosis hingga munculnya komplikasi serius.
- Kurangnya sistem prediksi atau alat bantu diagnostik berbasis data untuk membantu skrining awal pada populasi yang rentan.
- Keterbatasan sumber daya dan akses layanan medis membuat solusi manual menjadi tidak efisien dalam skala besar.

### Goals
- Membangun model prediktif yang mampu mendeteksi potensi diabetes secara dini menggunakan data kesehatan dasar.
- Mengembangkan solusi machine learning berbasis dataset medis untuk mempermudah skrining awal pada kelompok wanita suku Pima.
- Menyediakan model otomatis dan terukur yang dapat membantu mengurangi ketergantungan pada pemeriksaan manual dan memfasilitasi penyaringan berskala besar.

### Solution statements
   - Baseline Model: Menggunakan Logistic Regression sebagai baseline model untuk melihat performa awal.
    - Comparative Modeling: Menggunakan beberapa algoritma machine learning (Random Forest, SVM, KNN) untuk dibandingkan performanya.
    - Hyperparameter Tuning: Melakukan GridSearchCV pada model KNN dan Random Forest untuk meningkatkan akurasi dan generalisasi.

Model-model ini dievaluasi menggunakan metrik klasifikasi seperti akurasi, precision, recall, F1-score, dan ROC-AUC.

## Data Understanding

Dataset ini berasal dari National Institute of Diabetes and Digestive and Kidney Diseases. Tujuan utama dari dataset ini adalah untuk memprediksi secara diagnostik apakah seorang pasien memiliki diabetes atau tidak, berdasarkan beberapa pengukuran diagnostik yang terdapat dalam dataset tersebut. Ada beberapa batasan yang diterapkan dalam pemilihan data dari database yang lebih besar. Secara khusus, semua pasien yang ada dalam dataset ini adalah perempuan dengan usia minimal 21 tahun dan memiliki keturunan Pima Indian. Dataset yang digunakan adalah Pima Indians Diabetes Database, tersedia di [UCI Machine Learning Repository](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database/data)

Dari informasi dataset diatas, didapatkan sejumlah informasi dari dataset pima diabetes, yaitu:
- Jumlah baris: 768
- Jumlah kolom: 9 (8 fitur + 1 label Outcome)
- Terdapat beberapa nilai nol pada kolom-kolom seperti Glucose, BloodPressure, SkinThickness, Insulin, dan BMI, yang tidak logis secara medis dan oleh karena itu dianggap sebagai missing value.

### Variabel-variabel pada Pima Indians Diabetes UCI dataset adalah sebagai berikut:
- Pregnancies:  Jumlah kehamilan
- Glucose: Konsentrasi glukosa plasma selama 2 jam dalam tes toleransi glukosa oral
- BloodPressure: Tekanan darah diastolik (mm Hg)
- SkinThickness: Ketebalan lipatan kulit triceps (mm)
- Insulin: Kadar insulin serum 2 jam (mu U/ml)
- BMI: Indeks massa tubuh (berat dalam kg / (tinggi dalam m)^2)
- DiabetesPedigreeFunction: Keturunan silsilah diabetes dari keluarga (indikator genetik)
- Age: Usia (dalam tahun)
- Outcome: Target variable (0 = non-diabetes, 1 = diabetes)

Dari eksplorasi data yang dilakukan menggunakan fungsi Melalui fungsi diabetes.info() diperoleh bahwa semua fitur bertipe numerik (int64 dan float64). Tidak ada nilai null secara eksplisit. Distribusi statistik dari dataset juga menujukkan bahwa fitur seperti Glucose, BloodPressure, SkinThickness, Insulin, dan BMI memiliki nilai minimum 0, yang secara medis tidak logis (misalnya tekanan darah 0), dan akan ditindaklanjuti di tahap Data Preparation.

Berdasarkan gambar statistik deskriptif dari dataset Pima Indians Diabetes Database tersebut, terlihat bahwa setiap fitur memiliki jumlah data yang sama yaitu sebanyak 768 entri. Namun, beberapa fitur memiliki nilai minimum yang tidak logis secara medis dan perlu diperhatikan dalam tahap data cleaning.

Fitur Glucose memiliki nilai minimum sebesar 0, yang secara medis tidak mungkin terjadi karena manusia tidak bisa hidup tanpa kadar glukosa dalam darah. Hal ini menunjukkan bahwa nilai nol tersebut kemungkinan merupakan representasi dari data yang hilang (missing value). Nilai rata-rata glukosa berada di 120.89, dengan nilai maksimum mencapai 199. Hal serupa juga terjadi pada fitur BloodPressure, yang memiliki nilai minimum 0, padahal tekanan darah tidak bisa nol, sehingga perlu dipertimbangkan sebagai data hilang.

Fitur SkinThickness dan Insulin juga menunjukkan nilai minimum 0, yang tidak realistis dalam konteks medis, mengindikasikan adanya nilai hilang. Rata-rata insulin adalah 79.79 dengan standar deviasi cukup besar yaitu 115.24, serta nilai maksimum mencapai 846, yang menunjukkan kemungkinan adanya outlier. Fitur BMI pun memiliki nilai minimum 0, padahal nilai BMI tidak bisa nol, sehingga harus dianggap sebagai nilai yang tidak valid.

Sementara itu, fitur DiabetesPedigreeFunction, Age, dan Outcome tidak memiliki nilai nol. Diabetes Pedigree Function memiliki rata-rata 0.47 dan maksimum 2.42, menunjukkan distribusi yang cenderung skewed. Usia pasien memiliki rentang antara 21 hingga 81 tahun, dengan rata-rata usia 33.24 tahun, mencerminkan bahwa dataset ini memang ditujukan untuk populasi dewasa. Terakhir, fitur Outcome memiliki nilai rata-rata sekitar 0.35, yang berarti sekitar 35% dari pasien dalam dataset ini terdiagnosis diabetes.

Secara keseluruhan, terlihat bahwa banyak fitur yang mengandung nilai nol yang secara medis tidak masuk akal, sehingga perlu dilakukan proses penanganan missing values (seperti imputasi dengan median) sebelum masuk ke tahap pelatihan model.

Dilakukan juga korelsi antar fitur pada dataset menggunakan heatmap, Berdasarkan heatmap korelasi, terlihat bahwa variabel Outcome (indikator apakah seseorang mengidap diabetes atau tidak) memiliki korelasi positif paling kuat dengan Glucose sebesar 0.47. Ini menunjukkan bahwa semakin tinggi kadar glukosa seseorang, semakin besar kemungkinan orang tersebut terdeteksi diabetes. Hal ini sesuai dengan pemahaman medis bahwa kadar gula darah tinggi merupakan indikator utama diabetes.

Selain Glucose, fitur BMI juga menunjukkan korelasi sedang dengan Outcome sebesar 0.29, diikuti oleh Age (0.24) dan Pregnancies (0.22). Artinya, indeks massa tubuh, usia, dan jumlah kehamilan (pada perempuan) juga turut berkontribusi dalam risiko diabetes, meskipun tidak sekuat glukosa.

Sementara itu, fitur-fitur seperti BloodPressure, SkinThickness, Insulin, dan DiabetesPedigreeFunction memiliki korelasi yang lebih lemah terhadap Outcome, dengan nilai di bawah 0.2. Korelasi negatif maupun sangat kecil seperti ini menunjukkan bahwa keterkaitannya terhadap kemungkinan diabetes lebih rendah, meskipun bisa saja masih memiliki peran ketika digabungkan dengan fitur lainnya.

Dari sisi hubungan antar fitur, terlihat bahwa Insulin memiliki korelasi cukup kuat dengan SkinThickness (0.44) dan Glucose (0.33), serta BMI (0.20). Korelasi ini dapat dijelaskan karena parameter-parameter tersebut sering kali saling berkaitan dalam kondisi metabolik tubuh. distribusi kelas menunjukkan bahwa dataset ini memiliki lebih banyak orang yang tidak menderita diabetes dibandingkan dengan yang menderita, sehingga perlu penanganan khusus dalam proses pemodelan agar model tidak bias.

- Univariate Analysis
1. Distribusi kelas target (Outcome) diperiksa dengan menggunakan sns.countplot().
2. Visualisasi distribusi setiap fitur dilakukan dengan histogram dan boxplot untuk melihat potensi outlier dan distribusi yang skewed.

- Multivariate Analysis
1. Korelasi antar fitur dan dengan target Outcome dihitung menggunakan matriks korelasi.
2. Visualisasi hubungan antar fitur dilakukan menggunakan pairplot dan heatmap.
3. Hasil menunjukkan bahwa Glucose memiliki korelasi paling signifikan dengan Outcome.

## Data Preparation
1. Handling Missing Values
2. Handling Outliers
3. Feature Engineering
4. Pemisahan Fitur dan Target
5. Train-Test-Split
6. Standarisasi


1. Handling Missing Values
Meski secara eksplisit tidak ada nilai NaN, beberapa kolom memiliki nilai nol yang secara medis tidak mungkin, misalnya: Glucose = 0, BloodPressure = 0, dst. Nilai-nilai nol ini dianggap tidak valid dan digantikan dengan nilai median dari masing-masing kolom. Nilai-nilai ini diganti menggunakan median karena distribusi fitur bersifat skewed dan median lebih robust terhadap outlier.

Sebelum dilakukan imputasi dengan mengisi nilai nol menggunakan media, berikut adalah fitur yang memiliki angka 0:
Jumlah nilai nol sebelum imputasi:
Glucose: 5 nilai nol
BloodPressure: 35 nilai nol
SkinThickness: 227 nilai nol
Insulin: 374 nilai nol
BMI: 11 nilai nol
    
2. Handling Outliers
Mengidentifikasi dan menangani outliers menggunakan visualisasi boxplot. Jika diperlukan, outliers dapat dihapus atau distandarisasi. Outlier dideteksi menggunakan boxplot dan ditangani dengan metode IQR (Inter Quartile Range). Setiap nilai yang berada di luar [Q1 - 1.5IQR, Q3 + 1.5IQR] dianggap sebagai outlier dan ditangani sesuai strategi yang dipilih (misalnya diganti atau dihapus). Outlier perlu ditangani dalam proses analisis data dan pemodelan karena mereka dapat mempengaruhi hasil analisis dan akurasi model prediksi secara signifikan
Jumlah data sebelum outlier removal: (768, 9)
Jumlah data setelah outlier removal: (615, 9)

Setelah outliers ditangani data yang awalnya ada 768, menjadi 615.

3. Feature Engineering:
- BMI_Glucose_Ratio: Rasio antara BMI (Indeks Massa Tubuh) dan Glucose (Kadar glukosa darah).
Tujuan: Menggabungkan dua fitur medis penting yang berpotensi memiliki hubungan kompleks terhadap risiko diabetes. Rasio ini dapat menangkap interaksi antara obesitas dan kadar glukosa yang mungkin menjadi indikator kuat untuk prediksi.
Manfaat: Memberikan fitur turunan yang bisa mengungkap hubungan non-linier antara BMI dan Glucose terhadap Outcome.

- Age_Bins: Kategori umur yang dibagi dalam rentang 20–30, 31–40, dst.
Tujuan: Membantu model menangkap pola risiko berdasarkan kelompok usia. Fitur kategorikal ini dapat memperjelas tren yang tersembunyi jika hanya mengandalkan nilai numerik Age.
Manfaat: Mempermudah interpretasi dan bisa meningkatkan kinerja model yang sensitif terhadap variabel kategori.

4. Pemisahan Fitur dan Target
Dataset yang telah dibersihkan dipisahkan menjadi:
- X → seluruh fitur prediktor (independen)
- y → kolom Outcome sebagai target (label diabetes: 0 = tidak, 1 = ya)

5. Train-Test Split: Data dibagi menjadi data pelatihan dan pengujian menggunakan train_test_split() dengan parameter stratifikasi untuk menjaga distribusi kelas. 90% untuk pelatihan, 10% untuk pengujian. Tujuan dilakukannya untuk memastikan model dievaluasi pada data yang belum pernah dilihat (uji generalisasi model). Parameter random_state=123 digunakan untuk reproducibility (hasil tetap sama setiap kali dijalankan).

6. Standardisasi: Digunakan StandardScaler() dari Scikit-learn dalam pipeline sebelum modeling. Ini penting karena model seperti K-Nearest Neighbors (KNN) dan Support Vector Machine (SVM) sensitif terhadap skala fitur. 
- Digunakan StandardScaler() dari Scikit-learn untuk mengubah semua fitur numerik menjadi distribusi standar (mean = 0, std = 1).
- scaler.fit() dilakukan hanya pada data latih. 
- Transformasi kemudian diterapkan ke X_train dan X_test untuk mencegah data leakage.

Deskripsi statistik fitur setelah distandarisasi ditampilkan untuk memastikan distribusi sudah seragam. Tujuannyaa untuk mengecek bahwa setiap fitur memiliki nilai tengah mendekati 0 dan standar deviasi mendekati 1.

## Modeling
Pada tahap ini, dilakukan pengembangan beberapa model machine learning untuk memprediksi apakah seseorang mengidap diabetes berdasarkan fitur-fitur seperti kadar glukosa, tekanan darah, BMI, dan lain-lain. Model-model yang digunakan antara lain:
- Algoritma yang Digunakan:
1. Logistic Regression
2. Random Forest (Default)
3. Support Vector Machine (SVM)
4. K-Nearest Neighbors (KNN) dengan tuning
5. Random Forest (Tuned via GridSearchCV)

Semua model dibangun dengan menggunakan Scikit-learn, dan preprocessing dilakukan menggunakan Pipeline untuk menjaga alur yang efisien.

- Tahapan Modeling & Parameter:
1. Logistic Regression
Model ini menggunakan fungsi logistik (sigmoid) untuk mengestimasi probabilitas bahwa suatu data termasuk dalam kelas 1. Logistic Regression bekerja dengan mengoptimalkan fungsi log-loss untuk memisahkan dua kelas.

⚙️ Parameter:
- penalty: 'l2' (default) → regularisasi Ridge
- C: 1.0 (default) → kekuatan regularisasi
- solver: 'lbfgs' (default)
3. Random Forest (Default)
   Random Forest adalah ensemble dari banyak pohon keputusan (Decision Tree). Setiap pohon dilatih pada subset acak dari data (bootstrap sampling), dan fitur yang digunakan juga dipilih secara acak. Hasil prediksi diambil berdasarkan mayoritas voting dari semua pohon. Model ensemble berbasis pohon keputusan. Menghasilkan prediksi berdasarkan voting dari banyak pohon. dan digunakan Parameter default. digunakan parameter random_state=42.
  
⚙️ Parameter:
- n_estimators: 100 (default) → jumlah pohon
- criterion: 'gini' (default)
- max_depth: None
- random_state: 42
4. Support Vector Machine (SVM)
SVM mencoba mencari hyperplane terbaik yang memisahkan dua kelas dengan margin terbesar. Untuk data non-linear, SVM menggunakan kernel trick untuk memetakan data ke dimensi yang lebih tinggi.

⚙️ Parameter:
- kernel: 'rbf' (default)
- C: 1.0 (default) → trade-off antara margin dan error
- gamma: 'scale' (default)
4. K-Nearest Neighbors (Tuned)
  KNN adalah algoritma non-parametrik berbasis instance-based learning. KNN memprediksi kelas dari suatu sampel berdasarkan mayoritas label dari k tetangga terdekat (berdasarkan jarak—biasanya Euclidean). Model ini tidak melakukan proses training secara eksplisit, namun menyimpan data latih dan melakukan perhitungan saat prediksi.
- Dibungkus dalam Pipeline dengan StandardScaler.
- Tuning menggunakan GridSearchCV pada parameter:
    - n_neighbors: [3, 4, 5, 7, 9, 11]
    - weights: ['uniform', 'distance']
    - metric: ['euclidean', 'manhattan', 'minkowski']
5. Random Forest (Tuned)
- Juga dalam pipeline dengan scaler.
- Parameter yang dituning:
    - n_estimators: [50, 100, 200]
    - max_depth: [None, 10, 20, 30]
    - min_samples_split: [2, 5, 10]
    - min_samples_leaf: [1, 2, 4]
    - max_features: ['sqrt', 'log2', None]
    - bootstrap: [True, False]
    - criterion: ['gini', 'entropy']

- Kekurangan dan kelebihan masing-masing algoritma yang digunakan:
1. Logistic Regression
    - Kelebihan: Sederhana, interpretasi mudah
    - Kekurangan: Tidak cocok untuk data non-linear
2. Random Forest
    - Kelebihan: Menangani data kompleks, robust terhadap overfitting
    - Kekurangan: Interpretasi sulit, training lebih lama
3. SVM
    - Kelebihan: Baik untuk data dengan margin yang jelas
    - Kekurangan: Sensitif terhadap skala, tuning parameter rumit
4. KNN
    - Kelebihan: Simpel, hasil optimal setelah tuning, performa tinggi
    - Kelemahan: Lambat saat prediksi (karena lazy learner),  sangat tergantung skala dan jumlah fitur
5. RF Tuned
    - Kelebihan: Performanya meningkat setelah tuning
    - Kelemahan: Masih kalah dengan KNN Tuned dalam hal akurasi dan F1

Dari seluruh algoritma yang digunakan model KNN Tuned dipilih sebagai model terbaik berdasarkan evaluasi berikut:
- AUC tertinggi (0.84)
- Akurasi tertinggi (0.84)
- Precision dan F1-score paling tinggi dibanding model lain
- ROC curve menunjukkan performa yang konsisten lebih baik dibanding model lain
Model ini memberikan keseimbangan terbaik antara false positive dan false negative dalam konteks medis yang penting seperti diagnosis diabetes.

- Feature Importance
Dari hasil Permutation Feature Importance terhadap model KNN Tuned, fitur paling penting dalam prediksi adalah:
  1. Glucose
  2. BMI
  3. Age Bins
  4. Skin Thickness
  5. BMI-Glucose Ratio

## Evaluation
Pada tahap evaluasi, kita menggunakan beberapa metrik klasifikasi untuk mengukur kinerja model dalam memprediksi diabetes pada wanita suku Pima. Metrik yang digunakan meliputi Accuracy, Precision, Recall, F1-Score, dan ROC-AUC. Setiap metrik ini memberikan pandangan yang berbeda tentang performa model, khususnya dalam konteks deteksi dini diabetes, yang sangat penting untuk diagnosis yang cepat dan efektif.
1. Accuracy
   
Metrik ini memberikan gambaran umum tentang performa model, tetapi bisa menyesatkan jika dataset tidak seimbang (misalnya, banyak kasus negatif). Pada kasus deteksi diabetes, meskipun model memiliki accuracy tinggi, bisa saja model tersebut lebih sering memprediksi "non-diabetes" dan gagal mendeteksi sebagian besar kasus diabetes.

TP+TN / TP+TN+FP+FN

- TP = True Positives (kasus positif yang benar terdeteksi)
- TN = True Negatives (kasus negatif yang benar terdeteksi)
- FP = False Positives (kasus negatif yang salah terdeteksi sebagai positif)
- FN = False Negatives (kasus positif yang salah terdeteksi sebagai negatif)
  
2. Precision
   
Precision mengukur akurasi dari prediksi positif. Dengan kata lain, seberapa banyak dari prediksi yang positif benar-benar benar. Precision tinggi berarti model jarang membuat prediksi positif yang salah (false positives). Dalam konteks prediksi diabetes, precision yang tinggi penting karena bisa meminimalkan jumlah pasien yang salah diberi diagnosis diabetes (false positives).

 TP / (TP + FP)

4. Recall
   
Recall mengukur seberapa banyak kasus positif yang sebenarnya berhasil terdeteksi oleh model. Recall tinggi berarti model mampu mendeteksi sebagian besar kasus positif, yang dalam hal ini adalah pasien yang benar-benar mengidap diabetes. Recall yang tinggi sangat penting dalam konteks kesehatan karena dapat mengurangi jumlah pasien yang tidak terdeteksi (false negatives), yang dapat berisiko mengalami komplikasi serius jika tidak segera ditangani.

TP / (TP + FN)

6. F1-score
   
   F1-Score adalah rata-rata harmonik antara precision dan recall, yang memberikan keseimbangan antara keduanya. F1-Score sangat berguna ketika kita ingin menyeimbangkan antara precision dan recall. F1-Score memberikan gambaran yang lebih seimbang tentang performa model, terutama ketika precision dan recall memiliki peran yang sama penting. Dalam kasus prediksi diabetes, F1-Score yang tinggi mengindikasikan model berhasil mendeteksi pasien diabetes dengan sedikit kesalahan baik dalam hal false positives maupun false negatives.
   
8. ROC-AUC
   
  F1-Score memberikan gambaran yang lebih seimbang tentang performa model, terutama ketika precision dan recall memiliki peran yang sama penting. Dalam kasus prediksi diabetes, F1-Score yang tinggi mengindikasikan model berhasil mendeteksi pasien diabetes dengan sedikit kesalahan baik dalam hal false positives maupun false negatives.
AUC tinggi menunjukkan bahwa model dapat memisahkan dengan baik antara penderita diabetes dan non-diabetes. Ini sangat penting dalam aplikasi medis, karena membantu mengidentifikasi individu berisiko yang membutuhkan perhatian medis lebih lanjut.

Penjelasan Hasil Proyek Berdasarkan Metrik Evaluasi:

### 🔹 **1. Logistic Regression**
- **Accuracy**  : 0.7903  
- **Precision** : 0.7143  
- **Recall**    : 0.5263  
- **F1-Score**  : 0.6061  
- **ROC AUC**   : 0.83  


### 🔹 **2. Random Forest**
- **Accuracy**  : 0.7581  
- **Precision** : 0.6000  
- **Recall**    : 0.6316  
- **F1-Score**  : 0.6154  
- **ROC AUC**   : 0.80  

### 🔹 **3. Support Vector Machine (SVM)**
- **Accuracy**  : 0.7903  
- **Precision** : 0.7143  
- **Recall**    : 0.5263  
- **F1-Score**  : 0.6061  
- **ROC AUC**   : 0.80  


### 🔹 **4. K-Nearest Neighbors (Tuned)**
- **Accuracy**  : 0.8387  
- **Precision** : 0.8462  
- **Recall**    : 0.5789  
- **F1-Score**  : 0.6875  
- **ROC AUC**   : 0.84  


### 🔹 **5. Random Forest (Tuned)**
- **Accuracy**  : 0.7581  
- **Precision** : 0.6111  
- **Recall**    : 0.5789  
- **F1-Score**  : 0.5946  
- **ROC AUC**   : 0.81  

 
 Analisis dan Interpretasi:
- KNN (Tuned) memiliki performa terbaik:
    - Accuracy dan F1-Score tertinggi, artinya model membuat prediksi benar dalam proporsi terbesar.
    - Precision tinggi: mayoritas prediksi positif (diabetes) memang benar.
    - ROC AUC tertinggi menunjukkan kemampuan memisahkan dua kelas secara menyeluruh.
- Random Forest (Tuned) memberikan keseimbangan yang cukup baik, tetapi tidak mampu mengungguli KNN dalam metrik utama.
- Logistic Regression dan SVM memberikan hasil yang cukup seimbang, tetapi Recall-nya rendah — artinya banyak kasus diabetes yang tidak terdeteksi.
- Recall pada semua model cenderung lebih rendah dari Precision, artinya model lebih cenderung melewatkan kasus positif daripada memprediksi positif yang salah.

Model yang dikembangkan bertujuan untuk membantu deteksi dini diabetes pada wanita suku Pima, dengan memanfaatkan data kesehatan dasar. Evaluasi model bertujuan untuk mengetahui apakah solusi ini mampu menjawab problem statement dan mencapai goals yang telah ditentukan.
- Hasil Evaluasi:
1. KNN (Tuned) menunjukkan performa terbaik dengan akurasi tertinggi (0.84) dan ROC AUC tertinggi (0.84), yang mengindikasikan kemampuan model dalam membedakan antara penderita dan non-penderita diabetes.
2. Random Forest (Tuned) memberikan keseimbangan antara akurasi dan precision tetapi tidak mampu mengungguli KNN dalam hal performa keseluruhan.
2. Logistic Regression dan SVM memiliki hasil yang cukup seimbang namun lebih rendah pada recall, yang berarti banyak kasus diabetes yang tidak terdeteksi.

### Dampak Evaluasi terhadap Business Understanding
1. Apakah model sudah menjawab setiap Problem Statement?
- Problem Statement 1: Tingginya prevalensi diabetes pada wanita suku Pima dan keterbatasan dalam deteksi dini.
Dampak: Model yang dibangun dapat membantu menjawab problem ini dengan memberikan alat bantu yang efektif untuk deteksi dini diabetes pada wanita suku Pima. Dengan menggunakan algoritma machine learning, kita dapat memprediksi potensi diabetes lebih cepat dan lebih murah daripada prosedur medis tradisional. Hasil evaluasi menunjukkan bahwa KNN (Tuned) memiliki akurasi tinggi (0.84), memberikan dasar yang kuat untuk deteksi dini.

- Problem Statement 2: Kurangnya sistem prediksi atau alat bantu diagnostik berbasis data untuk membantu skrining awal.
Dampak: Evaluasi model menunjukkan bahwa model seperti KNN dan Random Forest (Tuned) mampu memberikan solusi berbasis data untuk skrining awal. Model ini memberikan keseimbangan yang baik antara recall dan precision, yang berarti dapat meminimalkan risiko gagal mendeteksi pasien dengan diabetes serta mengurangi jumlah pasien yang salah didiagnosis.

- Problem Statement 3: Keterbatasan sumber daya dan akses layanan medis.
Dampak: Model machine learning yang cepat dan efisien dapat mengatasi keterbatasan ini dengan memberikan solusi otomatis yang dapat digunakan dalam skala besar, tanpa perlu prosedur medis mahal dan memakan waktu. Ini sangat relevan mengingat keterbatasan dalam akses ke layanan kesehatan, terutama di daerah terpencil.

2. Apakah model berhasil mencapai setiap Goal yang diharapkan?
- Goal 1: Membangun model prediktif yang mampu mendeteksi potensi diabetes secara dini.
Dampak: Ya, model berhasil mencapai goal ini. Berdasarkan hasil evaluasi, model KNN (Tuned) berhasil mencapai akurasi tertinggi (0.84) dan ROC-AUC terbaik (0.84), yang menandakan bahwa model ini dapat diandalkan untuk deteksi dini diabetes.

- Goal 2: Mengembangkan solusi machine learning berbasis dataset medis untuk mempermudah skrining awal pada kelompok wanita suku Pima.
Dampak: Goal ini juga tercapai dengan baik. Model yang dikembangkan menyediakan solusi berbasis data yang mudah diakses dan diimplementasikan, memungkinkan skrining awal pada wanita suku Pima dengan cara yang lebih terukur dan efisien.

- Goal 3: Menyediakan model otomatis dan terukur yang dapat membantu mengurangi ketergantungan pada pemeriksaan manual dan memfasilitasi penyaringan berskala besar.
Dampak: Model yang telah dievaluasi dapat memfasilitasi penyaringan besar-besaran tanpa memerlukan pemeriksaan manual yang rumit, memungkinkan deteksi lebih luas pada populasi yang lebih besar, dan menyediakan sistem pendukung keputusan yang lebih cepat dan efisien.

3. Apakah setiap Solution Statement yang direncanakan berdampak?
- Solution Statement 1: Menggunakan Logistic Regression sebagai baseline model.
Dampak: Logistic Regression sebagai baseline memberikan gambaran awal tentang kinerja model. Meskipun model ini memberikan hasil yang layak, namun model lainnya seperti KNN (Tuned) menunjukkan hasil yang lebih baik dalam hal akurasi dan keseimbangan antara precision dan recall.

- Solution Statement 2: Membandingkan berbagai algoritma (Random Forest, SVM, KNN) untuk memilih yang terbaik.
Dampak: Perbandingan ini sangat penting dalam memilih model yang terbaik. Berdasarkan evaluasi, KNN (Tuned) memberikan performa terbaik di antara model yang diuji, yang memberikan dampak positif dalam meningkatkan deteksi dini diabetes.

- Solution Statement 3: Melakukan Hyperparameter Tuning untuk meningkatkan akurasi.
Dampak: Hyperparameter tuning pada KNN dan Random Forest terbukti efektif dalam meningkatkan performa model, terutama dalam hal akurasi dan keseimbangan antara false positive dan false negative.

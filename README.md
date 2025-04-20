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

![variabel](https://github.com/fidelsyaa/ML/blob/main/variabel.png)

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

![jenis](https://github.com/fidelsyaa/ML/blob/main/jenis.png)

Dari eksplorasi data yang dilakukan menggunakan fungsi Melalui fungsi diabetes.info() diperoleh bahwa semua fitur bertipe numerik (int64 dan float64). Tidak ada nilai null secara eksplisit. Distribusi statistik dari dataset juga menujukkan bahwa fitur seperti Glucose, BloodPressure, SkinThickness, Insulin, dan BMI memiliki nilai minimum 0, yang secara medis tidak logis (misalnya tekanan darah 0), dan akan ditindaklanjuti di tahap Data Preparation.

![describe](https://github.com/fidelsyaa/ML/blob/main/describe.png)

Berdasarkan gambar statistik deskriptif dari dataset Pima Indians Diabetes Database tersebut, terlihat bahwa setiap fitur memiliki jumlah data yang sama yaitu sebanyak 768 entri. Namun, beberapa fitur memiliki nilai minimum yang tidak logis secara medis dan perlu diperhatikan dalam tahap data cleaning.

Fitur Glucose memiliki nilai minimum sebesar 0, yang secara medis tidak mungkin terjadi karena manusia tidak bisa hidup tanpa kadar glukosa dalam darah. Hal ini menunjukkan bahwa nilai nol tersebut kemungkinan merupakan representasi dari data yang hilang (missing value). Nilai rata-rata glukosa berada di 120.89, dengan nilai maksimum mencapai 199. Hal serupa juga terjadi pada fitur BloodPressure, yang memiliki nilai minimum 0, padahal tekanan darah tidak bisa nol, sehingga perlu dipertimbangkan sebagai data hilang.

Fitur SkinThickness dan Insulin juga menunjukkan nilai minimum 0, yang tidak realistis dalam konteks medis, mengindikasikan adanya nilai hilang. Rata-rata insulin adalah 79.79 dengan standar deviasi cukup besar yaitu 115.24, serta nilai maksimum mencapai 846, yang menunjukkan kemungkinan adanya outlier. Fitur BMI pun memiliki nilai minimum 0, padahal nilai BMI tidak bisa nol, sehingga harus dianggap sebagai nilai yang tidak valid.

Sementara itu, fitur DiabetesPedigreeFunction, Age, dan Outcome tidak memiliki nilai nol. Diabetes Pedigree Function memiliki rata-rata 0.47 dan maksimum 2.42, menunjukkan distribusi yang cenderung skewed. Usia pasien memiliki rentang antara 21 hingga 81 tahun, dengan rata-rata usia 33.24 tahun, mencerminkan bahwa dataset ini memang ditujukan untuk populasi dewasa. Terakhir, fitur Outcome memiliki nilai rata-rata sekitar 0.35, yang berarti sekitar 35% dari pasien dalam dataset ini terdiagnosis diabetes.

Secara keseluruhan, terlihat bahwa banyak fitur yang mengandung nilai nol yang secara medis tidak masuk akal, sehingga perlu dilakukan proses penanganan missing values (seperti imputasi dengan median) sebelum masuk ke tahap pelatihan model.

## Data Preparation
1. Feature Engineering
2. Train-Test-Split
3. Standarisasi

**Rubrik/Kriteria Tambahan (Opsional)**: 
1. Feature Engineering:
- BMI_Glucose_Ratio: Rasio antara BMI (Indeks Massa Tubuh) dan Glucose (Kadar glukosa darah).
Tujuan: Menggabungkan dua fitur medis penting yang berpotensi memiliki hubungan kompleks terhadap risiko diabetes. Rasio ini dapat menangkap interaksi antara obesitas dan kadar glukosa yang mungkin menjadi indikator kuat untuk prediksi.
Manfaat: Memberikan fitur turunan yang bisa mengungkap hubungan non-linier antara BMI dan Glucose terhadap Outcome.

- Age_Bins: Kategori umur yang dibagi dalam rentang 20–30, 31–40, dst.
Tujuan: Membantu model menangkap pola risiko berdasarkan kelompok usia. Fitur kategorikal ini dapat memperjelas tren yang tersembunyi jika hanya mengandalkan nilai numerik Age.
Manfaat: Mempermudah interpretasi dan bisa meningkatkan kinerja model yang sensitif terhadap variabel kategori.

2. Pemisahan Fitur dan Target
Dataset yang telah dibersihkan dipisahkan menjadi:
- X → seluruh fitur prediktor (independen)
- y → kolom Outcome sebagai target (label diabetes: 0 = tidak, 1 = ya)

3. Train-Test Split: Data dibagi menjadi data pelatihan dan pengujian menggunakan train_test_split() dengan parameter stratifikasi untuk menjaga distribusi kelas. 90% untuk pelatihan, 10% untuk pengujian. Tujuan dilakukannya untuk memastikan model dievaluasi pada data yang belum pernah dilihat (uji generalisasi model). Parameter random_state=123 digunakan untuk reproducibility (hasil tetap sama setiap kali dijalankan).

4. Standardisasi: Digunakan StandardScaler() dari Scikit-learn dalam pipeline sebelum modeling. Ini penting karena model seperti K-Nearest Neighbors (KNN) dan Support Vector Machine (SVM) sensitif terhadap skala fitur. 
- Digunakan StandardScaler() dari Scikit-learn untuk mengubah semua fitur numerik menjadi distribusi standar (mean = 0, std = 1).
- scaler.fit() dilakukan hanya pada data latih. 
- Transformasi kemudian diterapkan ke X_train dan X_test untuk mencegah data leakage.

Deskripsi statistik fitur setelah distandarisasi ditampilkan untuk memastikan distribusi sudah seragam. Tujuannyaa untuk mengecek bahwa setiap fitur memiliki nilai tengah mendekati 0 dan standar deviasi mendekati 1.

## Modeling
Membangun dan membandingkan beberapa model machine learning untuk memprediksi apakah seorang wanita suku Pima memiliki diabetes berdasarkan data kesehatan numerik.
- Algoritma yang Digunakan:
1. Logistic Regression
2. Random Forest (Default)
3. Support Vector Machine (SVM)
4. K-Nearest Neighbors (KNN) dengan tuning
5. Random Forest (Tuned via GridSearchCV)

Semua model dibangun dengan menggunakan Scikit-learn, dan preprocessing dilakukan menggunakan Pipeline untuk menjaga alur yang efisien.

- Tahapan Modeling & Parameter:
1. Logistic Regression
Model regresi logistik digunakan sebagai baseline karena sifatnya yang sederhana, cepat, dan mudah diinterpretasikan. Model baseline yang sederhana dan interpretatif dan tidak dilakukan tuning.
2. Random Forest (Default)
Model ensemble berbasis pohon keputusan. Menghasilkan prediksi berdasarkan voting dari banyak pohon. dan digunakan Parameter default. digunakan parameter random_state=42.
3. Support Vector Machine (SVM)
- Model klasifikasi margin maksimal.
- Diaktifkan probability=True agar dapat digunakan untuk ROC AUC.
4. K-Nearest Neighbors (Tuned)
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

**Rubrik/Kriteria Tambahan (Opsional)**: 
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

Feature Importance
Dari hasil Permutation Feature Importance terhadap model KNN Tuned, fitur paling penting dalam prediksi adalah:
- Glucose
- BMI
- Age Bins
- Skin Thickness
- BMI-Glucose Ratio

## Evaluation
**Rubrik/Kriteria Tambahan (Opsional) dan Kriteria Wajib**: 
Metrik Evaluasi yang digunakan yaitu:
1. Accuracy: (TP + TN) / Total
Persentase jumlah prediksi yang benar dari total prediksi yang dilakukan, Semakin tinggi nilai akurasi, semakin banyak data yang diklasifikasikan dengan benar. Namun, akurasi bisa menyesatkan pada dataset yang tidak seimbang (misalnya jika sebagian besar pasien ternyata tidak mengidap diabetes).
2. Precision: TP / (TP + FP)
Proporsi prediksi positif yang benar, Seberapa akurat model saat menyatakan seseorang mengidap diabetes. Precision tinggi berarti sedikit false positive.
3. Recall: TP / (TP + FN)
Seberapa banyak kasus sebenarnya positif (penderita diabetes) yang berhasil terdeteksi oleh model, Recall tinggi menunjukkan sedikit false negative, yaitu pasien yang sebenarnya mengidap diabetes tetapi tidak terdeteksi.
4. F1-score: Harmonic mean dari precision dan recall, F1-Score digunakan untuk menyeimbangkan antara precision dan recall, sangat berguna ketika kedua metrik tersebut sama pentingnya.
5. ROC-AUC: Kemampuan model untuk membedakan antara kelas. Semakin mendekati 1, semakin baik model membedakan kelas, AUC sangat penting dalam proyek ini karena menilai kemampuan model untuk membedakan antara penderita dan non-penderita diabetes secara menyeluruh.

Penjelasan Hasil Proyek Berdasarkan Metrik Evaluasi:
==== Logistic Regression ====
Accuracy : 0.7903225806451613
Precision: 0.7142857142857143
Recall   : 0.5263157894736842
F1-Score : 0.6060606060606061
ROC AUC: 0.83
Confusion Matrix:
 [[39  4]
 [ 9 10]]

==== Random Forest ====
Accuracy : 0.7580645161290323
Precision: 0.6
Recall   : 0.631578947368421
F1-Score : 0.6153846153846154
ROC AUC: 0.80
Confusion Matrix:
 [[35  8]
 [ 7 12]]

==== SVM ====
Accuracy : 0.7903225806451613
Precision: 0.7142857142857143
Recall   : 0.5263157894736842
F1-Score : 0.6060606060606061
ROC AUC: 0.80
Confusion Matrix:
 [[39  4]
 [ 9 10]]

==== KNN (Tuned) ====
Accuracy : 0.8387096774193549
Precision: 0.8461538461538461
Recall   : 0.5789473684210527
F1-Score : 0.6875
ROC AUC: 0.84
Confusion Matrix:
 [[41  2]
 [ 8 11]]

==== Random Forest (Tuned) ====
Accuracy : 0.7580645161290323
Precision: 0.6111111111111112
Recall   : 0.5789473684210527
F1-Score : 0.5945945945945946
ROC AUC: 0.81
Confusion Matrix:
 [[36  7]
 [ 8 11]]
 
 Analisis dan Interpretasi:
- KNN (Tuned) memiliki performa terbaik:
    - Accuracy dan F1-Score tertinggi, artinya model membuat prediksi benar dalam proporsi terbesar.
    - Precision tinggi: mayoritas prediksi positif (diabetes) memang benar.
    - ROC AUC tertinggi menunjukkan kemampuan memisahkan dua kelas secara menyeluruh.
- Random Forest (Tuned) memberikan keseimbangan yang cukup baik, tetapi tidak mampu mengungguli KNN dalam metrik utama.
- Logistic Regression dan SVM memberikan hasil yang cukup seimbang, tetapi Recall-nya rendah — artinya banyak kasus diabetes yang tidak terdeteksi.
- Recall pada semua model cenderung lebih rendah dari Precision, artinya model lebih cenderung melewatkan kasus positif daripada memprediksi positif yang salah.
 


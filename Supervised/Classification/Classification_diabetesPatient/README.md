# SVM (Support Vector Machine)

**Keunggulan**
1. Efektif on data berdimensi tinggi (data with many attribute or feature )

2. Efektif on case where number of fitur lebih besar than number of sample

3. Svm use subset poin pelatihan in decision function (disebut support vector) so make usage of memory be more efficient

**Goal**
- try model SVM untuk mengklasifikasikan apakah seorang pasien positif diabetes atau tidak.

**Tahapan**

1. Ubah data ke dalam Dataframe.
2. Bagi dataset.
3. Lakukan standarisasi.
4. Buat dan latih model.
5. Evaluasi model.

**Dataset** 
- Dataset berisi 8 kolom atribut dan 1 kolom label yang berisi 2 kelas yaitu 1 dan 0. Angka 1 menandakan bahwa orang tersebut positif diabetes dan 0 menandakan sebaliknya. Terdapat 768 sampel yang merupakan 768 pasien perempuan keturunan suku Indian Pima.
- [dataset](https://www.kaggle.com/uciml/pima-indians-diabetes-database)

**Code**
- make SVC (Support Vector Classifier) Object & implement to data training
  ```py
  from sklearn.svm import SVC
  # membuat objek SVC dan memanggil fungsi fit untuk melatih model
  clf = SVC()
  clf.fit(X_train, y_train)
  ```

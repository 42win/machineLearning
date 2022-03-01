# K-Means

**Tahapan**
1. Konversi data menjadi Dataframe.
2. Lakukan preprocessing data.
3. Hilangkan kolom 'CustomerID' dan 'gender'.
4. Latih model K-Means.
5. Buat plot untuk Elbow dan Cluster.

**Goals**
- mall costumer segmentation

**Dataset** : [link](https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python) 


code

- hitung nilai inersia (jumlah optimal cluster) using elbow method
```py
from sklearn.cluster import KMeans

#membuat list yang berisi inertia
clusters = []
for i in range(1,11):
  km = KMeans(n_clusters=i).fit(X)
  clusters.append(km.inertia_)
```
- visualisasi inertia
```py
   import matplotlib.pyplot as plt
    %matplotlib inline
    import seaborn as sns
     
    # membuat plot inertia
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(x=list(range(1, 11)), y=clusters, ax=ax)
    ax.set_title('Cari Elbow')
    ax.set_xlabel('Clusters')
    ax.set_ylabel('Inertia')
```

- make k-means model
```py
  # membuat objek KMeans
    km5 = KMeans(n_clusters=5).fit(X)
     
    # menambahkan kolom label pada dataset
    X['Labels'] = km5.labels_
     
    # membuat plot KMeans dengan 5 klaster
    plt.figure(figsize=(8,4))
    sns.scatterplot(X['annual_income'], X['spending_score'], hue=X['Labels'],
                    palette=sns.color_palette('hls', 5))
    plt.title('KMeans dengan 5 Cluster')
    plt.show()

```


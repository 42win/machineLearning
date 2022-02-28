# Linear Regression 

Model regresi linier adalah salah satu model machine learning yang paling sederhana. Model ini memiliki kompleksitas rendah dan bekerja sangat baik pada dataset yang memiliki hubungan linier. Jadi, ketika Anda menemui masalah yang terlihat memiliki hubungan linier, regresi linier dapat menjadi pilihan pertama sebagai model untuk dikembangkan.

**Goals**
- we will predict House Price based on number of room

**Tahapan**
1. import library
2. buat dummy dataset dgn Numpy Array
3. buat plot from model


- show data as scatter plot
    ```py
    import matplotlib.pyplot as plt
    %matplotlib inline
    
    plt.scatter(bedrooms, house_price)
    ```

- train model using Linear Regression
    ```py
    from sklearn.linear_model import LinearRegression
        
    # latih model dengan Linear Regression.fit()
    bedrooms = bedrooms.reshape(-1, 1)
    linreg = LinearRegression()
    linreg = linreg.fit(bedrooms, house_price) # (x,y)
    ```

- shwo plot relation between jumlahKamar - hargaRumah
  ```py
  plt.scatter(bedrooms, house_price) #titik
  plt.plot(bedrooms, linreg.predict(bedrooms)) #garis
  ```

- prediction test
  ```py
  print(linreg.predict([[3]])[0])
  ```

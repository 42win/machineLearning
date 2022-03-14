# Image Classification 
- classify scissors, paper, or rock

## Dataset
- [link](https://github.com/dicodingacademy/assets/releases/download/release/rockpaperscissors.zip.)

## Criteria
- Dataset harus dibagi menjadi train set dan validation set.
- Ukuran validation set harus 40% dari total dataset (data training memiliki 1314 sampel, dan data validasi sebanyak 874 sampel).
- Harus mengimplementasikan augmentasi gambar.
- Model harus menggunakan model sequential.
- Pelatihan model tidak melebihi waktu 30 menit.
- Program dikerjakan pada Google Colaboratory.
- Akurasi dari model minimal 85%.
- menggunakan lebih dari 1 hidden layer.

## Note
- Model merupakan klasifikasi multi kelas sehingga loss function yang digunakan bukan binary_crossentropy.


## Code
- make directory
  ```py
  import os

  train_roc = os.path.join(train_dir, 'rock')
  os.mkdir(train_roc)
  ```

- split
  ```py
  from sklearn.model_selection import train_test_split
  
  train_roc_dir, val_roc_dir = train_test_split(os.listdir(roc_dir), test_size = 0.40)
  ```

- copy file
  ```py
  import shutil
  for file in train_roc_dir:
  shutil.copy(os.path.join(roc_dir, file), os.path.join(train_roc, file))

  #shutil.copy( (path_source), (path_tujuan) )
  ```

- callback
  ```py
  # Penggunaan Callback mencegah overfitting dan menghentikan training setelah akurasi terpenuhi

    class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') > 0.97):
        print("\nAkurasi > 97%, training Stop!")
        self.model.stop_training = True

    callbacks = myCallback()

    #implement
    model.fit(
        ...,
        verbose,
            callbacks=[callbacks]
    )
  ```

- show accurary plot
  ```py
    from matplotlib import pyplot as plt
    
    #accuracy train & validation
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Plot')
    plt.ylabel('Value')
    plt.xlabel('Epoch')
    plt.legend(loc="lower right")
    plt.show()
  ```

- show loss plot
  ```py
    from matplotlib import pyplot as plt
    #loss train & validation
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Plot')
    plt.ylabel('Value')
    plt.xlabel('Epoch')
    plt.legend(loc="upper right")
    plt.show()
  ```

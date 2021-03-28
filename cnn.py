import keras
from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D
from keras.layers import Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
import os
import dataset

train_image_width = 25
train_image_height = 25
model_path = "C:\\Users\\Ahmet\\Desktop\\tez_verileri\\model\\cnn_model.h5"

images, labels = dataset.get_test_mini_data()
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.1)

model = Sequential()
# 32 tane 3x3 boyutunda filtreler eklenir
# Görüntünün boyutunu korumak için padding uygulanır
# İlk filtre uygulanırken sonraki Conv katmanları için mümkün oldukça çok bilgi korunmalıdır.
# Bu yüzden padding işlemi uygulanır
model.add(Conv2D(32, (3, 3),
                 padding='same',
                 input_shape=(25, 25, 1))
          )
# Eğer aktivasyon fonksiyonu kullanılmazsa;
# sinir ağı sadece doğrusal olan durumları öğrenir
# öğrenmesi sınırlı olur.
# Aktivasyon fonksiyonu kullanarak doğrusal olmayan durumlarda öğrenilir
# bu sayede sinir ağı verilerden anlamlı özellikleri de öğrenebilir
model.add(Activation('relu'))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))

# Pooling katmanı ile görüntünün kayma boyutu,
# ağ içindeki parametreler ve hesaplama sayısı azaltılır
model.add(MaxPooling2D(pool_size=(2, 2)))

# Rastgele olacak şekilde nöronların %25'ini kapatıldı
# eğitim sırasındaki ezberlemeyi önlemek için
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 2 boyutlu görüntüler 1 boyutlu vektöre çevrilir
model.add(Flatten())

# Önceki katmandaki tüm düğümler mevcut katmandaki düğümlere bağlanır
# 512 nöron modelimize eklenir
model.add(Dense(512))

model.add(Activation('relu'))
model.add(Dropout(0.5))

# honey combing, ground glass ve sağlıklı doku için; 3 sınıf eklenir
model.add(Dense(3))

# Sınıfların olasılıklarını hesaplamak için "Softmax" fonksiyonu eklenir:
model.add(Activation('softmax'))

# Modeli eğitirken kullanılacak optimizasyon fonksiyonu hazırlanır
opt = keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)

# Hata hesaplama için (sınıflandırma yapılacağından) "categorical_crossentropy" fonksiyonu kullanılır
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

#model.summary()

# Modelimize verilerimizi veriyoruz ve eğitimimizi başlatıyoruz:
model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test), shuffle=True)


# eğitilmiş model kaydedilir
#model.save(model_path)
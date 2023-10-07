#import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Przygotowanie danych (załóżmy, że masz dane w katalogach 'train' i 'validation')
train_datagen = ImageDataGenerator(
    # Tutaj dodaj operacje normalizacji i inne przetwarzanie wstępne
)

validation_datagen = ImageDataGenerator(
    # Tutaj dodaj operacje normalizacji i inne przetwarzanie wstępne
)

train_generator = train_datagen.flow_from_directory(
    'ścieżka_do_katalogu_z_danymi_treningowymi',
    target_size=(224, 224),  # Rozmiar docelowy obrazów
    batch_size=32,  # Rozmiar batcha
    class_mode='categorical'  # Tryb kategorialny dla wielu klas
)

validation_generator = validation_datagen.flow_from_directory(
    'ścieżka_do_katalogu_z_danymi_walidacyjnymi',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Tworzenie modelu CNN (to tylko przykład, dostosuj architekturę do swoich potrzeb)
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(n_classes, activation='softmax')  # n_classes - liczba klas
])

# Kompilacja modelu
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Trenowanie modelu
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=10,  # Określ liczbę epok treningowych
    validation_data=validation_generator,
    validation_steps=len(validation_generator)
)

# Ewaluacja modelu
model.evaluate(validation_generator)

# Dostosowanie modelu w zależności od wyników
# ...

# Zapisz model, aby móc go użyć w przyszłości
model.save('moj_model.h5')

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os

# -------------------------
# Paths & Parameters
# -------------------------
train_dir = "dataset/train"
test_dir = "dataset/test"
img_size = (224, 224)
batch_size = 32
epochs = 10  # Increase for better accuracy
learning_rate = 0.0001

# -------------------------
# Data Augmentation
# -------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# -------------------------
# Build Model (MobileNetV2 Transfer Learning)
# -------------------------
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(train_gen.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=preds)

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# -------------------------
# Train Model
# -------------------------
history = model.fit(
    train_gen,
    steps_per_epoch=train_gen.samples//batch_size,
    validation_data=test_gen,
    validation_steps=test_gen.samples//batch_size,
    epochs=epochs
)

# -------------------------
# Save Model
# -------------------------
os.makedirs("models", exist_ok=True)
model.save("models/crop_disease_model.h5")
print("✅ Crop disease model trained and saved!")


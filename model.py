import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Build the CNN model with parameters close to 25,000
model = Sequential([
    Conv2D(8, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),  # Reduced filters
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(16, kernel_size=(3, 3), activation='relu'),  # Reduced filters
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(48, activation='relu'),  # Reduced dense units
    Dropout(0.25),  # Added dropout for regularization
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model for one epoch
model.fit(x_train, y_train, epochs=1, batch_size=128, verbose=1)

# Evaluate the model
train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)
print(f'Training accuracy: {train_acc * 100:.2f}%')
print(f'Total number of parameters: {model.count_params()}')

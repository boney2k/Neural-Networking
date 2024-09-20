import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.datasets import mnist

# Load and preprocess the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28*28).astype('float32') / 255
x_test = x_test.reshape(-1, 28*28).astype('float32') / 255

# Build the model
model = Sequential([
    Input(shape=(28*28,)),  # Use Input layer as the first layer
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train,
                    epochs=5,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=1)  # Set verbose to 1 to show training progress

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test, verbose=1)

# Print the results
print(f"Test loss: {loss}")
print(f"Test accuracy: {accuracy}")

# Optionally: Print the training history for more details
print("\nTraining History:")
for epoch in range(5):
    print(f"Epoch {epoch+1}:")
    print(f"  Training loss: {history.history['loss'][epoch]}")
    print(f"  Training accuracy: {history.history['accuracy'][epoch]}")
    print(f"  Validation loss: {history.history['val_loss'][epoch]}")
    print(f"  Validation accuracy: {history.history['val_accuracy'][epoch]}")

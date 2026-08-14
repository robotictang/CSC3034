# Copyright Author: Dr Tang Tiong Yew
"""
Lab 8b: Artificial Neural Networks using Keras (Text Classification)
====================================================================
This script provides the complete solution for Lab 8b exercises, including:
1. IMDb movie review dataset loading & text decoding
2. Sequence padding with `pad_sequences`
3. Keras LSTM model architecture (Embedding + LSTM + Dropout + Dense)
4. Model compilation & training with validation split
5. Test accuracy evaluation & loss/accuracy training curve plots
6. Sentiment prediction on new sample text reviews

Execution Mode:
`python3 src/files/lab8b_keras_lstm.py`
"""

import matplotlib.pyplot as plt

try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
    from tensorflow.keras.datasets import imdb
    HAS_TF = True
except ImportError:
    HAS_TF = False


def main():
    print("=========================================================")
    print(" Lab 8b: Text Classification Using Keras (IMDb LSTM)    ")
    print("=========================================================")

    if not HAS_TF:
        print("[!] TensorFlow / Keras not found. Install via: pip install tensorflow")
        return

    # 1. Load and Explore Dataset
    vocab_size = 10000
    max_length = 100

    print("\n[1] Loading IMDb movie reviews dataset (top 10,000 words)...")
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

    print(f"    Train reviews: {len(x_train)} | Test reviews: {len(x_test)}")
    print(f"    First review (encoded integer indices): {x_train[0][:15]}...")
    print(f"    First review label (0=negative, 1=positive): {y_train[0]}")

    # Decode first review back to text
    word_index = imdb.get_word_index()
    reverse_word_index = {v: k for k, v in word_index.items()}

    def decode_review(encoded_review):
        return " ".join([reverse_word_index.get(i - 3, "?") for i in encoded_review])

    decoded = decode_review(x_train[0])
    print(f"\n    Decoded review text preview:\n    '{decoded[:120]}...'")

    # 2. Preprocess & Pad Sequences
    print("\n[2] Preprocessing and padding sequences to fixed length = 100...")
    raw_test_review = x_test[0]
    x_train = pad_sequences(x_train, maxlen=max_length, padding='pre', truncating='pre')
    x_test = pad_sequences(x_test, maxlen=max_length, padding='pre', truncating='pre')

    # 3. Build Keras Embedding + LSTM Model
    print("\n[3] Building Sequential Keras model (Embedding -> LSTM -> Dropout -> Dense)...")
    model = Sequential([
        tf.keras.layers.Input(shape=(max_length,)),
        Embedding(input_dim=vocab_size, output_dim=32, mask_zero=True),
        LSTM(64, return_sequences=False),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')  # Binary sentiment output: 0=Negative, 1=Positive
    ])

    model.summary()

    # 4. Compile Model
    print("\n[4] Compiling Keras LSTM model...")
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # 5. Train Model
    print("\n[5] Training LSTM model for 5 epochs (validation_split=0.2)...")
    history = model.fit(
        x_train, y_train,
        validation_split=0.2,
        epochs=5,
        batch_size=32,
        verbose=1
    )

    # 6. Evaluate Model
    print("\n[6] Evaluating model on test set...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
    print(f"\n---> Test Accuracy: {test_accuracy * 100:.2f}% | Test Loss: {test_loss:.4f}")

    # Plot Accuracy and Loss Curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 7. Make Sentiment Predictions
    print("\n[7] Testing sentiment prediction on sample test review...")
    sample_review = x_test[:1]
    prediction = model.predict(sample_review)
    sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
    true_sentiment = "Positive" if y_test[0] == 1 else "Negative"

    print(f"    Sample Review Text: '{decode_review(raw_test_review)[:120]}...'")
    print(f"    Predicted Sentiment Score: {prediction[0][0]:.4f} ({sentiment})")
    print(f"    Ground Truth Sentiment: {true_sentiment}")

    print("\n[SUCCESS] Lab 8b Keras LSTM pipeline completed successfully.")


if __name__ == '__main__':
    main()

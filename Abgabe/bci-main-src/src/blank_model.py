import tensorflow as tf

def create_model(
    n_channels: int = 16,
    n_times: int = 250,
    n_classes: int = 2,
):
    """
    Erstellt simples Modell für EEG Daten.
    :param n_channels: Anzahl der Kanäle
    :param n_times: Anzahl der Zeitpunkte
    :param n_classes: Anzahl der Klassen
    """
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(n_channels, n_times)),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        
        tf.keras.layers.Dense(n_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    # Beispielaufruf
    model = create_model()
    model.summary()
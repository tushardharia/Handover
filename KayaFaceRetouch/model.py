import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os

# Encoder
def build_encoder(latent_dim):
    inputs = keras.Input(shape=(256, 256, 3))  # assuming 256x256 RGB images
    x = layers.Conv2D(64, (4, 4), strides=2, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(128, (4, 4), strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2D(256, (4, 4), strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2D(512, (4, 4), strides=2, padding='same', activation='relu')(x)
    latent = layers.Flatten()(x)
    latent = layers.Dense(latent_dim)(latent)  # Latent space with specific dimension
    
    return keras.Model(inputs, latent, name="encoder")

# Decoder
def build_decoder(latent_dim):
    inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(32 * 32 * 512, activation='relu')(inputs)
    x = layers.Reshape((32, 32, 512))(x)
    x = layers.Conv2DTranspose(256, (4, 4), strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(128, (4, 4), strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(64, (4, 4), strides=2, padding='same', activation='relu')(x)
    outputs = layers.Conv2DTranspose(3, (4, 4), strides=2, padding='same', activation='sigmoid')(x)  # Output RGB image
    
    return keras.Model(inputs, outputs, name="decoder")

# Attention Module
def build_attention_module(input_shape):
    inputs = keras.Input(shape=input_shape)
    alpha = layers.Conv2D(1, (1, 1), activation='sigmoid')(inputs)  # Attention weights
    outputs = layers.Multiply()([inputs, alpha])
    return keras.Model(inputs, outputs, name="attention_module")

# Generator
def build_generator(latent_dim):
    latent_inputs = keras.Input(shape=(latent_dim,))
    deco_inputs = keras.Input(shape=(512, 512, 3))  # Assuming decoder intermediate outputs
    
    # Ensure the latent dense output is reshaped to match the desired spatial dimensions
    x = layers.Dense(512 * 512, activation='relu')(latent_inputs)  # Larger dense layer
    x = layers.Reshape((512, 512, 1))(x)  # Reshape to a spatial structure
    
    # Intermediate features from decoder and generator's latent input
    x = layers.Concatenate()([deco_inputs, x])  # Combine with features from decoder
    
    x = layers.Conv2D(256, (4, 4), strides=1, padding='same', activation='relu')(x)
    x = layers.Conv2D(128, (4, 4), strides=1, padding='same', activation='relu')(x)
    x = layers.Conv2D(64, (4, 4), strides=1, padding='same', activation='relu')(x)
    outputs = layers.Conv2D(3, (4, 4), strides=1, padding='same', activation='sigmoid')(x)
    
    return keras.Model([latent_inputs, deco_inputs], outputs, name="generator")

# Build Models
latent_dim = 128
encoder = build_encoder(latent_dim)
decoder = build_decoder(latent_dim)
attention_1 = build_attention_module((512, 512, 3))  # Example feature map
attention_1.summary()
generator = build_generator(latent_dim)

# Inference Models
input_img = keras.Input(shape=(256, 256, 3))

# Stage 1: Coarse retouching
latent = encoder(input_img)
decoded = decoder(latent)

# Attention on decoder outputs
attention_decoded = attention_1(decoded)

# Stage 2: Generator for additional realism
generated = generator([latent, attention_decoded])

# Output the retouched image
face_retoucher = keras.Model(input_img, generated, name="face_retoucher")

# Display the model architecture
face_retoucher.summary()

#################################################################################
# Build the Discriminator
def build_discriminator():
    inputs = keras.Input(shape=(256, 256, 3))  # RGB image
    x = tf.keras.layers.Conv2D(64, (4, 4), strides=2, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.Conv2D(128, (4, 4), strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(256, (4, 4), strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(512, (4, 4), strides=2, padding='same', activation='relu')(x)

    latent = layers.Flatten()(x)
    outputs = layers.Dense(latent_dim)(latent)  # Latent space with specific dimension
    return tf.keras.Model(inputs, outputs, name="discriminator")

discriminator = build_discriminator()

# Define loss functions
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(fake_output):
    # The generator's goal is to "fool" the discriminator, so it should be classified as real
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    # The discriminator's goal is to distinguish real from fake
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

# Optimizers for generator and discriminator
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Define the training step
@tf.function  # Compiles the function for optimized execution
def train_step(images):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Stage 1: Encoder and decoder for coarse retouching
        latent = encoder(images)
        decoded = decoder(latent)

        # Ensure that the generated images have the correct shape
        generated_images = generator([latent, decoded])
        
        # Stage 2: Attention and generator for further refinement
        attention_decoded = attention_1(decoded)  # Apply attention module
        generated_images = generator([latent, attention_decoded])  # Generate new images
        generated_images = tf.image.resize(generated_images, size=(256, 256))
        
        # Discriminator's outputs
        real_output = discriminator(images)  # Discriminator's prediction on real images
        fake_output = discriminator(generated_images)  # Discriminator's prediction on generated images
        
        # Calculate losses
        gen_loss = generator_loss(fake_output)  # Generator's loss
        disc_loss = discriminator_loss(real_output, fake_output)  # Discriminator's loss
        
        # Compute gradients
        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)  # Generator gradients
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)  # Discriminator gradients
        
        # Apply gradients
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    return gen_loss, disc_loss  # Return generator and discriminator losses

# Load Dataset
def load_local_dataset(folder_path='dataset/', batch_size=32, shuffle_buffer=1000):
    # Get a list of all PNG file paths in the folder
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".png")]

    # Define a function to read and preprocess an image
    def preprocess_image(image_path):
        image = tf.io.read_file(image_path)  # Read the image file
        image = tf.image.decode_png(image, channels=3)  # Decode the PNG image (assuming 3 channels for RGB)
        image = tf.image.resize(image, (256, 256))  # Resize to 256x256
        image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
        print(f"Image data type: {image.dtype}")
        return image, image  # Output and target for unsupervised training

    # Create a TensorFlow dataset from the image paths and apply preprocessing
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    # Shuffle, batch, and prefetch the dataset for training
    dataset = dataset.shuffle(shuffle_buffer).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset

# Dataset 
dataset = load_local_dataset()

# Basic training loop
def train_gan(dataset, epochs=10):
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for batch, (images, targets) in enumerate(dataset):
            generated_images = train_step(images)
            print(generated_images)
            gen_loss, disc_loss = train_step(images)
            if batch % 10 == 0:
                print(f"Batch {batch}: Generator loss = {gen_loss.numpy()}, Discriminator loss = {disc_loss.numpy()}")

# Start Training
train_gan(dataset)
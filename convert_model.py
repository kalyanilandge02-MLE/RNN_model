import tensorflow as tf

# Convert .h5 model to SavedModel format
h5_model_path = "next_word_lstm.h5"
saved_model_dir = "saved_model"

# Load the .h5 model
model = tf.keras.models.load_model(h5_model_path)

# Save the model in SavedModel format
model.save(saved_model_dir, save_format="tf")

print(f"Model converted and saved to {saved_model_dir}")
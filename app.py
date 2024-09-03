import tensorflow.keras as keras
import numpy as np
import sqlite3
import pandas as pd

# Load the Fashion MNIST dataset
(xtrain, ytrain), (xtest, ytest) = keras.datasets.fashion_mnist.load_data()

# Print the shapes of the dataset
print(xtrain.shape)  # Should be (60000, 28, 28)
print(ytrain.shape)  # Should be (60000,)
print(xtest.shape)   # Should be (10000, 28, 28)
print(ytest.shape)   # Should be (10000,)

# Normalize the pixel values to the range [0, 1]
xtrain = xtrain.astype('float32') / 255
xtest = xtest.astype('float32') / 255

# Reshape the data to include a channel dimension
xtrain = np.reshape(xtrain, (xtrain.shape[0], 28, 28, 1))
xtest = np.reshape(xtest, (xtest.shape[0], 28, 28, 1))

# Print the new shapes of the dataset
print(xtrain.shape)  # Should be (60000, 28, 28, 1)
print(ytrain.shape)  # Should be (60000,)
print(xtest.shape)   # Should be (10000, 28, 28, 1)
print(ytest.shape)   # Should be (10000,)

# Connect to SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect('fashion_mnist.db')

# Create a table for storing images and labels
conn.execute('''CREATE TABLE IF NOT EXISTS images
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
             image BLOB NOT NULL,
             label INTEGER NOT NULL);''')

# Insert training data into the database
for i in range(xtrain.shape[0]):
    conn.execute('INSERT INTO images (image, label) VALUES (?, ?)',
                 [sqlite3.Binary(xtrain[i].tobytes()), ytrain[i]])

# Commit the transaction
conn.commit()

# Insert test data into the database
for i in range(xtest.shape[0]):
    conn.execute('INSERT INTO images (image, label) VALUES (?, ?)',
                 [sqlite3.Binary(xtest[i].tobytes()), ytest[i]])

# Commit the transaction
conn.commit()

# Now retrieve the data and load it into a Pandas DataFrame
data = pd.read_sql_query('SELECT * FROM images', conn)

# Close the connection
conn.close()

# Print the first few rows of the DataFrame
print(data.head())

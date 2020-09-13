import numpy as np
import tensorflow as tf

alpha1, mach1 = input("Enter the first set of alpha and mach values: ").split(', ')
alpha2, mach2 = input("Enter the second set of alpha and mach values: ").split(', ')
examples = np.array([[alpha1, mach1], [alpha2, mach2]], dtype=float)

model = tf.keras.models.load_model('model.h5')
print(model.predict(examples))

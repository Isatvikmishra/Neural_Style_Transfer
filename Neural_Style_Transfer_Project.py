import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt

def load_image(image_path, max_dim=512):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

content_path = tf.keras.utils.get_file('human.jpg', 'https://upload.wikimedia.org/wikipedia/commons/7/7f/Emma_Watson_2013.jpg')
style_path = tf.keras.utils.get_file('styling_char.jpg', 'https://upload.wikimedia.org/wikipedia/commons/thumb/1/16/Sapeck-La_Joconde_fumant_la_pipe.jpg/599px-Sapeck-La_Joconde_fumant_la_pipe.jpg')


content_image = load_image(content_path)
style_image = load_image(style_path)

imshow(content_image, 'Content Image')
imshow(style_image, 'Style Image')

hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]

output_image = tensor_to_image(stylized_image)
output_image.save("stylized_output.jpg")
imshow(stylized_image, "Stylized Image")

import tensorflow as tf
import tensorflow_hub as hub
import PIL.Image as Image
import numpy as npI
from scipy.spatial.distance import euclidean, cosine


def img_classification(f1):
    m = tf.keras.Sequential([
        hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4", output_shape=[1001])
    ])
    m.build([None, 224, 224, 3])

    IMAGE_SHAPE = (224, 224)
    img = Image.open(f1).convert('RGB').resize(IMAGE_SHAPE)
    img = np.array(img) / 255.0

    result = m.predict(img[np.newaxis, ...])
    predicted_class = np.argmax(result[0], axis=-1)

    labels_path = tf.keras.utils.get_file('ImageNetLabels.txt',
                                          'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
    imagenet_labels = np.array(open(labels_path).read().splitlines())

    predicted_class_name = imagenet_labels[predicted_class]

    return predicted_class_name


def img_similarity(f1, f2):
    model = tf.keras.Sequential([
        hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v1_075_192/quantops/feature_vector/3"),
    ])

    IMAGE_SHAPE = (192, 192)

    img1 = np.array(Image.open(f1).convert('RGB').resize(IMAGE_SHAPE)) / 255.0
    img2 = np.array(Image.open(f2).convert('RGB').resize(IMAGE_SHAPE)) / 255.0

    inputs = tf.convert_to_tensor([img1, img2])

    print (inputs.shape)
    features = model.predict(inputs)  # Features with shape [batch_size, num_features].

    print(features.shape)

    euclidean_distance = euclidean(features[0], features[1])
    cosine_distance = cosine(features[0], features[1])
    return euclidean_distance,cosine_distance

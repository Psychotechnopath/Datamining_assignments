from sklearn.model_selection import cross_validate, train_test_split
import openml as oml
from matplotlib import pyplot as plt
import numpy as np


SVHN = oml.datasets.get_dataset(41081)                                   #Load data
X, y, cats, attrs = SVHN.get_data(dataset_format='array',
    target=SVHN.default_target_attribute)

X_90_percent, X_10_percent, y_90_percent, y_10_percent = train_test_split(X, y, test_size=0.1, stratify=y, random_state=47)


def rgb2gray(X_data, dim=32):
    return np.squeeze(np.expand_dims(np.dot(X_data.reshape(len(X_data), dim * dim, 3), [0.2990, 0.5870, 0.1140]), axis=3))

X_gray = rgb2gray(X)

def std_function(X_data):
    std_array = np.zeros(X_gray.shape)
    for counter, value in enumerate(X_10_percent):
        std_array[counter] = (X_data[counter] - np.mean(X_data[counter]))/ np.std(X_data[counter])
    return std_array

X_gray_std = std_function(X_gray)
np.save('Standardized grayscale array', X_gray_std)

# Plots image. Use grayscale=True for plotting grayscale images
def plot_images(X, y, grayscale=False):
    fig, axes = plt.subplots(1, len(X),  figsize=(10, 5))
    if grayscale:
        [ax.imshow(X[n].reshape(32, 32)/255, cmap='gray')
         for n,ax in enumerate(axes)]
    else:
        [ax.imshow(X[n].reshape(32, 32, 3)/255) for n,ax in enumerate(axes)]
    [ax.set_title((y[n]+1)) for n,ax in enumerate(axes)]
    [ax.axis('off') for ax in axes]
    plt.show()


plot_images(X[0:5], y[0:5])
plot_images(X_gray[0:5], y[0:5], grayscale=True)
plot_images(X_gray_std[0:5], y[0:5], grayscale=True)



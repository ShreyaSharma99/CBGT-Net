import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import seaborn as sns
from sklearn.metrics import confusion_matrix
from keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping


log_run = False
pad_images = True
lenet5 = True

def pad_patches(img):
    # Desired output shape
    desired_shape = (28, 28, 3)

    if desired_shape[0] == img.shape[0]:
        return img

    # Calculate the amount of padding for each dimension
    pad_height = desired_shape[0] - img.shape[0]
    pad_width = desired_shape[1] - img.shape[1]

    top_pad = pad_height // 2
    bottom_pad = pad_height - top_pad
    left_pad = pad_width // 2
    right_pad = pad_width - left_pad
    padded_image = cv2.copyMakeBorder(img, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return padded_image

# (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

train_sz, test_sz = 5421, 892
# train_sz, test_sz = 500, 100


image_sz = [28, 28, 3]
# all_patch = True


data_indx_train, data_indx_test  = [], []
for i in range(10):
    data_indx_train.append(np.random.choice(np.where(train_labels == i)[0], train_sz, replace=False))
    data_indx_test.append(np.random.choice(np.where(test_labels == i)[0], test_sz, replace=False))

data_indx_train = np.array(data_indx_train)
data_indx_test = np.array(data_indx_test)

data_train = np.zeros((10, train_sz, image_sz[0], image_sz[1], 3))
y_train = np.zeros((10, train_sz))
data_test = np.zeros((10, test_sz, image_sz[0], image_sz[1], 3))
y_test = np.zeros((10, test_sz))

print("Data loaded!")

for i in range(10):
    for j in range(train_sz):
        data_train[i, j, :, :, :] =  np.repeat(train_images[data_indx_train[i, j],:, :, np.newaxis], repeats=3, axis=2)
        y_train[i, j] = i
    for j in range(test_sz):
        data_test[i, j, :, :, :] = np.repeat(test_images[data_indx_test[i, j],:, :, np.newaxis], repeats=3, axis=2)
        y_test[i, j] = i

# data_train = data_train.numpy()
# data_test = data_test.numpy()

patch_sz_list = [[28, 28, 3], [20, 20, 3], [16, 16, 3], [12, 12, 3], [10, 10, 3], [8, 8, 3], [5, 5, 3]]
x_i_list = [[0], [0, 8], [0, 6, 12], [0, 6, 12, 16], [0, 6, 12, 18], [0, 4, 8, 12, 16, 20], [0, 5, 10, 15, 20, 23]]
# y_i_list = [[0], [0, 11], [0, 7, 15], [0, 5, 11, 19], [0, 3, 7, 11, 15, 19, 23], [0, 3, 6, 9, 12, 15, 18, 21, 24, 26]]

if log_run: 
    out_file = open('/home/shreya/cbgt_net/baselines/supervised_mnist_lenet5/out_2048batch_0.01lr_lenet5_5patch.txt', 'w')

# for i in range(len(patch_sz_list)-2, -1, -1):
for i in [-1]:
    patch_sz = patch_sz_list[i]
    if log_run: out_file.write(f'For patch_size = {str(patch_sz[0])} \n') 
    x_i, y_i = x_i_list[i], x_i_list[i]
    patch_dataset = []
    patch_label = []
    for i in range(data_train.shape[0]):
        for j in range(data_train.shape[1]):
            for x in x_i:
            # range(image_sz[0]+1-patch_sz[0]):
                for y in y_i:
                    if pad_images:
                        patch = pad_patches(data_train[i, j, x:x+patch_sz[0], y:y+patch_sz[1], :])
                    else:
                        patch = data_train[i, j, x:x+patch_sz[0], y:y+patch_sz[1], :]
                    patch_dataset.append(patch)
                    patch_label.append(i)

    test_data = []
    test_label = []
    for i in range(data_test.shape[0]):
        for j in range(data_test.shape[1]):
            for x in x_i:
                for y in y_i:
                    if pad_images:
                        patch = pad_patches(data_test[i, j, x:x+patch_sz[0], y:y+patch_sz[1], :])
                    else:
                        patch = data_test[i, j, x:x+patch_sz[0], y:y+patch_sz[1], :]
                    test_data.append(patch)
                    test_label.append(i)

    patch_dataset = np.array(patch_dataset)
    patch_label = np.array(patch_label)
    idx = np.arange(patch_dataset.shape[0])
    random.shuffle(idx)

    train_sz = int(patch_dataset.shape[0])
    print("Train size = ", train_sz)
    print("Train label size = ", patch_label.shape)
    patch_dataset = patch_dataset[idx, ...]
    patch_label = patch_label[idx, ...]

    # import cv2
    # import numpy as np
    # image_array = (patch_dataset[2000, ...]* 255).astype(np.uint8)
    # print("PATCH LABEL  =", patch_label[2000])
    # file_name = "/home/shreya/cbgt_net/baselines/supervised_mnist_lenet5/output_image.png"
    # cv2.imwrite(file_name, image_array)
    # print(f"Image saved as {file_name}")

    patch_dataset = tf.convert_to_tensor(patch_dataset)
    patch_label = tf.convert_to_tensor(patch_label)

    print("patch_dataset = ", patch_dataset.shape)
    print("patch_label = ", patch_label.shape)

    batch_size = 2048
    t_dataset = tf.data.Dataset.from_tensor_slices((patch_dataset[:train_sz], patch_label[:train_sz]))
    train_dataset = t_dataset.batch(batch_size)

    test_sz = int(len(test_data))
    print("Test size = ", test_sz)
    test_data = tf.convert_to_tensor(np.array(test_data))
    test_label = tf.convert_to_tensor(np.array(test_label))
    e_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_label))
    test_dataset = e_dataset.batch(batch_size)

    # print("Data loaded!")


    model = models.Sequential()

    if lenet5:
        if pad_images:
            model.add(layers.Conv2D(6, (5, 5), activation='relu', input_shape=(image_sz[0], image_sz[1], 3)))
        else:
            model.add(layers.Conv2D(6, (5, 5), activation='relu', input_shape=(patch_sz[0], patch_sz[1], 3)))

        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(16, (5, 5), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        # model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(120, activation='relu'))
        model.add(layers.Dense(84, activation='relu'))
        model.add(layers.Dense(10))

    else:
        if pad_images:
            model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(image_sz[0], image_sz[1], 3)))
        else:
            model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(patch_sz[0], patch_sz[1], 3)))

        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(10))
    
    model.compile(
                optimizer=tf.keras.optimizers.Adam(0.01),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
                )
    
    # print(model.summary())

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


    history_cnn = model.fit(
        train_dataset,
        epochs=200,
        validation_data=test_dataset,
        batch_size=batch_size,
        callbacks=[early_stopping]
    )

    # summarize history for accuracy
    plt.plot(history_cnn.history['sparse_categorical_accuracy'])
    plt.plot(history_cnn.history['val_sparse_categorical_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    model_name = "_LeNet5" if lenet5 else ""
    if log_run:
        file_name = "/home/shreya/cbgt_net/baselines/supervised_mnist_lenet5/acc_patch" + str(patch_sz[0]) + "_dataPadded" + model_name
        plt.savefig(file_name)
    plt.clf()
    # summarize history for loss
    plt.plot(history_cnn.history['loss'])
    plt.plot(history_cnn.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    if log_run:
        file_name = "/home/shreya/cbgt_net/baselines/supervised_mnist_lenet5/loss_patch" + str(patch_sz[0]) + "_dataPadded" + model_name
        plt.savefig(file_name)
    plt.clf()

    test_results = model.evaluate(test_dataset, verbose=False)
    print(f'Test results - Loss: {test_results[0]} - Accuracy: {100*test_results[1]}%')
    if log_run: out_file.write(f'Test results - Loss: {test_results[0]} - Accuracy: {100*test_results[1]}% \n \n')

    predictions = model.predict(test_dataset)
    pred = tf.cast(tf.argmax(predictions, axis=-1), dtype=tf.int64)
    confusion_matrix_values = confusion_matrix(test_label, pred)

    # Define class labels
    class_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    # Create a heatmap for the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(confusion_matrix_values, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    if log_run:
        file_name = "/home/shreya/cbgt_net/baselines/supervised_mnist_lenet5/cm_patch" + str(patch_sz[0]) + "_dataPadded" + model_name
        plt.savefig(file_name)
    plt.clf()

    test_results = model.evaluate(test_dataset, verbose=False)
    print(f'Test results - Loss: {test_results[0]} - Accuracy: {100*test_results[1]}%')
    if log_run: out_file.write(f'Test results - Loss: {test_results[0]} - Accuracy: {100*test_results[1]}% \n \n')

if log_run: out_file.close()
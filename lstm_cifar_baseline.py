from math import fabs
import os
os.environ["CUDA_VISIBLE_DEVICES"]= "0"
import yappi

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers.legacy import Adam
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
from cbgt_net.environments import CIFAR_CategoricalEnvironment
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from baselines.resnet import *
import time


start_time = time.time()

tf.random.set_seed(
    21
    # old = 10
)

log_results = False
padded_patches = True
resume_ckpt = False


def get_CNN_Encoder(input_shape):
    encoder = models.Sequential()
    encoder.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    # encoder.add(layers.MaxPooling2D((2, 2)))
    encoder.add(layers.Conv2D(64, (3, 3), activation='relu'))
    encoder.add(layers.MaxPooling2D((2, 2)))
    encoder.add(layers.Conv2D(64, (3, 3), activation='relu'))
    encoder.add(layers.Flatten())
    return encoder

def get_LeNet5_Encoder(input_shape):
    encoder = models.Sequential()
    encoder.add(layers.Conv2D(6, (5, 5), activation='relu', input_shape=input_shape))
    encoder.add(layers.MaxPooling2D((2, 2)))
    encoder.add(layers.Conv2D(16, (5, 5), activation='relu'))
    encoder.add(layers.MaxPooling2D((2, 2)))
    encoder.add(layers.Flatten())
    encoder.add(layers.Dense(120, activation='relu'))
    encoder.add(layers.Dense(84, activation='relu'))
    # tf.keras.layers.Dense(self._num_categories, activation='softmax')
    return encoder

class LSTMModel(tf.keras.Model):
    def __init__(self, seq_length, patch_size, num_classes):
        super(LSTMModel, self).__init__()
        resnet_encoder = ResNet('resnet18', num_classes)
        
        # # then create our final model
        self.lstm_model = tf.keras.Sequential([tf.keras.layers.TimeDistributed(resnet_encoder, input_shape=(seq_length, patch_size[0],  patch_size[1],  patch_size[2])),
                                    tf.keras.layers.LSTM(10, kernel_initializer='he_normal'),
                                    tf.keras.layers.Dense(num_classes, activation = 'softmax')])
        
    def call(self, inputs, states=None):
        logits = self.lstm_model(inputs)
        return logits

def train_step(inputs, labels, model, optimizer, loss_fn, accuracy, tf_step):
    # nonlocal tf_step
    with tf.GradientTape() as tape:
        logits = model(inputs)
        # l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables])
        loss = loss_fn(labels, logits) #+ l2_loss * 5e-4

    gradients = tape.gradient(loss, model.trainable_variables)
    clipped_grads = [tf.clip_by_value(grad, clip_value_min=-1.0, clip_value_max=1.0) for grad in gradients]

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    accuracy.update_state(labels, logits)

    return loss, accuracy.result(), tf_step

def eval_step(inputs, labels, model, loss_fn, accuracy):
    logits = model(inputs)
    loss = loss_fn(labels, logits)
    accuracy.update_state(labels, logits)
    return loss, accuracy.result()


batch_size = 512
# seq_length = [[1], [1,2,3,4,5,6] , [1,2,3,4,5,6] , [1,2,3,4,5,6,7,8], [1,2,3,4,5,6,7,8,9,10,11], [1,2,3,4,5,6,7,8,9,10,12,15], [1,2,3,4,5,6,9,15,16,21,28] ]
seq_length = [[1], [2, 3, 4, 5, 7], [2, 4, 5, 6, 8], [2, 3, 5, 8, 10], [2, 3, 6, 8, 11], [2, 5, 7, 10, 14], [2, 5, 9, 14, 17]]

num_classes = 10
image_sz = [32, 32, 3]
patch_size_list = [[32, 32, 3], [20, 20, 3], [16, 16, 3], [12, 12, 3], [10, 10, 3], [8, 8, 3], [5, 5, 3]]
noise = 0
number_of_epochs = 200000
eval_freq = 100

# training = False
last_x_losses = 1000

if log_results: out_file = open('./lstm_cifar/runs_seed20/out_3layer_ResNet18_padded_8_LSTM_300k_new', 'a')

for i in range(len(patch_size_list)):
    patch_size = patch_size_list[i]
    for s_len in seq_length[i]:
        if padded_patches:
            model = LSTMModel(s_len, image_sz, num_classes)
        else:
            model = LSTMModel(s_len, patch_size, num_classes)

        initial_learning_rate = 0.005
        lr_schedule = ExponentialDecay(
            initial_learning_rate, decay_steps=number_of_epochs, decay_rate=0.9, staircase=True
        )
        decay_steps = int(number_of_epochs*100)
        learning_rate_fn = tf.keras.experimental.CosineDecay(0.1, decay_steps=decay_steps)
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9)

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        shape_env = CIFAR_CategoricalEnvironment(10, noise=0.0, batch_size=batch_size, image_shape= image_sz,
                patch_size= patch_size,
                images_per_class_train= 5000,
                images_per_class_test= 1000,
                max_steps_per_episode= s_len)

        oneHotWrapper = shape_env.OneHotWrapper(shape_env)

        print("Patch size = ", patch_size, " s_len = ", str(s_len))
        if log_results: out_file.write(f'For patch_size = {str(patch_size[0])} and s_len = {str(s_len)} \n')
        loss_list, acc_list , acc_list_test, loss_list_test = [], [], [], []
        best_test_acc = 0
        tf_step=0 
        best_acc, curr_epoch = tf.Variable(0.0), tf.Variable(0.0)
        ckpt_path = "./lstm_cifar/runs_seed20/checkpoints/" + "ResNet18_padded_"+ str(patch_size[0]) +"patch_" +  str(int(s_len)) + "_seqLen_LSTM_300k"
        
        ckpt = tf.train.Checkpoint(curr_epoch=curr_epoch, best_acc=best_acc, optimizer=optimizer, model=model)
        manager = tf.train.CheckpointManager(ckpt, ckpt_path, max_to_keep=1)

        if resume_ckpt:
            # Load checkpoint.
            print('==> Resuming from checkpoint...')
            assert os.path.isdir(ckpt_path), 'Error: no checkpoint directory found!'

            # Restore the weights
            ckpt.restore(manager.latest_checkpoint)

        for epoch in range(number_of_epochs):
            observation = []
            shape_env.reset(training=True)
            oneHotWrapper.reset(training=True)
            y_train = oneHotWrapper.target_index

            
            for t in range(s_len):
                observation.append(oneHotWrapper.observe(training=True, time_step=t))
            observation =  tf.transpose(tf.stack(observation), perm=[1, 0, 2, 3, 4]) 

            loss, acc, new_tf_step = train_step(observation, y_train, model, optimizer, loss_fn, accuracy, tf_step)
            tf_step = new_tf_step
            loss_list.append(loss.numpy()), acc_list.append(acc.numpy())
            if epoch % eval_freq == 0:
                obs_test = []
                shape_env.reset(training=False)
                oneHotWrapper.reset(training=False)
                y_test = oneHotWrapper.target_index
                for t in range(s_len):
                    obs_test.append(oneHotWrapper.observe(training=False, time_step=t))
                obs_test =  tf.transpose(tf.stack(obs_test), perm=[1, 0, 2, 3, 4]) 
                loss_test, acc_test = eval_step(obs_test, y_test, model, loss_fn, accuracy)
                best_test_acc = max(best_test_acc, acc_test)
                acc_list_test.append(acc_test.numpy())
                loss_list_test.append(loss_test.numpy())
                print('Epoch ', epoch, ' Loss: ', loss.numpy(), ' Accuracy: ', acc.numpy(), 'Test Accuracy: ', acc_test.numpy(), 'Test Loss: ', loss_test.numpy())
                if log_results: out_file.write(f'Epoch: {epoch}, Loss: {loss.numpy()} - Accuracy: {acc.numpy()}%, Test Loss: {loss_test.numpy()}, Test Accuracy: {acc_test.numpy()}% \n \n')

                if epoch % 5000 == 0:
                    print('Saving...')
                    if not os.path.isdir(ckpt_path):
                        os.mkdir(ckpt_path)
                    best_acc.assign(acc_test.numpy())
                    curr_epoch.assign(epoch)
                    manager.save()

                # early stopping criteria
                # if len(loss_list_test) > last_x_losses and loss_test.numpy() > np.mean(loss_list_test[-last_x_losses:]):
                #     print("EARLY STOPPIBG AT - ", epoch)
                #     if log_results: out_file.write(f'EARLY STOPPIBG AT -  {epoch} \n')
                #     break

        print("Just plotting for seq_len " + str(s_len) + "  -----------------------------------------------------------")

        # summarize history for accuracy
        x = np.arange(len(acc_list))
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))

        ax1.plot(x, acc_list)
        ax1.set_title('Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('epoch')

        ax2.plot(x, loss_list)
        ax2.set_title('Loss')
        ax2.set_ylabel('Accuracy')
        ax2.set_xlabel('epoch')

        x_test = (np.arange(len(acc_list_test)) * eval_freq).astype(int)
        ax3.plot(x_test, acc_list_test)
        ax3.set_title('Test Accuracy')
        ax3.set_ylabel('Test Accuracy')
        ax3.set_xlabel('epoch')

        if log_results:
            file_name = "./baselines/lstm_cifar/runs_seed20/plot_3layer_ResNet18_padded_"+ str(patch_size[0]) +"patch_" +  str(int(s_len)) + "_seqLen_300k_new"
            fig.savefig(file_name)

        plt.clf()
        if log_results:
            out_file.write(f'Test results - Loss: {loss_list[-1]} - Accuracy: {acc_list_test[-1]}%, Best Accuracy: {best_test_acc}% \n \n')

if log_results: out_file.close()
    # plt.show()


end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

# Print or log the elapsed time
print(f"Total time taken: {elapsed_time:.2f} seconds")
if log_results:
    out_file.write(f"Total time taken: {elapsed_time:.2f} seconds \n")

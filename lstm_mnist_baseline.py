import os
os.environ["CUDA_VISIBLE_DEVICES"]= "3"


import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers.legacy import Adam
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
from cbgt_net.environments import MNISTCategoricalEnvironmentPadded
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from keras.datasets import mnist
# from baselines.resnet import *

tf.random.set_seed(
    10
)

log_results = False
padded_patches = True
resume_ckpt = False

def calculate_validation_loss(model, validation_data):
    # Perform validation on the validation_data and return the validation loss or metric of interest
    # Example:
    inputs, labels = validation_data
    logits = model(inputs)
    loss = loss_fn(labels, logits)
    return loss

# Define the early stopping function
def early_stopping(validation_loss, patience):
    if len(validation_loss) < patience:
        return True
    if all(validation_loss[-1] >= x for x in validation_loss[-patience - 1:-1]):
        return False  # Stop training if validation loss didn't improve for 'patience' consecutive steps
    return True  # Continue training


def get_LeNet5_Encoder(input_shape):
    encoder = models.Sequential()
    encoder.add(layers.Conv2D(6, (5, 5), activation='relu', input_shape=input_shape))
    encoder.add(layers.MaxPooling2D((2, 2)))
    encoder.add(layers.Conv2D(16, (5, 5), activation='relu'))
    encoder.add(layers.MaxPooling2D((2, 2)))
    encoder.add(layers.Flatten())
    encoder.add(layers.Dense(120, activation='relu'))
    encoder.add(layers.Dense(84, activation='relu'))
    encoder.add(layers.Dense(10, activation='softmax'))
    
    return encoder


class RNNModel(tf.keras.Model):
    def __init__(self, seq_length, patch_size, num_classes):
        super(RNNModel, self).__init__()
        
        lenet_encoder = get_LeNet5_Encoder(patch_size)
        # resent_encoder = ResNet('resnet18', num_classes)
        
        # then create our final model
        self.lstm_model = tf.keras.Sequential([tf.keras.layers.TimeDistributed(lenet_encoder, input_shape=(seq_length, patch_size[0],  patch_size[1],  patch_size[2])),
                                    # tf.keras.layers.SimpleRNN(10, kernel_initializer='he_normal'),
                                    tf.keras.layers.LSTM(10, kernel_initializer='he_normal'),
                                    # tf.keras.layers.Dense(128, activation = 'relu', kernel_initializer='he_normal'),
                                    tf.keras.layers.Dense(num_classes, activation = 'softmax')])
                                    # activation = 'softmax'
                                    

    def call(self, inputs, states=None):
        logits = self.lstm_model(inputs)
        return logits

def train_step(inputs, labels, model, optimizer, loss_fn, accuracy):
    with tf.GradientTape() as tape:
        logits = model(inputs)
        # l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables])
        loss = loss_fn(labels, logits) # + l2_loss * 0.001
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    accuracy.update_state(labels, logits)
    # print(model.summary())
    return loss, accuracy.result()

def eval_step(inputs, labels, model, loss_fn, accuracy):
    logits = model(inputs)
    loss = loss_fn(labels, logits)
    accuracy.update_state(labels, logits)
    return loss, accuracy.result()


batch_size = 512 
seq_length = [[1], [2, 3, 4, 5, 6] , [2, 3, 4, 5, 6] , [2, 3, 4, 6, 7], [2, 4, 6, 8, 11], [2, 5, 8, 11, 14], [3, 9, 14, 20, 27]]

num_classes = 10
image_sz = [28, 28, 3]
patch_size_list = [[28, 28, 3], [20, 20, 3], [16, 16, 3], [12, 12, 3], [10, 10, 3], [8, 8, 3], [5, 5, 3]]
noise = 0
number_of_epochs = 70000 #100000
eval_freq = 50
patience = 100
# training = False

last_x_losses = 500
last_x_acc = 500
conf_acc = 0.005

# file_name = '/home/shreya/cbgt_net/baselines/rnn_cifar/out_complete_patch_' + str(patch_size_list[0][0]) '_10seq.txt'
log_file = 'out_lenet5_10_8_5patch_70Kepochs_lstm_lr1e-3'
if log_results: out_file = open('./lstm_mnist/updated2/' + log_file + '.txt', 'a')

for i in range(len(patch_size_list)):
    patch_size = patch_size_list[i]
    # for s_len in seq_length:
    for s_len in seq_length[i]:
        if padded_patches:
            model = RNNModel(s_len, image_sz, num_classes)
        else:
            model = RNNModel(s_len, patch_size, num_classes)

        initial_learning_rate = 0.005
        lr_schedule = ExponentialDecay(
            initial_learning_rate, decay_steps=number_of_epochs, decay_rate=0.9, staircase=True
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  #0.005
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        shape_env = MNISTCategoricalEnvironmentPadded(10, noise=0.0, batch_size=batch_size, image_shape= image_sz,
                patch_size= patch_size,
                images_per_class_train= 5421,
                images_per_class_test= 892,
                max_steps_per_episode= s_len)

        oneHotWrapper = shape_env.OneHotWrapper(shape_env)

        print("Patch size = ", patch_size, " s_len = ", str(s_len))
        if log_results: out_file.write(f'For patch_size = {str(patch_size[0])} and s_len = {str(s_len)} after {str(number_of_epochs)} epochs \n')
        loss_list, acc_list , acc_list_test, loss_list_test = [], [], [], []
        best_test_acc = 0
        best_acc, curr_epoch = tf.Variable(0.0), tf.Variable(0.0)
        ckpt_path = "/home/shreya/cbgt_net/baselines/lstm_mnist/updated2/checkpoints/" + str(patch_size[0]) +"patch_" +  str(int(s_len)) + "_seqLen_50k"
        # ckpt_path = "/home/shreya/cbgt_net/baselines/lstm_mnist/new_runs
        #/checkpoints/8patch_" +  str(int(s_len)) + "_seqLen_50k"
        
        ckpt = tf.train.Checkpoint(curr_epoch=curr_epoch, best_acc=best_acc, optimizer=optimizer, model=model)
        manager = tf.train.CheckpointManager(ckpt, ckpt_path, max_to_keep=1)

        if resume_ckpt:
            # Load checkpoint.
            print('==> Resuming from checkpoint...')
            assert os.path.isdir(ckpt_path), 'Error: no checkpoint directory found!'

            # Restore the weights
            ckpt.restore(manager.latest_checkpoint)
        
        for epoch in range(number_of_epochs):
            # print("Epoch = ", epoch)
            observation = []
            shape_env.reset(training=True)
            oneHotWrapper.reset(training=True)
            y_train = oneHotWrapper.target_index
            for t in range(s_len):
                observation.append(oneHotWrapper.observe(training=True, time_step=t))
            observation =  tf.transpose(tf.stack(observation), perm=[1, 0, 2, 3, 4]) 
            # print("observe shape - ", observation.shape)

            loss, acc = train_step(observation, y_train, model, optimizer, loss_fn, accuracy)
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
                # # early stopping condition
                # if len(loss_list_test) < patience:
                #     continue
                # if all(loss_list_test[-1] >= x for x in loss_list_test[-patience - 1:-1]):
                #     if log_results: out_file.write(f'Early stopping at epoch: {epoch} \n')
                #     break # Stop training if validation loss didn't improve for 'patience' consecutive steps

                # if acc_test.numpy() > best_acc.numpy():
                if epoch % 5000 == 0:
                    print('Saving...')
                    if not os.path.isdir(ckpt_path):
                        os.mkdir(ckpt_path)
                    best_acc.assign(acc_test.numpy())
                    curr_epoch.assign(epoch)
                    manager.save()

                # if len(loss_list_test) > last_x_losses and loss_test.numpy() > np.mean(loss_list_test[-last_x_losses:]):
                #     print("EARLY STOPPIBG AT - ", epoch)
                #     if log_results: out_file.write(f'EARLY STOPPIBG AT -  {epoch} \n')
                #     break

                # if len(acc_list_test) > last_x_acc and np.std(acc_list_test[-last_x_acc:]) < conf_acc:
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
            file_name = "/home/shreya/cbgt_net/baselines/lstm_mnist/updated2/plot_lenet_padded_"+ str(patch_size[0]) +"patch_" +  str(int(s_len)) + "_seqLen_70k_lstm"
            fig.savefig(file_name)

        plt.clf()
        if log_results:
            out_file.write(f'Test results - Loss: {loss_list[-1]} - Accuracy: {acc_list_test[-1]}%, Best Accuracy: {best_test_acc}% \n \n')

if log_results: out_file.close()
    # plt.show()

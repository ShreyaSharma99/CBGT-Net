import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import matplotlib.pyplot as plt
import numpy as np
import yappi
import tensorflow as tf
from cbgt_net.training import REINFORCE_Trainer
from cbgt_net.components import CBGT_Net
from cbgt_net.training import losses
from cbgt_net.training.reward import SimpleCategoricalReward
from cbgt_net.training.losses import REINFORCE_Loss, EntropyLoss, CE_Loss
from cbgt_net.training.losses import CompositeLoss
from cbgt_net.components import EvidenceShapePatchEncoderCIFAR, EvidenceShapePatchEncoder
from cbgt_net.components import SimpleAccumulatorModule
from cbgt_net.components import FixedDecisionThresholdModule
from cbgt_net.training.measures import AccuracyMeasure
from cbgt_net.environments import CIFAR_CategoricalEnvironment, MNISTCategoricalEnvironment
import seaborn as sns
from sklearn.metrics import confusion_matrix
import argparse
import warnings

warnings.filterwarnings("ignore")

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Describe the --dataset, --patch_sz, --threshold')

# Add an argument for 'env'
parser.add_argument('--env', type=str, default='cifar', help='Specify the environment (cifar, mnist)')
parser.add_argument('--patch_sz', type=int, default=16, help='Specify the environment (5, 8, 10, 12, 16, 20)')
parser.add_argument('--threshold', type=int, default=4, help='Specify the environment (1, 2, 3, 4, 5)')

# Parse the command-line arguments
args = parser.parse_args()
env_data = args.env
patch_sz = args.patch_sz
threshold = args.threshold

if env_data == "cifar":
	batch_size = 256 if threshold < 3  else 128
	if patch_sz==20 and threshold==2: batch_size = 128
	ckpt_file = "checkpoints_cbgt/cifar/checkpoints_cifar_patch_" + str(patch_sz) + "_2Losses_0ElossMasked_" + str(threshold) + "fixedThreshold_" + str(threshold*10 + 1) + "maxSteps_" + str(batch_size) + "batchSize_softmax_allData_Padded_ceLoss_3layerResnet_30M"
elif env_data == "mnist":
	batch_size = 512
	ckpt_file = "checkpoints_cbgt/mnist/checkpoints_mnist_patch_" + str(patch_sz) + "_fulldata_ceLoss_" + str(threshold) + "fixedThreshold_" + str(threshold*10 + 1) + "maxSteps_" + str(batch_size) + "batchSize_softmax_LeNet5_100M_final"
else:
	raise ValueError("--dataset should be 'cifar' or 'mnist'")


# batch_size = 128
org_patch_size = [patch_sz, patch_sz, 3]
patch_sz = [32, 32, 3] if env_data == "cifar" else [28, 28, 3]
max_steps_per_episode = threshold*10 + 1
# threshold = 2


if env_data == "cifar":
	encoder = EvidenceShapePatchEncoderCIFAR(num_hidden_units = 25, num_categories=10,
					hidden_activation = "relu",
					output_activation = "softmax",
					input_shape = patch_sz,
					evidence_dim = 128,
					use_resnet18 = True)
	shape_env = CIFAR_CategoricalEnvironment(10, noise=0.0, batch_size=batch_size, 
					image_shape= patch_sz,
					patch_size= org_patch_size,
					images_per_class_train= 5000,
					images_per_class_test= 1000,
					max_steps_per_episode= max_steps_per_episode)

elif env_data == "mnist":
	encoder = EvidenceShapePatchEncoder(num_hidden_units = 25, num_categories=10,
					hidden_activation = "relu",
					output_activation = "softmax",
					input_shape = patch_sz,
					evidence_dim = 128,
					use_resnet18 = True)
	shape_env = MNISTCategoricalEnvironmentPadded(10, noise=0.0, batch_size=batch_size, 
					image_shape= patch_sz,
					patch_size= org_patch_size,
					images_per_class_train= 5421,
					images_per_class_test= 892,
					max_steps_per_episode= max_steps_per_episode)

accumular = SimpleAccumulatorModule(batch_size = batch_size, num_categories=10)
threshold_module = FixedDecisionThresholdModule(num_categories=10, decision_threshold=threshold)
oneHotWrapper = shape_env.OneHotWrapper(shape_env)

reward = SimpleCategoricalReward(correct_guess_reward= 30,
                                incorrect_guess_reward= -30,
                                timeout_reward= -30,
                                no_guess_reward=0,
                                tardiness_rate = 0)

ce_loss = CE_Loss(timesteps=max_steps_per_episode, num_batches=batch_size)
entropy_loss =  EntropyLoss()
loss = CompositeLoss()
loss.add_loss("reinforce", ce_loss)
loss.add_loss("entropy", entropy_loss, weight=0.0)
measure = AccuracyMeasure()

# ckpt_file = "checkpoints_cifar_patch_16_2Losses_0ElossMasked_4fixedThreshold_41maxSteps_128batchSize_softmax_allData_Padded_ceLoss_3layerResnet_30M"

# ckpt_file ="/home/shreya/CBGT-NET/checkpoints_cbgt/cifar/checkpoints_cifar_patch_20_2Losses_0ElossMasked_2fixedThreshold_21maxSteps_128batchSize_softmax_allData_Padded_ceLoss_3layerResnet_30M"
#  "/home/shreya/CBGT-NET/checkpoints_cbgt/checkpoints_cifar_patch_16_2Losses_0ElossMasked_4fixedThreshold_41maxSteps_128batchSize_softmax_allData_Padded_ceLoss_3layerResnet_30M"

	# ckf = ckpt_file[i]
	# batch_size = batch_size[i]
	# max_steps_per_episode = max_steps_per_episode_list[i]


model = CBGT_Net(10, evidence_module=encoder, accumulator_module=accumular, threshold_module=threshold_module)
trainer = REINFORCE_Trainer(model, shape_env, reward, loss)

checkpoint = tf.train.Checkpoint(model=model, optimizer=trainer.optimizer, step=tf.Variable(1))

checkpoint_path = ckpt_file
# './ichms_cbgt_checkpoints/' + ckpt_file
manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=10)  # Replace max_to_keep with the desired number of checkpoints to keep

# Restore the latest checkpoint
status = checkpoint.restore(manager.latest_checkpoint)
status.expect_partial()

# Check if the checkpoint is successfully restored
if manager.latest_checkpoint:
	print("Model restored from:", manager.latest_checkpoint)
else:
	print("No checkpoint found.")

training = False
# max_steps_per_episode = 21
shape_env.reset(training=training)
oneHotWrapper.reset(training=training)
model.reset()

# Setup tensor arrays to return once the function is complete
observations = tf.TensorArray(tf.float32, size=max_steps_per_episode, dynamic_size=False, clear_after_read=False)
targets = tf.TensorArray(tf.int32, size=max_steps_per_episode, dynamic_size=False, clear_after_read=False)
evidence = tf.TensorArray(tf.float32, size=max_steps_per_episode, dynamic_size=False, clear_after_read=False)
accumulators = tf.TensorArray(tf.float32, size=max_steps_per_episode, dynamic_size=False, clear_after_read=False)
decision_distributions = tf.TensorArray(tf.float32, size=max_steps_per_episode, dynamic_size=False, clear_after_read=False)
did_decide_probabilities = tf.TensorArray(tf.float32, size=max_steps_per_episode, dynamic_size=False, clear_after_read=False)
did_decides = tf.TensorArray(tf.bool, size=max_steps_per_episode, dynamic_size=False, clear_after_read=False)
decisions = tf.TensorArray(tf.int32, size=max_steps_per_episode, dynamic_size=False, clear_after_read=False)
decision_masks = tf.TensorArray(tf.bool, size=max_steps_per_episode+1, dynamic_size=False, clear_after_read=False)

threshold_vectors = tf.TensorArray(tf.float32, size=max_steps_per_episode, dynamic_size=False, clear_after_read=False)

# Set up the first decision mask to False
decision_masks = decision_masks.write(0, tf.cast(tf.zeros((shape_env.batch_size, 1)), tf.bool))

# Run for the maximum number of steps
for t in tf.range(max_steps_per_episode):

	observation = oneHotWrapper.observe(training=training, time_step=t)
	# print("Observation - ", observation.shape)

	# Store values of the environment at this time step, and the value of
	# the accumulator prior to execution
	observations = observations.write(t, observation)#.mark_used()
	targets = targets.write(t, shape_env.target_index)#.mark_used()
	accumulators = accumulators.write(t, model.accumulator)#.mark_used()


	# Do a forward pass on the model
	decision_distribution, did_decide_probability = model(observation)
	
	# Store model parameters and outputs for this time step
	evidence = evidence.write(t, model.evidence_tensor)#.mark_used()
	decision_distributions = decision_distributions.write(t, decision_distribution)#.mark_used()
	did_decide_probabilities = did_decide_probabilities.write(t, did_decide_probability)#.mark_used()


	# NOTE:  It may be quicker to split this into two functions.  Will
	#        need to explore and benchmark
	if training:
		decision = tf.random.categorical(tf.math.log(decision_distribution+1e-32), 1, dtype=tf.int32)
		did_decide = tf.math.less(tf.random.uniform(did_decide_probability.shape), did_decide_probability)
	else:
		decision = tf.reshape(tf.cast(tf.math.argmax(decision_distribution+1e-32, 1), tf.int32), (-1,1))
		did_decide = tf.math.greater(did_decide_probability, 0.5)

	if t == max_steps_per_episode-1:
		did_decide = tf.constant(True, dtype=None, shape=did_decide.shape)

	did_decides = did_decides.write(t, did_decide)#.mark_used()
	decisions = decisions.write(t, decision)#.mark_used()

	decision_masks = decision_masks.write(t+1, tf.math.logical_or(decision_masks.read(t), did_decide))

observations = observations.stack()
targets = targets.stack()
evidence = evidence.stack()
accumulators = accumulators.stack()
decision_probabilities = decision_distributions.stack()
did_decide_probabilities = did_decide_probabilities.stack()
did_decide = did_decides.stack()
decisions = decisions.stack()
decision_masks = decision_masks.stack()[:-1]

decision_idx = max_steps_per_episode - tf.reduce_sum(tf.cast(decision_masks, tf.int32), axis=0) - 1
batch_idx = tf.reshape(tf.range(batch_size), (-1,1))
idx = tf.concat([decision_idx, batch_idx], 1)

predictions = tf.squeeze(tf.gather_nd(decisions, idx))
targets_indexed = tf.squeeze(tf.gather_nd(targets, idx))

# Calculate the number of correct predictions
correct = tf.reduce_sum(tf.cast(predictions==targets_indexed, tf.int32))
accuracy = tf.cast(correct, tf.float32) / batch_size
tf.print(f'Test Accuracy = {accuracy*100}%')
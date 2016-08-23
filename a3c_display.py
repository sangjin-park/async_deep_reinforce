# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import random
import os

from game_state import GameState
from game_ac_network import GameACFFNetwork, GameACLSTMNetwork
from a3c_training_thread import A3CTrainingThread
from rmsprop_applier import RMSPropApplier

import options
options = options.options

def choose_action(pi_values):
    pi_values -= np.finfo(np.float32).epsneg
    action_samples = np.random.multinomial(options.num_experiments, pi_values)
    return action_samples.argmax(0)

# use CPU for display tool
device = "/cpu:0"

if options.use_lstm:
  global_network = GameACLSTMNetwork(options.action_size, -1, device)
else:
  global_network = GameACFFNetwork(options.action_size, device)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

saver = tf.train.Saver()
checkpoint = tf.train.get_checkpoint_state(options.checkpoint_dir)
if checkpoint and checkpoint.model_checkpoint_path:
  saver.restore(sess, checkpoint.model_checkpoint_path)
  print("checkpoint loaded:", checkpoint.model_checkpoint_path)
else:
  print("Could not find old checkpoint")


game_state = GameState(0, options, display=options.display, no_op_max=30)

episode = 0
while True:
  episode += 1
  episode_record_dir = None
  if options.record_screen_dir is not None:
    episode_dir = "episode{:03d}".format(episode)
    episode_record_dir = os.path.join(options.record_screen_dir, episode_dir)
    os.makedirs(episode_record_dir)
    game_state.set_record_screen_dir(episode_record_dir)

  steps = 0
  reward = 0
  while True:
    pi_values = global_network.run_policy(sess, game_state.s_t)

    action = choose_action(pi_values)
    game_state.process(action)
    reward += game_state.reward

    # terminate if the play time is too long
    steps += 1
    terminal = game_state.terminal
    if steps > options.max_play_steps:
      terminal =  True

    game_state.update()

    if terminal:
      print("Game finised with score=", reward)
      game_state.reset()
      break

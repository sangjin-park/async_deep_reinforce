# -*- coding: utf-8 -*-
import sys
import numpy as np
import cv2
import os
from math import sqrt
from ale_python_interface import ALEInterface


class GameState(object):
  def __init__(self, rand_seed, options, display=False, no_op_max=30, thread_index=-1):
    self.ale = ALEInterface()
    self.ale.setInt(b'random_seed', rand_seed)
    self.ale.setFloat(b'repeat_action_probability', options.repeat_action_probability)
    self.ale.setInt(b'frame_skip', options.frames_skip_in_ale)
    self.ale.setBool(b'color_averaging', options.color_averaging_in_ale)
    self._no_op_max = no_op_max
 
    self.options = options
    self.color_maximizing = options.color_maximizing_in_gs
    # for screen output in _process_frame()
    self.thread_index = thread_index
    self.record_gs_screen_dir = self.options.record_gs_screen_dir
    self.episode_record_dir = None
    self.episode = 1

    if display:
      self._setup_display()
    
    self.ale.loadROM(options.rom.encode('ascii'))

    # collect minimal action set
    self.real_actions = self.ale.getMinimalActionSet()
    print("real_actions=", self.real_actions)
    if (len(self.real_actions) != self.options.action_size):
      print("***********************************************************")
      print("* action_size != len(real_actions)")
      print("***********************************************************")
      sys.exit(1)

    # height=210, width=160
    self._screen = np.empty((210 * 160 * 1), dtype=np.uint8)
    if self.color_maximizing:
      self._screen_RGB = np.empty((210 * 160 * 3), dtype=np.uint8)
      self._prev_screen_RGB = np.empty((210 *  160 * 3), dtype=np.uint8)

    # for pseudo-count
    self.psc_use = options.psc_use
    if options.psc_use:
      self.psc_frsize = options.psc_frsize
      self.psc_k = options.psc_frsize ** 2
      self.psc_beta = options.psc_beta
      self.psc_maxval = options.psc_maxval
      self.psc_vcount = np.zeros((self.psc_k, self.psc_maxval + 1), dtype=np.float32)
      self.psc_n = 0

    self.reset()
 
  # for pseudo-count
  def psc_add_image(self, psc_image):
    k = self.psc_k
    n = self.psc_n
    if n > 0:
      nr = (n + 1.0)/n
      vcount = self.psc_vcount[range(k), psc_image]
      self.psc_vcount[range(k), psc_image] += 1.0
      r_over_rp = np.prod(nr * vcount / (1.0 + vcount))
      psc_count = r_over_rp / (1.0 - r_over_rp)
      psc_reward = self.psc_beta / sqrt(psc_count + 0.01)
    else:
      self.psc_vcount[range(k), psc_image] += 1.0
      psc_reward = 0.0
    
    self.psc_n += 1

    if self.psc_n % self.options.log_interval == 0:
      print("th={},psc_n={}:psc_reward = {:.8f}".format(self.thread_index, self.psc_n, psc_reward))

    return psc_reward   

  def set_record_screen_dir(self, record_screen_dir):
    print("record_screen_dir", record_screen_dir)
    self.ale.setString(b'record_screen_dir', str.encode(record_screen_dir))
    self.ale.loadROM(self.options.rom.encode('ascii'))
    self.reset()

  def _process_action(self, action):
    reward = self.ale.act(action)
    terminal = self.ale.game_over()
    self.terminal = terminal
    self._have_prev_screen_RGB = False
    return reward, terminal
    
  def _process_frame(self, action, reshape):
    if self.terminal:
      reward = 0
      terminal = True
    else:
      # get previous screen
      if self.color_maximizing and not self._have_prev_screen_RGB:
        self.ale.getScreenRGB(self._prev_screen_RGB)
        self._have_prev_screen_RGB = True

      # make action
      reward = self.ale.act(action)
      terminal = self.ale.game_over()
      self.terminal = terminal

    # screen shape is (210, 160, 1)
    if self.color_maximizing:
      self.ale.getScreenRGB(self._screen_RGB)
      if self._have_prev_screen_RGB:
        screen = np.maximum(self._prev_screen_RGB, self._screen_RGB)
      else:
        screen = self._screen_RGB
      screen = screen.reshape((210, 160, 3))
      self._screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
      # swap screen_RGB
      swap_screen_RGB = self._prev_screen_RGB
      self._prev_screen_RGB = self._screen_RGB
      self._screen_RGB = swap_screen_RGB
      self._have_prev_screen_RGB = True
    else:
      self.ale.getScreenGrayscale(self._screen)
    
    # reshape it into (210, 160)
    reshaped_screen = np.reshape(self._screen, (210, 160))
    
    # resize to height=110, width=84
    resized_screen = cv2.resize(reshaped_screen, (84, 110))
    
    x_t = resized_screen[18:102,:]
    
    # pseudo-count
    psc_reward = 0.0
    if self.psc_use:
      psc_image = cv2.resize(x_t, (self.psc_frsize, self.psc_frsize))
      psc_image = np.reshape(psc_image, (self.psc_k))
      psc_image = np.uint8(psc_image * (self.psc_maxval / 255.0))
      psc_reward = self.psc_add_image(psc_image)

    if reshape:
      x_t = np.reshape(x_t, (84, 84, 1))
    x_t = x_t.astype(np.float32)
    x_t *= (1.0/255.0)
    return reward, terminal, x_t, psc_reward
    
    
  def _setup_display(self):
    if sys.platform == 'darwin':
      import pygame
      pygame.init()
      self.ale.setBool(b'sound', False)
    elif sys.platform.startswith('linux'):
      self.ale.setBool(b'sound', True)
    self.ale.setBool(b'display_screen', True)

  def reset(self):
    self.ale.reset_game()
    
    # randomize initial state
    if self._no_op_max > 0:
      no_op = np.random.randint(0, self._no_op_max // self.options.frames_skip_in_ale + 1)
      for _ in range(no_op):
        self.ale.act(0)

    self._have_prev_screen_RGB = False
    self.terminal = False
    _, _, x_t, _ = self._process_frame(0, False)
    
    self.reward = 0
    self.s_t = np.stack((x_t, x_t, x_t, x_t), axis = 2)

    self.lives = float(self.ale.lives())
    self.initial_lives = self.lives

    if (self.thread_index == 0) and (self.record_gs_screen_dir is not None):
      episode_dir = "episode{:03d}".format(self.episode)
      self.episode_record_dir = os.path.join(self.record_gs_screen_dir, episode_dir)
      os.makedirs(self.episode_record_dir)
      self.episode += 1
      self.stepNo = 1
      print("game_state: writing screen images to ", self.episode_record_dir)
    
  def process(self, action):
    # convert original 18 action index to minimal action set index
    real_action = self.real_actions[action]

    # altered for speed up (reduce getScreen and color_maximizing)
    reward = 0
    for _ in range(self.options.frames_skip_in_gs - 1):
      r, t = self._process_action(real_action)
      reward = reward + r
      if t:
        break

    r, t, x_t1, psc_r = self._process_frame(real_action, True)
    reward = reward + r

    self.reward = reward
    self.terminal = t
    self.s_t1 = np.append(self.s_t[:,:,1:], x_t1, axis = 2)
    self.psc_reward = psc_r

    self.lives = float(self.ale.lives())

    if self.episode_record_dir is not None:
      filename = "{:06d}.png".format(self.stepNo)
      filename = os.path.join(self.episode_record_dir, filename)
      self.stepNo += 1
      screen_image = x_t1.reshape((84, 84)) * 255.
      cv2.imwrite(filename, screen_image)


  def update(self):
    self.s_t = self.s_t1

# -*- coding: utf-8 -*-
import argparse
import sys

LOCAL_T_MAX = 5 # repeat step size
RMSP_ALPHA = 0.99 # decay parameter for RMSProp
RMSP_EPSILON = 0.1 # epsilon parameter for RMSProp
CHECKPOINT_DIR = 'checkpoints'
LOG_FILE = 'tmp/a3c_log'
INITIAL_ALPHA_LOW = 1e-4    # log_uniform low limit for learning rate
INITIAL_ALPHA_HIGH = 1e-2   # log_uniform high limit for learning rate

PARALLEL_SIZE = 8 # parallel thread size
ROM = "breakout.bin"     # action size = 3
ACTION_SIZE = 4 # action size

INITIAL_ALPHA_LOG_RATE = 0.4226 # log_uniform interpolate rate for learning rate (around 7 * 10^-4)
GAMMA = 0.99 # discount factor for rewards
ENTROPY_BETA = 0.01 # entropy regurarlization constant
MAX_MEGA_STEP = 100 # max  learning step (in Mega step)
SAVE_MEGA_STEP = 30 # save learning step (in Mega step)
SAVE_MEGA_INTERVAL = 3 # save interval (in Mega step)
MAX_TO_KEEP = None # maximum number of recent checkpoint files to keep (None means no-limit)

GRAD_NORM_CLIP = 40.0 # gradient norm clipping
USE_GPU = True # To use GPU, set True
USE_LSTM = False # True for A3C LSTM, False for A3C FF

MAX_PLAY_TIME  = 300 # Max play time in seconds

TERMINATE_ON_LIVES_LOST = False # Terminate game if lives lost
TRAIN_IN_EVAL = False # Train in evaluation thread(thread 0) in "TERMINATE_ON_LIVES_LOST mode"

NUM_EXPERIMENTS = 1 # number of experiments to determin action
LIVES_LOST_REWARD =-1.0 # Reward for lives lost (-1.0 - 0.0)
LIVES_LOST_WEIGHT = 1.0 # Weight of lives lost envet

BASIC_INCOME_TIME  = 10 ** 20 # Basic income time for reward 1.0 in seconds (huge number means no basic income)

PSC_USE = False # use pseudo-count
PSC_BETA = 0.01 # Beta in pseudo-count
PSC_FRSIZE = 42 # frame size in pseudo-count
PSC_MAXVAL = 127 # max value of pixels in pseudo-count 
REPEAT_ACTION_PROBABILITY = 0.0 # stochasticity option for ALE

NO_REWARD_TIME  = 15 # Permitted No reward time in seconds

RANDOMNESS_TIME = 300 # Time to max randomness(1.0)
RANDOMNESS_LOG_NUM = 30 # The number of randmness log

COLOR_AVERAGING_IN_ALE = True # Color averagin in ALE
COLOR_MAXIMIZING_IN_GS = False # Color maximizing in GS

TRAIN_EPISODE_STEPS = 0 # train steps for new record (no train if "< LOCAL_T_MAX". record only)
REWARD_CLIP = 1.0 # Clip reward by -REWARD_CLIP - REWARD_CLIP. (0.0 means no clip)
RESET_MAX_REWARD = False # Reset max reward in new episode

LOG_INTERVAL = 100 # Log output interval (steps)
SCORE_LOG_INTERVAL = 900 # Score log output interval (steps)
PERFORMANCE_LOG_INTERVAL = 1500 # Performance log output interval (steps)

RECORD_SCREEN_DIR = None # Game screen (output of ALE) record directory 
RECORD_GS_SCREEN_DIR = None # Game screen (input to A3C) record directory
RECORD_NEW_RECORD_DIR = None # New record record dirctory

DISPLAY = True # Display in a3c_display.py (set False in headless environment)
VERBOSE = True # Output options (to record run parameter)


# utility for args conversion
# convert boolean string to boolean value
def convert_boolean_arg(args, name):
  args = vars(args)
  if args[name] == "True" or args[name] == "true" :
    args[name] = True
  elif args[name] == "False" or args[name] == "false" :
    args[name] = False
  else:
    print("ERROR: --{} '{}' (must be 'True' or 'False')".format(
          name.replace("_", "-") ,args[name]))
    sys.exit(1)

# get steps from time(seconds)
def sec_to_steps(args, sec):
  return int((60. / (args.frames_skip_in_ale * args.frames_skip_in_gs)) * sec)

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--local-t-max', type=int, default=LOCAL_T_MAX)
parser.add_argument('--rmsp-alpha', type=float, default=RMSP_ALPHA)
parser.add_argument('--rmsp-epsilon', type=float, default=RMSP_EPSILON)
parser.add_argument('--checkpoint-dir', type=str, default=CHECKPOINT_DIR)
parser.add_argument('--log-file', type=str, default=LOG_FILE)
parser.add_argument('--initial-alpha-low', type=float, default=INITIAL_ALPHA_LOW)
parser.add_argument('--initial-alpha-high', type=float, default=INITIAL_ALPHA_HIGH)

parser.add_argument('--parallel-size', type=int, default=PARALLEL_SIZE)
parser.add_argument('--rom', type=str, default=ROM)
parser.add_argument('--action-size', type=int, default=ACTION_SIZE)

parser.add_argument('--initial-alpha-log-rate', type=float, default=INITIAL_ALPHA_LOG_RATE)
parser.add_argument('--gamma', type=float, default=GAMMA)
parser.add_argument('--entropy-beta', type=float, default=ENTROPY_BETA)
parser.add_argument('--max-mega-step', type=int, default=MAX_MEGA_STEP)
parser.add_argument('--max-time-step', type=int, default=None)
parser.add_argument('--save-mega-step', type=int, default=SAVE_MEGA_STEP)
parser.add_argument('--save-time-step', type=int, default=None)
parser.add_argument('--save-mega-interval', type=int, default=SAVE_MEGA_INTERVAL)
parser.add_argument('--save-time-interval', type=int, default=None)
parser.add_argument('--max-to-keep', type=int, default=MAX_TO_KEEP)

parser.add_argument('--grad-norm-clip', type=float, default=GRAD_NORM_CLIP)
parser.add_argument('--use-gpu', type=str, default=str(USE_GPU))
parser.add_argument('--use-lstm', type=str, default=str(USE_LSTM))

parser.add_argument('--max-play-time', type=int, default=MAX_PLAY_TIME)
parser.add_argument('--max-play-steps', type=int, default=None)
parser.add_argument('--terminate-on-lives-lost', type=str, default=str(TERMINATE_ON_LIVES_LOST))
parser.add_argument('--train-in-eval', type=str, default=str(TRAIN_IN_EVAL))
parser.add_argument('--num-experiments', type=int, default=NUM_EXPERIMENTS)
parser.add_argument('--lives-lost-reward', type=float, default=LIVES_LOST_REWARD)
parser.add_argument('--lives-lost-weight', type=float, default=LIVES_LOST_WEIGHT)
parser.add_argument('--basic-income-time', type=int, default=BASIC_INCOME_TIME)
parser.add_argument('--basic-income', type=float, default=None)

parser.add_argument('--psc-use', type=str, default=str(PSC_USE))
parser.add_argument('--psc-beta', type=float, default=PSC_BETA)
parser.add_argument('--psc-frsize', type=int, default=PSC_FRSIZE)
parser.add_argument('--psc-maxval', type=int, default=PSC_MAXVAL)
parser.add_argument('--repeat-action-probability', type=float, default=REPEAT_ACTION_PROBABILITY)

parser.add_argument('--no-reward-time', type=int, default=NO_REWARD_TIME)
parser.add_argument('--no-reward-steps', type=int, default=None)
parser.add_argument('--randomness-time', type=float, default=RANDOMNESS_TIME)
parser.add_argument('--randomness-steps', type=float, default=None)
parser.add_argument('--randomness', type=float, default=None)
parser.add_argument('--randomness-log-num', type=int, default=RANDOMNESS_LOG_NUM)
parser.add_argument('--randomness-log-interval', type=int, default=None)
parser.add_argument('--color-averaging-in-ale', type=str, default=str(COLOR_AVERAGING_IN_ALE))
parser.add_argument('--frames-skip-in-ale', type=int, default=None)
parser.add_argument('--color-maximizing-in-gs', type=str, default=str(COLOR_MAXIMIZING_IN_GS))
parser.add_argument('--frames-skip-in-gs', type=int, default=None)
parser.add_argument('--train-episode-steps', type=int, default=TRAIN_EPISODE_STEPS)
parser.add_argument('--reward-clip', type=float, default=REWARD_CLIP)
parser.add_argument('--reset-max-reward', type=str, default=str(RESET_MAX_REWARD))

parser.add_argument('--log-interval', type=int, default=LOG_INTERVAL)
parser.add_argument('--score-log-interval', type=int, default=SCORE_LOG_INTERVAL)
parser.add_argument('--performance-log-interval', type=int, default=PERFORMANCE_LOG_INTERVAL)

parser.add_argument('--record-screen-dir', type=str, default=RECORD_SCREEN_DIR)
parser.add_argument('--record-gs-screen-dir', type=str, default=RECORD_GS_SCREEN_DIR)
parser.add_argument('--record-new-record-dir', type=str, default=RECORD_NEW_RECORD_DIR)

parser.add_argument('--display', type=str, default=str(DISPLAY))

parser.add_argument('-v', '--verbose', type=str, default=str(VERBOSE))

args = parser.parse_args()

convert_boolean_arg(args, "use_gpu")
convert_boolean_arg(args, "use_lstm")
convert_boolean_arg(args, "terminate_on_lives_lost")
convert_boolean_arg(args, "train_in_eval")
convert_boolean_arg(args, "psc_use")
convert_boolean_arg(args, "color_averaging_in_ale")
convert_boolean_arg(args, "color_maximizing_in_gs")
convert_boolean_arg(args, "reset_max_reward")
convert_boolean_arg(args, "display")
convert_boolean_arg(args, "verbose")

if (args.color_averaging_in_ale and args.color_maximizing_in_gs) or\
   not (args.color_averaging_in_ale or args.color_maximizing_in_gs):
  print("Specify just one of color_averaging_in_ale or args_color_maximizing")
  sys.exit(1)

if args.color_averaging_in_ale:
  if args.frames_skip_in_ale is None:
    args.frames_skip_in_ale = 4
  args.frames_skip_in_gs = 1
if args.color_maximizing_in_gs:
  if args.frames_skip_in_gs is None:
    args.frames_skip_in_gs = 4
  args.frames_skip_in_ale = 1

if args.max_time_step is None:
  args.max_time_step = args.max_mega_step * 10**6
if args.save_time_step is None:
  args.save_time_step = args.save_mega_step * 10**6
if args.save_time_interval is None:
  args.save_time_interval = args.save_mega_interval * 10**6

if args.max_play_steps is None:
  args.max_play_steps = sec_to_steps(args, args.max_play_time)

if args.basic_income is None:
  args.basic_income = 1.0 / sec_to_steps(args, args.basic_income_time)
if args.basic_income < 1e-10:
  args.basic_income = 0.0

if args.no_reward_steps is None:
  args.no_reward_steps = sec_to_steps(args, args.no_reward_time)

if args.randomness_steps is None:
  args.randomness_steps = sec_to_steps(args, args.randomness_time)
if args.randomness is None:
  args.randomness = 1.0 / args.randomness_steps
if args.randomness_log_interval is None:
  args.randomness_log_interval = args.randomness_steps / args.randomness_log_num

options = args
if options.verbose:
  print("******************** options ********************")
  print(options)
  print("*************************************************")

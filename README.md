# async_deep_reinforce

Asynchronous deep reinforcement learning + Pseudo-count based reward + On-highscore-learning

## About

This code is fork from [miyosuda's code](https://github.com/miyosuda/async_deep_reinforce). I added many functions for my Deep Learning experiments. Of which, pseudo-count based reward based on following DeepMind's paper and on-highscore-learning (my original) enable over 500 point average score in Montezuma's Revenge, which is higher than the paper as for A3C.
 
[https://arxiv.org/abs/1606.01868 (Unifying Count-Based Exploration and Intrinsic Motivation, DeepMind)](https://arxiv.org/abs/1606.01868)

"on-highscore-learning" is my original idea, which learn from state-action-rewards-history when getting highscore. But in evaluation of Montezuma's Revenge, I set option to reset highscore in every episode, so learning occured in every score. (I'm changing this now. In new version, only highscore episode will be selected automatically based on history of scores) 

## Learning curve of Montezuma's Revenge

The following graph is the average score of Montezuma's Revenge.

![learning result after 39M steps](https://github.com/Itsukara/async_deep_reinforce/blob/master/learning-curves/montezuma-psc-39M.png)

0 - 30M steps: Pseudo-count based reward is ON, on-highscore-learning is ON. (--psc-use=True --train-episode-steps=150)

30 - 30M steps: Above + Reset highscore in every episode. (--reset-max-reward=True)


## Play movie

The following is a play movie of Montezuma's Revenge after training 50M steps. Its score is 2600.

[![](http://img.youtube.com/vi/tts3wOPnKQE/0.jpg)](https://www.youtube.com/watch?v=tts3wOPnKQE)

## How to prepare environment

This code needs Anaconda, tensorflow, opencv3 and Arcade Learning Environment (ALE). You can use scrips in "gcp-install" directory. Run following.
 
    $ mkdir Download
    $ cp gcp-install-a3c-env.tgz Download/
    $ cd Download/
    $ tar zxvf gcp-install-a3c-env.tgz
    $ bash -x install-Anaconda.sh
    $ . ~/.bashrc
    $ bash -x install-tensorflow.sh
    $ bash -x install-opencv3.sh
    $ bash -x install-ALE.sh
    $ bash -x install-additionals.sh
    $ git clone https://github.com/Itsukara/async_deep_reinforce.git
    $ cd async_deep_reinforce
    $ mv checkpoints checkpoints.old

When program requests input, just hit Enter or input "y" or "yes" and hit Enter. But as for Anaconda, you have to input "q" when License and "--More--" is displayed.

## How to run

To train,

    $ run-option montezuma-4

To display the result with game play,

    $ python a3c_display.py --rom=montezuma_revenge.bin

## Run options

As for options, see options.py.

## Status of code

The source code is still under development and may chage frequently. Currently, I'm searching best parameters to speed-up learning and get higher score. In this search, I'm adding new functions to change behavior of the program. So, it might be degraded sometimes. Sorry for that in advance.

## Sharing experiment result

I'd appreciate if you could write your experiment result to thread ["Experiment Results"](https://github.com/Itsukara/async_deep_reinforce/issues/3) in Issues. 

## Blog

I'm writing blog on this program. See following (in Japanese):

[http://itsukara.hateblo.jp/ (Itsukara's Blog)](http://itsukara.hateblo.jp/)

## How to refer

I'd appreciate if you woud refer my code in your blog or paper as following:

https://github.com/Itsukara/async_deep_reinforce (On-Highscore-Learning code based on A3C+ and Pseudo-count developed by Itsukara) 

## Acknowledgements

- [@miosuda](https://github.com/miyosuda/async_deep_reinforce) for providing very fast A3C program.


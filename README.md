# MGCM

The Modular Gradient Conflict Mitigation (MGCM) strategy can detect conflicts at a finer-grained modular level and resolves them utilizing gradient projection.

## Installation

- You can install any version of fairseq:

  ```bash
  git clone https://github.com/pytorch/fairseq
  cd fairseq
  pip install --editable ./
  ```

## Quick Start

- First you need to choose your task.

- After that reload the train_step of your chosen task according to the code we provided.

- Finally tweak your criterion to make sure it returns a list with the losses of multiple tasks.

  It's easy enough for you to execute as you want without any problems!

## Tips

​	If you want to do multitasking optimization, you can change line 96 in the code to random sampling.



Please feel free to contact me if you have bugs while using it!

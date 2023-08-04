# FSRS Optimizer Tiny

Rewrite [FSRS Optimizer](https://github.com/open-spaced-repetition/fsrs-optimizer) in [tinygrad](https://github.com/tinygrad/tinygrad). It is a work in progress.

**Motivation**: We plan to integrate [FSRS](https://github.com/open-spaced-repetition/fsrs4anki), a modern spaced repetition algorithm, into [Anki](https://github.com/ankitects/anki), which requires a localized optimization module to train the parameters from users' review logs.

**Background**: We have tried some Rust-based training methods like [tch-rs](https://github.com/LaurentMazare/tch-rs). But it is relied on the libtorch, whose size is about 200MB. It is too large for a spaced repetition software. So we choose tinygrad, a small deep learning framework in Python.

## Requirements

tinygrad from the latest source

## Give it a try

seq2one implementation (accurate but slow):

```shell
LAZY=0 CPU=1 python seq2one.py
```

seq2seq implementation (fast but inaccurate):


```shell
LAZY=0 CPU=1 python seq2seq.py
```

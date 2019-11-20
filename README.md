# Adversarial Training with ATN (CIFAR-10)

## Overview

Adversasrial examples can be generated via neural networks, and these special networks are called Adversarial Transformation Network. This repository verifies that ATN is useful for adversarial training.

## Demo: Adversarial Attack

### PGD Attack

```
python demo_pgd.py --device cpu
                   --pgd_type [linf, l2]
```

<p align="center">
    <img width=50% src="./md/demo1.png">
</p>

### ATN Attack

```
python demo_atn.py --device cuda
```

<p align="center">
    <img width=50% src="./md/demo2.png">
</p>

## Result: Adversarial Training

### Standard Training

<p align="center">
    <img width=50% src="./md/eps2/plot0.png"><img width=50% src="./md/eps8/plot0.png">
</p>

### PGD-Training

#### Linf-PGD

<p align="center">
    <img width=50% src="./md/eps2/plot2.png"><img width=50% src="./md/eps8/plot2.png">
</p>

#### L2-PGD

<p align="center">
    <img width=50% src="./md/eps2/plot4.png"><img width=50% src="./md/eps8/plot4.png">
</p>

### ATN-Training

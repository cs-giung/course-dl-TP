## Adversarial Training (CIFAR-10)

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

## Appendix

### Demo 01. PGD Attack

```
python demo_pgd.py --device cpu
                   --pgd_type [linf, l2]
```

<p align="center">
    <img width=50% src="./md/demo1.png">
</p>

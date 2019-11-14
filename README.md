## Adversarial Training (CIFAR-10)

### Original Classifier

#### Training

```
python train.py --device cuda
                --epochs 100
                --batch_size 32
                --lr 0.01
                --lr_decay 10

* data augmentation: RandomHorizontalFlip, RandomCrop
```

```
./weights/vgg16_e086_90.62.pth
```

#### Evaluation

```
python eval.py --device cuda
               --weight ./weights/vgg16_e086_90.pth
               --attack [None, 'fgsm', 'linf_pgd', 'l2_pgd']
```

| epoch | Original Acc. | FGSM-Attacked Acc. | PGD-Attacked Acc. |
| :-:   |  -:           |  -:                |  -:               |
| [e086](https://drive.google.com/a/korea.ac.kr/file/d/1AB8ipF9e_t0Du7W79sZtQIOFK9Q9waiQ/view?usp=sharing) | 90.73 %       | 25.09 %            |  0.55 %           |
| -     | 70.89 it/s    | 27.23 it/s         | 4.48 it/s         |

### PGD-Trained Classifier

#### PGD-Training #1 (L_inf, true label)

<p align="center">
    <img width=50% src="./md/plot1.png">
</p>

```
python train.py --device cuda
                --epochs 200
                --batch_size 32
                --lr 0.01
                --lr_decay 20
                --pgd_train linf

* data augmentation: RandomHorizontalFlip
```

| epoch | Original Acc. | FGSM-Attacked Acc. | Linf-PGD-Attacked Acc. | L2-PGD Attacked Acc. |
| :-:   |  -:           |  -:                |  -:                    | -:                   |
| e022  | 74.39 %       | 46.18 %            | 47.76 %                | 43.32 %              |
| e040  | 77.62 %       | 43.55 %            | 43.91 %                | 38.57 %              |

#### PGD-Training #2 (L_inf, predicted label)

<p align="center">
    <img width=50% src="./md/plot2.png">
</p>

| epoch | Original Acc. | FGSM-Attacked Acc. | Linf-PGD-Attacked Acc. | L2-PGD Attacked Acc. |
| :-:   |  -:           |  -:                |  -:                    | -:                   |
| e021  | 78.49 %       | 43.77 %            | 46.47 %                | 40.61 %              |
| e035  | 79.03 %       | 43.43 %            | 44.95 %                | 38.74 %              |

#### PGD-Training #3 (L_2, true label)

<p align="center">
    <img width=50% src="./md/plot3.png">
</p>

```
python train.py --device cuda
                --epochs 200
                --batch_size 32
                --lr 0.01
                --lr_decay 20
                --pgd_train l2

* data augmentation: RandomHorizontalFlip
```

| epoch | Original Acc. | FGSM-Attacked Acc. | Linf-PGD-Attacked Acc. | L2-PGD Attacked Acc. |
| :-:   |  -:           |  -:                |  -:                    | -:                   |
| e021  | 71.07 %       | 47.48 %            | 49.57 %                | 45.30 %              |
| e040  | 75.42 %       | 45.66 %            | 45.86 %                | 41.00 %              |

#### PGD-Training #4 (L_2, predicted label)

<p align="center">
    <img width=50% src="./md/plot4.png">
</p>

| epoch | Original Acc. | FGSM-Attacked Acc. | Linf-PGD-Attacked Acc. | L2-PGD Attacked Acc. |
| :-:   |  -:           |  -:                |  -:                    | -:                   |
| e021  | 71.19 %       | 47.20 %            | 48.84 %                | 45.44 %              |
| e039  | 74.28 %       | 45.99 %            | 46.56 %                | 41.90 %              |

## Appendix

### Demo 01. PGD Attack

```
python demo_pgd.py --device cpu
                   --pgd_type [linf, l2]
```

<p align="center">
    <img width=50% src="./md/demo1.png">
</p>

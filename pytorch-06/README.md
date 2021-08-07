
## 1 使用net_cnn.py
```bash
python cifar.py --model cnn --gpu_id 1
cnn : val_acc : 0.6813, val_loss : 0.3075759708881378
```


```bash
2021-08-07 03:17:48 [10,  2000] train acc: 0.889 loss: 0.308 ; val acc: 0.694 loss: 0.299  lr: 0.001
2021-08-07 03:18:07 [10,  4000] train acc: 0.869 loss: 0.356 ; val acc: 0.689 loss: 0.295  lr: 0.001
2021-08-07 03:18:26 [10,  6000] train acc: 0.863 loss: 0.387 ; val acc: 0.678 loss: 0.307  lr: 0.001
2021-08-07 03:18:46 [10,  8000] train acc: 0.856 loss: 0.414 ; val acc: 0.680 loss: 0.303  lr: 0.001
2021-08-07 03:19:06 [10, 10000] train acc: 0.848 loss: 0.431 ; val acc: 0.689 loss: 0.295  lr: 0.001
2021-08-07 03:19:25 [10, 12000] train acc: 0.844 loss: 0.445 ; val acc: 0.678 loss: 0.295  lr: 0.001
Finished Training
Accuracy of plane : 72 %
Accuracy of   car : 84 %
Accuracy of  bird : 61 %
Accuracy of   cat : 45 %
Accuracy of  deer : 56 %
Accuracy of   dog : 59 %
Accuracy of  frog : 80 %
Accuracy of horse : 72 %
Accuracy of  ship : 81 %
Accuracy of truck : 68 %
```

## 2 使用net_gap.py
```bash
python cifar.py --model gap --gpu_id 1
gap : val_acc : 0.6217, val_loss : 0.27244704961776733
```

```bash
2021-08-07 03:19:08 [10,  2000] train acc: 0.651 loss: 1.004 ; val acc: 0.616 loss: 0.272  lr: 0.001
2021-08-07 03:19:27 [10,  4000] train acc: 0.655 loss: 0.982 ; val acc: 0.650 loss: 0.253  lr: 0.001
2021-08-07 03:19:45 [10,  6000] train acc: 0.653 loss: 0.993 ; val acc: 0.629 loss: 0.264  lr: 0.001
2021-08-07 03:20:04 [10,  8000] train acc: 0.637 loss: 1.015 ; val acc: 0.633 loss: 0.265  lr: 0.001
2021-08-07 03:20:22 [10, 10000] train acc: 0.656 loss: 0.984 ; val acc: 0.634 loss: 0.259  lr: 0.001
2021-08-07 03:20:41 [10, 12000] train acc: 0.665 loss: 0.971 ; val acc: 0.623 loss: 0.268  lr: 0.001
Finished Training
Accuracy of plane : 64 %
Accuracy of   car : 57 %
Accuracy of  bird : 34 %
Accuracy of   cat : 44 %
Accuracy of  deer : 39 %
Accuracy of   dog : 63 %
Accuracy of  frog : 75 %
Accuracy of horse : 71 %
Accuracy of  ship : 80 %
Accuracy of truck : 90 %
```


## 3 net_vgg.py
```bash
python cifar.py --model vgg --gpu_id 1
vgg : val_acc : 0.8284, val_loss : 0.1539972424507141
```


```bash
2021-08-07 04:01:42 [10,  2000] train acc: 0.948 loss: 0.156 ; val acc: 0.828 loss: 0.159  lr: 0.001
2021-08-07 04:02:41 [10,  4000] train acc: 0.946 loss: 0.158 ; val acc: 0.828 loss: 0.158  lr: 0.001
2021-08-07 04:03:41 [10,  6000] train acc: 0.945 loss: 0.170 ; val acc: 0.829 loss: 0.154  lr: 0.001
2021-08-07 04:04:40 [10,  8000] train acc: 0.939 loss: 0.174 ; val acc: 0.822 loss: 0.162  lr: 0.001
2021-08-07 04:05:40 [10, 10000] train acc: 0.941 loss: 0.175 ; val acc: 0.821 loss: 0.169  lr: 0.001
2021-08-07 04:06:40 [10, 12000] train acc: 0.939 loss: 0.188 ; val acc: 0.820 loss: 0.163  lr: 0.001
Finished Training
Accuracy of plane : 84 %
Accuracy of   car : 89 %
Accuracy of  bird : 68 %
Accuracy of   cat : 77 %
Accuracy of  deer : 79 %
Accuracy of   dog : 75 %
Accuracy of  frog : 81 %
Accuracy of horse : 89 %
Accuracy of  ship : 89 %
Accuracy of truck : 93 %
```

## 4 集成方法
```
epoch:95集成模型的正确率74.17
模型0的正确率为：67.18
模型1的正确率为：68.91
模型2的正确率为：71.32
epoch:96集成模型的正确率74.51
模型0的正确率为：67.15
模型1的正确率为：69.28
模型2的正确率为：71.69
epoch:97集成模型的正确率74.47
模型0的正确率为：66.29
模型1的正确率为：68.93
模型2的正确率为：71.34
epoch:98集成模型的正确率75.12
模型0的正确率为：67.02
模型1的正确率为：69.45
模型2的正确率为：71.43
epoch:99集成模型的正确率74.89
模型0的正确率为：67.39
模型1的正确率为：69.26
模型2的正确率为：71.87
```
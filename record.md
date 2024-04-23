## 测试记录

### version 1

    1. 第一层卷积层为分组卷积，所有卷积层为边界填充

    2. 30epoch后训练得到的model在剩余haze4k数据集的300张图像和havev2的五张图像上测试的结果：("D:\pythonFile\PriorNet\snapshots\record-DehazeNet_epoch30.pth")

```
haze4k:
    ===> Avg_PSNR: 18.8345 dB 
    ===> Avg_SSIM: 0.9236 

hazev2:
    ===> AVg_PSNR: 20.3823 dB
    ===> AVg_SSIM: 0.9510
    



```

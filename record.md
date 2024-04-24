## 测试记录

### version 1

    1. 第一层卷积层为分组卷积，所有卷积层为边界填充

    2. 30epoch后训练得到的model在剩余haze4k数据集的300张图像和havev2的五张图像上测试的结果：("D:\pythonFile\PriorNet\snapshots\record-DehazeNet_epoch30.pth")


    haze4k:(之前的haze4k的300张)record-snapshots/DehazeNet_epoch30.pth
        ===> Avg_PSNR: 18.8345 dB 
        ===> Avg_SSIM: 0.9236 
    
    hazev2:(之前的NYU2的5张)record-snapshots/DehazeNet_epoch30.pth
        ===> Avg_PSNR: 20.3823 dB
        ===> Avg_SSIM: 0.9510
        
    haze4k:(完全数据集的1000张)record-snapshots/DehazeNet_epoch30.pth
        ===> Avg_PSNR: 18.8270 dB 
        ===> Avg_SSIM: 0.8606 

    <!-- haze4k:(完全数据集的1000张)record-snapshots/DehazeNet_epoch12.pth
    ===> Avg_PSNR: 18.2405 dB 
    ===> Avg_SSIM: 0.8401 -->

    haze4k:(完全数据集的1000张)record-snapshots/DehazeNet_epoch199.pth
    Avg_PSNR: 19.408277160082722 dB, Avg_SSIM: 0.8619338401625675


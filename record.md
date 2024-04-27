## 测试记录

### version 1

    1. 第一层卷积层为分组卷积，所有卷积层为边界填充（仅第一层边界填充）

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

    haze4k:(完全数据集的1000张)record-snapshots/DehazeNet_epoch199.pth net1
    Avg_PSNR: 19.408277160082722 dB, Avg_SSIM: 0.8619338401625675


### version2
    参数量从version1的18k上升到20k
    net1 copy.py batch_size = 8 version1加通道
    snapshots1/DehazeNet_epoch26.pth -> recordsnapshots/net1-copy-DehazeNet_epoch26.pth
    [2024-04-26 20:08:09.740741] test start with snapshots1/DehazeNet_epoch26.pth
    [2024-04-26 20:08:45.894079] test end with snapshots1/DehazeNet_epoch26.pth
    [2024-04-26 20:10:22.610169] Avg_PSNR: 19.294888829254795 dB, Avg_SSIM: 0.8653365157432823

    net1 copy.py batch_size = 4 version1加通道
    snapshots1/DehazeNet_epoch27.pth -> recordsnapshots/net1-copy-DehazeNet_epoch27.pth
    [2024-04-26 21:10:51.365006] test start with snapshots1/DehazeNet_epoch27.pth
    [2024-04-26 21:11:26.556437] test end with snapshots1/DehazeNet_epoch27.pth
    [2024-04-26 21:13:03.705762] Avg_PSNR: 19.162622620001997 dB, Avg_SSIM: 0.8673404486575047


    net1 copy 2.py batch_size = 4 原版 version1加通道，注意力不用加法（仍在运行dehaze找较好结果）
    [2024-04-27 13:52:55.635695] Avg_PSNR: 19.57026879088937 dB, Avg_SSIM: 0.8653976652324564

### version3
    大改注意力机制
    注意力机制卷积层采用边界填充
    

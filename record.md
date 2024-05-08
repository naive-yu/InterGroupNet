## 测试记录
    雾图水平
    [2024-05-05 20:06:52.979633] Avg_PSNR: 17.08376834398991 dB, Avg_SSIM: 0.8276328193247319
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


    net1 copy 2.py batch_size = 4 原版 version1加通道，注意力不用加法, 重新训练结果放入snapshots1 copy文件夹（仍在运行dehaze找较好结果）
    [2024-04-27 13:52:55.635695] Avg_PSNR: 19.57026879088937 dB, Avg_SSIM: 0.8653976652324564

    [2024-04-27 16:17:12.220361] test start with snapshots1/DehazeNet_epoch199.pth
    [2024-04-27 16:17:49.333954] test end with snapshots1/DehazeNet_epoch199.pth
    [2024-04-27 16:19:37.268586] Avg_PSNR: 19.840134845721987 dB, Avg_SSIM: 0.8661485987205748

### version3
    大改注意力机制
    注意力机制卷积层分组卷积
    NET3
    D:\Anaconda\python.exe D:\pythonFile\InterGroupNet\dehaze.py 
    [2024-05-04 20:52:02.019988] test start with snapshots3/DehazeNet_epoch50.pth
    [2024-05-04 20:52:14.884324] test end with snapshots3/DehazeNet_epoch50.pth
    [2024-05-04 20:53:07.092281] Avg_PSNR: 16.77854178735921 dB, Avg_SSIM: 0.8108847352564335

    21:56:加x9
    [2024-05-04 23:40:45.799232] test start with snapshots3/DehazeNet_epoch42.pth
    [2024-05-04 23:41:35.399046] test end with snapshots3/DehazeNet_epoch42.pth
    [2024-05-04 23:42:29.296977] Avg_PSNR: 18.83234544079723 dB, Avg_SSIM: 0.8518585262298584

    
    5/5/14:32:first 500 pictures submit version3.0 *** 9cbab9ca ***
    详见result3.csv 保存在snapshots3-version3.0
    net3,34,DehazeNet_epoch34.pth,18.76292264248695,0.8803929291963577

    [2024-05-05 17:12:31.025952] test start with snapshots2/DehazeNet_epoch12.pth
    [2024-05-05 17:12:50.101495] test end with snapshots2/DehazeNet_epoch12.pth
    [2024-05-05 17:13:06.041401] Avg_PSNR: 18.557426890024708 dB, Avg_SSIM: 0.8718973098993301
    之后需要关注的3点：残差网络，多头注意力，相对位置改善

    D:\Anaconda\python.exe D:\pythonFile\InterGroupNet\dehaze.py 
    [2024-05-06 00:13:27.886060] test start with snapshots2/DehazeNet_epoch158.pth
    [2024-05-06 00:14:10.667272] test end with snapshots2/DehazeNet_epoch158.pth
    [2024-05-06 00:14:24.098362] Avg_PSNR: 19.00836200499259 dB, Avg_SSIM: 0.883252209186554
    
    record-result2.csv 前500
    net2,198,DehazeNet_epoch198.pth,19.135995795382243,0.8830249757766724  
    前1500(不同尺寸泛化)：Avg_PSNR: 19.336173125425443 dB, Avg_SSIM: 0.8639760804474353
    拟合1500：Avg_PSNR: 19.704309846832924 dB, Avg_SSIM: 0.8826771590709687
    net2,184,DehazeNet_epoch184.pth,19.089449619884494,0.8856727446317673
    前1500(不同尺寸泛化)：Avg_PSNR: 18.969806954009965 dB, Avg_SSIM: 0.8621244140267372
    拟合1500：Avg_PSNR: 19.67469587178321 dB, Avg_SSIM: 0.8852835761308671

    NYU
    D:\Anaconda\python.exe D:\pythonFile\InterGroupNet\dehaze.py 
    [2024-05-07 19:13:03.458309] Avg_PSNR: 19.724768315908353 dB, Avg_SSIM: 0.8640415560603142

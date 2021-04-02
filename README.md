# dliknet、unet、deeplabv3+
整体框架为北京邮电大学的道路识别比赛，在此基础上学习，修改，添加了deeplabv3+网络，可以实现遥感数据的数据集制作，网络切换，模型训练，以及结果输出。

主体代码为train.py文件，设置好40行训练数据，和标签的路径，选择56行自己需要的网络结构，执行即可开始训练。

同时提供ToolGUI.py工具可以将遥感大图裁剪为256*256训练集，自动生成dataset文件夹。

![image](https://user-images.githubusercontent.com/43696193/113390189-cfd9df00-93c3-11eb-86b8-55baf2df4b77.png)

同时添加了网络训练过程可视化，可实时查看训练过程，代码中已经注释，需要自己选择。

![image](https://user-images.githubusercontent.com/43696193/113390629-b38a7200-93c4-11eb-818d-43e36082b80d.png)

![image](https://user-images.githubusercontent.com/43696193/113390655-bdac7080-93c4-11eb-95c3-2540bdd2bb55.png)

![image](https://user-images.githubusercontent.com/43696193/113390677-c8670580-93c4-11eb-86fb-f5560726db3f.png)

训练完毕，调用predict_best.py设置好mian后的参数，执行预测，预测采用只取中间1/4，经过测试，这样效果比较好。

![image](https://user-images.githubusercontent.com/43696193/113391153-a7eb7b00-93c5-11eb-87ca-e4053ed75030.png)![image](https://user-images.githubusercontent.com/43696193/113391172-b20d7980-93c5-11eb-9a4a-13789867cb9a.png)![image](https://user-images.githubusercontent.com/43696193/113391187-b9348780-93c5-11eb-8e57-38865c2da3af.png)

![image](https://user-images.githubusercontent.com/43696193/113391349-ff89e680-93c5-11eb-923d-f2ac560af63d.png)![image](https://user-images.githubusercontent.com/43696193/113391478-28aa7700-93c6-11eb-98d7-9c2c6f00b84b.png)![image](https://user-images.githubusercontent.com/43696193/113391498-3102b200-93c6-11eb-953b-4f19f28ed912.png)




![image](https://user-images.githubusercontent.com/43696193/113391574-4d9eea00-93c6-11eb-8a76-06afe30be3ee.png)![image](https://user-images.githubusercontent.com/43696193/113391604-5c859c80-93c6-11eb-9f94-539e7d376cd2.png)![image](https://user-images.githubusercontent.com/43696193/113391633-6a3b2200-93c6-11eb-8cfd-73315e295517.png)

分别为预测结果、影像、和标签，可以看出来网络训练不好，还是会出现棋盘现象，训练时间1080T 20个epoch，样本总数为30000张256*256.

技术交流联系方式：mrwanghongji@qq.com



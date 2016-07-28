# CS534
CS534-Machine Learning

This is the final project of CS534-Machine Learning course. 

Inside the project, our team want to classify the software release note into three categories: bug fix, add new features and performance enhancement. We used both supervised and unsupervised method to compare the accuracy difference between them.

To convert the words information into some comparable value, we use TF-IDF representation model. The supervised method being used is KNN and the unsupervised method is K-means. 

Insdie the Final Program Package, there are three .m files. Please follow the instruction to run this program:

1. Run FP_Part_1_Raw_data_modify.m

2. Run FP_Part_2_Unsupervised_Kmeans.m

3. Run FP_Part_3_Supervised_KNN.m


中文说明：
该文件是CS534 机器学习课程的期末设计。

在这个项目中，处理的数据对象是软件的更新说明文本，我们小组使用了有监督和非监督学习，将数据元分类为三类：1.Bug修复 2. 增加新功能 3. 性能提升。
我们在最后将对比监督学习和非监督学习在分类准确性上的差异。

其中，我使用TF-IDF模型将无法对比的文字转化为可以互相对比的数据。监督学习部分，使用了KNN算法；非监督学习部分，试用了K-means算法。

语言使用：MATLAB

运行指南：

1. clone Final Program 包到目标位置

2. 打开Matlab，运行FP_Part_1_Raw_data_modify.m

3. 运行FP_Part_2_Unsupervised_Kmeans.m

4. 运行FP_Part_3_Supervised_KNN.m

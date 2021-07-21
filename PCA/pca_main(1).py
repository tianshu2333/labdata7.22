import numpy as np
import glob
from sklearn import datasets
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import pandas as pd
from sklearn import decomposition
import xlrd
from sklearn import preprocessing


def excel_to_matrix(path):
    # 读取csv文件并讲其转化为矩阵输出
    rawdata = pd.read_csv(path)  # 读取csv文件
    output_raw = rawdata.values
    output = output_raw[0:180, :]
    # csv文件格式对应
    return output


class PCA():
    def calculate_covariance_matrix(self, X, Y=None):
        # 计算协方差矩阵

        m = X.shape[0]
        # .shape 表示两个维度上的数据计数，0表示纵向，1表示横向
        # 表示有多少个项目
        X = X - np.mean(X, axis=0)
        # mean()
        # 函数功能：求取均值
        # 经常操作的参数为axis，以m * n矩阵举例：
        # axis
        # 不设置值，对 m * n个数求均值，返回一个实数
        # axis = 0：压缩行，对各列求均值，返回1 * n矩阵
        # axis = 1 ：压缩列，对各行求均值，返回m * 1矩阵
        if Y == None:
            Y = X
        else:
            Y - np.mean(Y, axis=0)
            # print(Y)

        return 1 / m * np.matmul(X.T, Y)
        # 返回协方差矩阵
        # 相乘

    def transform(self, X, n_components):
        # 设n=X.shape[1]，将n维数据降维成n_component维

        covariance_matrix = self.calculate_covariance_matrix(X)

        # 获取特征值，和特征向量
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        #print(eigenvalues)
        # 返回特征值和特征向量
        # 对特征向量排序，并取最大的前n_component组
        idx = eigenvalues.argsort()[::-1]
        # argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引号)
        # 当num>=0时，np.argsort()[num]就可以理解为y[num];
        # 当num<0时，np.argsort()[num]就是把数组y的元素反向输出，例如np.argsort()[-1]即输出x中最大值对应的index，np.argsort()[-2]即输出x中第二大值对应的index，依此类推。
        # 此处用来从大到小输出对应的特征值的大小
        eigenvectors = eigenvectors[:, idx]
        # 取最大特征值所在列为向量
        eigenvectors = eigenvectors[:, :n_components]
        #
        # 转换
        return eigenvectors, eigenvalues
        # return np.matmul(X, eigenvectors)


# 主函数
def main():
    # Demo of how to reduce the dimensionality of the data to two dimension
    # and plot the results.

    # Load the dataset
    # data = datasets.load_digits()
    # X = data.data
    # y = data.target
    # z = len(np.unique(y))
    # print("y",z)
    # for files in range(0, 3):
    #     i = files
    #     # print(i)
    #     a = glob.glob('C:/Users/lhy12/Desktop/PCA/*.csv')
    #     strlist1 = a[i].split('')
    #     strlist2 = strlist1
    #     print(strlist)
    #     X = strlist[5]
    for files in range(0,5):
        filenames = ['C:/Users/lhy12/Desktop/PCA/merge/HA_HpA_merge.csv','C:/Users/lhy12/Desktop/PCA/merge/HA_merge.csv','C:/Users/lhy12/Desktop/PCA/merge/HA_OA_merge.csv','C:/Users/lhy12/Desktop/PCA/merge/HpA_merge.csv','C:/Users/lhy12/Desktop/PCA/merge/HpA_OA_merge.csv']
        print(files)
        X = excel_to_matrix(filenames[files])
        #print(filenames[files])
        # Project the data onto the 2 primary principal components
        X_trans,accuracy = PCA().transform(X, 2)
        precision =(accuracy[0]+accuracy[1])/accuracy.sum()
        print(filenames[files],"Accuracy:{:.2%}".format(precision))
        # print(X_trans)
        X_resu = np.matmul(X, X_trans)
        # print(X_resu)
        x1 = X_trans[:, 0]
        x2 = X_trans[:, 1]
        # print(X_trans[0, 0], X_trans[0, 1])
        # print(x1, x2)
        # cmap = plt.get_cmap('viridis')
        colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']
        for i in range(1, 10):
            if files == 0:
                a, = plt.plot(X_trans[i - 1, 0], X_trans[i - 1, 1], 'o', color='blue')

                #print('a')
            elif files == 1:
                b, = plt.plot(X_trans[i - 1, 0], X_trans[i - 1, 1], 'o', color='yellow')
                #print('b')
            elif files == 2:
                c, = plt.plot(X_trans[i - 1, 0], X_trans[i - 1, 1], 'o', color='black')
                #print('b')
            elif files == 3:
                d, = plt.plot(X_trans[i - 1, 0], X_trans[i - 1, 1], 'o', color='red')
                #print('b')
            elif files == 4:
                e, = plt.plot(X_trans[i - 1, 0], X_trans[i - 1, 1], 'o', color='orange')
                #print('b')
        print('over')
            # elif (i-1)%3 == 2:
            #   c, = plt.plot(X_trans[i-1,0], X_trans[i-1,1], 'o', color='red')
            #   print('c')
            # if (i-1)%6 == 3:
            #   d, = plt.plot(X_trans[i-1,0], X_trans[i-1,1], 'o', color='lime')
            #   print('d')
            # if (i-1)%6 == 4:
            #   e, = plt.plot(X_trans[i-1,0], X_trans[i-1,1], 'o', color='cyan')
            #   print('e')
            #   #print(i)
            # if (i-1)%6 == 5:
            #   f, = plt.plot(X_trans[i-1,0], X_trans[i-1,1], 'o', color='orange')
            #   print('f')
            #   print(i)
            # if i == 49:
            #   plt.suptitle("PCA Dimensionality Reduction")
            #   plt.title("Digit Dataset")
            #   plt.xlabel('Principal Component 1')
            #   plt.ylabel('Principal Component 2')
            #   plt.show()
        #plt.legend([a, b, c, d, e], ['HA', 'HA_HpA','HA_OA','HpA','HpA_OA'])
        plt.suptitle("PCA Dimensionality Reduction")
        plt.title("Digit Dataset")
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        # plt.figure(figsize=(60,40))
        # plt.show()
        plt.savefig('fig_cat.png')

        # 绘图
        x_0 = []
        y_0 = []
        # for i in range (0,np.size(x1)):
        #     x_0[i] = x1[i]
        #     y_0[i] = x2[i]
        # plt.plot(x1, x2, 'o', color='b')


    # class_distr = []
    # # Plot the different class distributions
    # for i, l in range(len(colors)):
    #     px = x1[y_digits == i]
    #     py = x2[y_digits == i]
    #     class_distr.append(plt.scatter(_x1, _x2, color=colors[i]))

    # Add a legend
    # plt.legend(class_distr, 180, loc=1)

    # Axis labels


if __name__ == "__main__":
    main()
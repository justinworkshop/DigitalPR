import matplotlib.pyplot as plt


def showLabel(res, resLable, index):
    print("text count:%d, lable count:%d"%(res.size, resLable.size))
    print(res[index])       #是一个包含784个元素且值在[0,1]之间的向量
    print(resLable[index])
    
def showImage(res, count):
    fig, ax = plt.subplots(nrows=int(count/5),ncols=5,sharex='all',sharey='all')
    ax = ax.flatten()
    for i in range(count):
        img = res[i].reshape(28, 28)
        ax[i].imshow(img,cmap='Greys')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()
    

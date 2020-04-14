import matplotlib.pyplot as plt

debug = False


def showLabel(res, resLable, index):
    if not debug:
        return

    print("text count:%d, lable count:%d" % (res.size, resLable.size))
    print(res[index])       # 是一个包含784个元素且值在[0,255]之间的向量
    print(resLable[index])


def showImage(res, count):
    if not debug:
        return

    cNum = 5
    rNum = int(count/cNum)
    fig, ax = plt.subplots(nrows=rNum, ncols=cNum, sharex='all', sharey='all')
    ax = ax.flatten()
    for i in range(count):
        img = res[i].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()

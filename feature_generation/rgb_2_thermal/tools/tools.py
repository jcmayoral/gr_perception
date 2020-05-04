import matplotlib.pyplot as plt

def visualize(rgb_imgs, thermal_imgs):
    r,c = rgb_imgs.shape[0],2
    titles = ['Condition','Original']
    #plt.figure(figsize=(5,5))
    fig, axs = plt.subplots(r, c,figsize=[20,20])
    cnt = 0
    for i in range(r):
        axs[i,0].imshow(rgb_imgs[i])
        axs[i,1].imshow(thermal_imgs[i][:,:,0],cmap="hot")
        for j in range(c):
            axs[i, j].set_title(titles[j])
            axs[i,j].axis('off')
        #fig.savefig("images_2020_04_13_2nd_Arch/{}/{}/image.png".format(epoch,batch_i))
        #plt.close()
    plt.show()
    plt.close()

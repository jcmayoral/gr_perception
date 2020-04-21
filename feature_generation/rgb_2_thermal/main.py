from data_loader import DataLoader
import matplotlib.pyplot as plt


num_imgs = 4

data_loader = DataLoader("test", img_res=(128, 128),
             rgb_dataset_folder="/home/jose/Pictures",
             thermal_dataset_folder="/home/jose/Pictures")
rgb_imgs, thermal_imgs = data_loader.load_samples(num_imgs=num_imgs,thermal_ext=".jpg")

r,c = num_imgs,2
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

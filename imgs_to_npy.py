import numpy as np
import os
from PIL import Image

def imgs_to_npy(dataloc,savefile):
    sfile = []
    for i in range (len(os.listdir(dataloc))):
        pixdata = np.array(Image.open(dataloc + str(i) + ".png").getdata()).T
        pixdata = np.delete(pixdata,1,0)
        sfile.append(pixdata[0] / 255)
    npyfile = np.array(sfile)
    np.save(savefile,npyfile)

def txt_to_npy(txtfile,savefile):
    data = np.array([open(txtfile).read().split('\n')])
    data = data.astype(int)
    sfile = np.zeros((data.shape[1],10),int)
    for i in range(sfile.shape[0]):
        sfile[i][data[0][i]] = 1
    np.save(savefile,sfile)

def main():
    imgs_to_npy("MNIST/training_data/","TrainingData.npy")
    imgs_to_npy("MNIST/test_data/","TestData.npy")
    txt_to_npy("MNIST/training_labels.txt","TrainingLabel.npy")
    txt_to_npy("MNIST/test_labels.txt","TestLabel.npy")
    print("Done.")

if __name__ == "__main__":
    main()
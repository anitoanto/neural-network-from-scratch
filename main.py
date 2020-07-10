
import numpy as np

def sigmoid(arr,deriv = False):
	arr = np.clip(arr,-100,100)
	temp = 1 / (1 + np.exp(-arr))
	if deriv == False:
		return temp
	else:
		return temp * (1 - temp)

def initmodel(nx,nh1,nh2,ny):
    w1 = np.random.standard_normal((nh1,nx))
    b1 = np.zeros((nh1,1))
    w2 = np.random.standard_normal((nh2,nh1))
    b2 = np.zeros((nh2,1))
    w3 = np.random.standard_normal((ny,nh2))
    b3 = np.zeros((ny,1))
    return [w1,b1,w2,b2,w3,b3,nx,nh1,nh2,ny]

def trainbatch(x,y,modeldata,batch,batchsize):
    (w1,b1,w2,b2,w3,b3,nx,nh1,nh2,ny) = modeldata
    dw1 = np.zeros((nh1,nx))
    db1 = np.zeros((nh1,1))
    dw2 = np.zeros((nh2,nh1))
    db2 = np.zeros((nh2,1))
    dw3 = np.zeros((ny,nh2))
    db3 = np.zeros((ny,1))
    for i in range(batch * batchsize, (batch + 1) * batchsize):
        xx = x[i].reshape((nx,1))
        yy = y[i].reshape((ny,1))
        h1 = sigmoid(np.dot(w1 + dw1,xx) + (b1 + db1))
        zh2 = np.dot(w2 + dw2,h1) + (b2 + db2)
        h2 = sigmoid(zh2)
        zh3 = np.dot(w3 + dw3,h2) + (b3 + db3)
        h3 = sigmoid(zh3)
        fac1 = 2 * (h3 - yy) * sigmoid(zh3,deriv = True)
        fac2 = np.dot(w3.T,fac1) * sigmoid(zh2,deriv = True)
        fac3 = np.dot(w2.T,fac2)
        dw3 -= np.dot(fac1,h2.T)
        db3 -= fac1
        dw2 -= np.dot(fac2,h1.T)
        db2 -= fac2
        dw1 -= np.dot(fac3,xx.T)
        db1 -= fac3
    w1 += dw1 / batchsize
    b1 += db1 / batchsize
    w2 += dw2 / batchsize
    b2 += db2 / batchsize
    w3 += dw3 / batchsize
    b3 += db3 / batchsize
    return w1,b1,w2,b2,w3,b3,nx,nh1,nh2,ny

def testmodel(modeldata,testx,testy):
    (w1,b1,w2,b2,w3,b3,nx,nh1,nh2,ny) = modeldata
    temp = 0
    for i in range(testx.shape[0]):
        xx = testx[i].reshape((nx,1))
        yy = testy[i].reshape((ny,1))
        h1 = sigmoid(np.dot(w1,xx) + b1)
        h2 = sigmoid(np.dot(w2,h1) + b2)
        h3 = sigmoid(np.dot(w3,h2) + b3)
        if np.argmax(h3) == np.argmax(yy):
            temp += 1
    return (temp / testx.shape[0]) * 100

def trainmodel(modeldata,nepoch,nbatch,resume = False):

    x = np.load("TrainingData.npy")
    y = np.load("TrainingLabel.npy")
    testx = np.load("TestData.npy")
    testy = np.load("TestLabel.npy")

    bestacc = 0
    acc = 0
    startepoch = 0

    if resume == True:
        resumedata = open("ResumeTrainData/ResumeData.txt").readlines()[0].split(':')
        startepoch = int(resumedata[1])
        print("\nCompleted epoch : " + str(startepoch) + "  |  Accuracy : " + str(resumedata[3]) + "%\n")
        print("Resuming training...\n")
        modeldata[0] = np.load("ResumeTrainData/w1.npy")
        modeldata[1] = np.load("ResumeTrainData/b1.npy")
        modeldata[2] = np.load("ResumeTrainData/w2.npy")
        modeldata[3] = np.load("ResumeTrainData/b2.npy")
        modeldata[4] = np.load("ResumeTrainData/w3.npy")
        modeldata[5] = np.load("ResumeTrainData/b3.npy")
        bestacc = float(resumedata[5])
        acc = float(resumedata[3])

    batchsize = int(x.shape[0] / nbatch)

    for e in range(nepoch):

        for batch in range(nbatch):
            modeldata = trainbatch(x,y,modeldata,batch,batchsize)
        acc = round(testmodel(modeldata,testx,testy),3)

        np.save("ResumeTrainData/w1.npy",modeldata[0])
        np.save("ResumeTrainData/b1.npy",modeldata[1])
        np.save("ResumeTrainData/w2.npy",modeldata[2])
        np.save("ResumeTrainData/b2.npy",modeldata[3])
        np.save("ResumeTrainData/w3.npy",modeldata[4])
        np.save("ResumeTrainData/b3.npy",modeldata[5])

        if acc > bestacc:
            bestacc = acc
            data = "BestEpoch:" + str(e + 1 + startepoch) + ":Accuracy:" + str(bestacc)
            open("TrainedModelData/ModelData.txt",'w').write(data)
            np.save("TrainedModelData/w1.npy",modeldata[0])
            np.save("TrainedModelData/b1.npy",modeldata[1])
            np.save("TrainedModelData/w2.npy",modeldata[2])
            np.save("TrainedModelData/b2.npy",modeldata[3])
            np.save("TrainedModelData/w3.npy",modeldata[4])
            np.save("TrainedModelData/b3.npy",modeldata[5])
        
        data = "CompletedEpoch:" + str(startepoch + e + 1) + ":Accuracy:" + str(acc) + ":BestAccuracy:" + str(bestacc)
        open("ResumeTrainData/ResumeData.txt",'w').write(data)

        print("Epoch : " + str(e + 1 + startepoch) + "  |  Accuracy : " + str(acc) + "%")

def FASTtrainmodel(modeldata,nepoch,nbatch,resume = False):

    x = np.load("TrainingData.npy")
    y = np.load("TrainingLabel.npy")
    testx = np.load("TestData.npy")
    testy = np.load("TestLabel.npy")

    batchsize = int(x.shape[0] / nbatch)

    startepoch = 0
    if resume == True:
        resumedata = open("FastSaves/Details.txt").readlines()[0].split(':')
        startepoch = int(resumedata[1])
        modeldata[0] = np.load("FastSaves/w1.npy")
        modeldata[1] = np.load("FastSaves/b1.npy")
        modeldata[2] = np.load("FastSaves/w2.npy")
        modeldata[3] = np.load("FastSaves/b2.npy")
        modeldata[4] = np.load("FastSaves/w3.npy")
        modeldata[5] = np.load("FastSaves/b3.npy")

    for e in range(nepoch):
        for batch in range(nbatch):
            modeldata = trainbatch(x,y,modeldata,batch,batchsize)
        print("Epoch : " + str(e + 1 + startepoch))

    acc = round(testmodel(modeldata,testx,testy),3)
    data = "CompletedEpoch:" + str(e + 1 + startepoch) + ":Accuracy:" + str(acc)
    open("FastSaves/Details.txt",'w').write(data)
    np.save("FastSaves/w1.npy",modeldata[0])
    np.save("FastSaves/b1.npy",modeldata[1])
    np.save("FastSaves/w2.npy",modeldata[2])
    np.save("FastSaves/b2.npy",modeldata[3])
    np.save("FastSaves/w3.npy",modeldata[4])
    np.save("FastSaves/b3.npy",modeldata[5])

mymodel = initmodel(784,16,16,10)
FASTtrainmodel(mymodel,5,60)

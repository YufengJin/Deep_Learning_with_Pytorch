import torch.nn as nn
import torch 
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from make_training_data import DogsVSCats
#build neural network
# all test and train images is 50*50 pixel

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,32,5) # input is 1 image, 32 output channels, 5x5 kernel / window
        self.conv2 = nn.Conv2d(32,64,5) # input is 32, bc the first layer output 32. Then we say the output will be 64 channels, 5x5 conv
        self.conv3 = nn.Conv2d(64,128,5)
        
        # here we implement get_dimension() to get dimension of linear function
   
        # 2 linear layers, shrinking dimensions of output, final output is 2-dimensions for classification
        self.fc1 = nn.Linear(128*2, 512) #flattening.
        self.fc2 = nn.Linear(512, 2) # 512 in, 2 out bc we're doing 2 classes (dog vs cat).


    def forward(self, x):
    	# three convoluitonal layers
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        # two fully connection layers
        x = x.view(-1, 128*2)  # .view is reshape ... this flattens X before 
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # bc this is our output layer. No activation here.
        return F.softmax(x, dim=1)

def test_sample_training_data(training_data,image_index):

	#test the .npy file(datasets), if loaded correctly
	plt.imshow(training_data[image_index][0],cmap ="gray")
	plt.show()
	print(training_data[image_index][1])

def get_dimension():
	# calculate the dimension between convs and linear layer, here our tested images are 50*50 pixels
	x = torch.randn(50,50).view(-1,1,50,50)
	layer1 = nn.Conv2d(1,32,5) #one image input, padding size: 5, 32 output channels
	layer2 = nn.Conv2d(32,64,5)
	layer3 = nn.Conv2d(64,128,5)
	layer4 = nn.Linear(128*2,512)
	layer5 = nn.Linear(512,2)
	x = F.max_pool2d(F.relu(layer1(x)),(2,2))
	print(x.shape)
	x = F.max_pool2d(F.relu(layer2(x)),(2,2))
	print(x.shape)
	x = F.max_pool2d(F.relu(layer3(x)),(2,2))
	print(x.shape)
	x = x.view(-1,128*2)
	print(x.shape)
	x = F.relu(layer4(x))
	print(x.shape)
	x = layer5(x)
	print(x.shape)

if __name__ == "__main__":

	#make training dataset
	#dogsvscats = DogsVSCats()
	#dogsvscats.make_training_data()
	
	#pickle is a more efficient way to process data, cos it's in binary
	training_data = np.load("training_data.npy",allow_pickle=True)

	#test_sample_training_data(training_data,2)
	net = Net()

	# set Adam optimizer 
	optimizer = optim.Adam(net.parameters(),lr=0.001)
	# loss funtion
	loss_function = nn.MSELoss()

	# the torch size of X is [24946, 50, 50], contain all images, y contains all labels
	X = torch.Tensor([i[0] for i in training_data]).view(-1,50,50)
	X = X/255.0
	y = torch.Tensor([i[1] for i in training_data])

	TestInPercent = 0.1  # lets reserve 10% of our data for validation
	Test_size = int(len(X)*TestInPercent)

	# contribute all datasets
	train_X = X[:-Test_size]
	train_y = y[:-Test_size]
	
	test_X = X[-Test_size:]
	test_y = y[-Test_size:]

	print("Training data: {}\nTest data: {}\n".format(len(train_X), len(test_X)))

	BATCH_SIZE = 100
	EPOCHS = 1

	for epoch in range(EPOCHS):
		for i in tqdm(range(0, len(train_X), BATCH_SIZE)): # from 0, to the len of x, stepping BATCH_SIZE at a time. [:50] ..for now just to dev
			print(f"{i}:{i+BATCH_SIZE}")
			batch_X = train_X[i:i+BATCH_SIZE].view(-1,1,50,50)
			batch_y = train_y[i:i+BATCH_SIZE]
			#initialize gradient to 0 iterally
			net.zero_grad()
			outputs = net(batch_X)
			loss = loss_function(outputs, batch_y)
			loss.backward()
			optimizer.step()   # Does the update

	correct = 0
	total = 0
	with torch.no_grad():
	    for i in tqdm(range(len(test_X))):
	        real_class = torch.argmax(test_y[i])
	        net_out = net(test_X[i].view(-1, 1, 50, 50))[0]  # returns a list, 
	        predicted_class = torch.argmax(net_out)

	        if predicted_class == real_class:
	            correct += 1
	        total += 1
	print("Accuracy: ", round(correct/total, 3))
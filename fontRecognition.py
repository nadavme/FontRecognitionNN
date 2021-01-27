# Nadav Meidan I.D. 200990240

# ------------------imports----------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import h5py
import cv2
import os
import gc
import csv


# Filter harmless warnings
import warnings
warnings.filterwarnings("ignore")


# ------------------Global Variables-------------------------------

#Paths

trainPath = '/content/gdrive/MyDrive/Colab Notebooks/intro2ComputerVision/SynthText.h5'
validationPath = '/content/gdrive/MyDrive/Colab Notebooks/intro2ComputerVision/SynthText_val.h5'
testPath = '/content/gdrive/MyDrive/Colab Notebooks/intro2ComputerVision/Final Project/test.h5'
dataPath = '/content/gdrive/MyDrive/Colab Notebooks/intro2ComputerVision/Data'
pathToModel = '/content/gdrive/MyDrive/Colab Notebooks/intro2ComputerVision/NadavMeidan-Model.h5'
pathToCSVfile = '/content/gdrive/MyDrive/Colab Notebooks/intro2ComputerVision/testPredictionsNadavMeidann.csv'


# Parameters

max_epochs = 100
n_epochs = 32
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.00095
momentum = 0.9 
log_interval = 60 


#----------------------------------------------------------------------------
#-------------Train and validation--------------------------------------------
#----------------------------------------------------------------------------


# ------------------Preprocessing the data-------------------------------


def preProcessData(dataPath, charListImgs, charListLables):

    file_name = dataPath
    db = h5py.File(file_name, 'r')
    im_names = list(db['data'].keys())
    # The charList holds tuples of images and their lables
    
    for im in im_names:
        PER = 128
        # image
        img = np.array(db['data'][im])
        # display(img, im)  # debug
        font = db['data'][im].attrs['font']
        # chars boxes
        char_bb = db['data'][im].attrs['charBB']
        # print('number of char in image = {}'.format(char_bb.shape[2]))
        for i in range((char_bb.shape[2])):

            # char image - perspective transform
            pts1 = np.float32([char_bb[:, :, i].T[0], char_bb[:, :, i].T[1], char_bb[:, :, i].T[3], char_bb[:, :, i].T[2]])
            pts2 = np.float32([[0, 0], [PER, 0], [0, PER], [PER, PER]])
            if font[i] ==  b'Skylark':
              lable = 0.0   
            elif font[i] ==  b'Ubuntu Mono':
              lable = 1.0 
            elif font[i] ==  b'Sweet Puppy':
              lable = 2.0 
            else:
              print('something went wrong! in file name = {}'.format(im))
              exit()
            m = cv2.getPerspectiveTransform(pts1, pts2)
            dst = cv2.warpPerspective(img, m, (PER, PER))
            charListImgs.append(dst)
            charListLables.append(lable)
      





#--------------------------------------------------------------------
def writeToFolder(charListImgs, charListLables, whichData):

    dataPath = '/content/gdrive/MyDrive/Colab Notebooks/intro2ComputerVision/Data'
    if not os.path.exists(dataPath):
        os.mkdir(dataPath) 
    
    if whichData == 'train': 
        PATH = os.path.join(dataPath, 'trainData')
        if not os.path.exists(PATH):
          os.mkdir(PATH)
        os.mkdir(os.path.join(PATH, 'Skylark'))
        os.mkdir(os.path.join(PATH, 'Ubuntu Mono'))
        os.mkdir(os.path.join(PATH, 'Sweet Puppy'))
    
    if whichData == 'validation': 
        PATH = os.path.join(dataPath, 'valData')
        if not os.path.exists(PATH):
            os.mkdir(PATH)
        os.mkdir(os.path.join(PATH, 'Skylark'))
        os.mkdir(os.path.join(PATH, 'Ubuntu Mono'))
        os.mkdir(os.path.join(PATH, 'Sweet Puppy'))

    for i in range(len(charListImgs)):
        img = charListImgs[i]

        if charListLables[i] ==  0.0:
          path = os.path.join(PATH, 'Skylark')
          cv2.imwrite(os.path.join(path , str(i)+ '.jpg'), img)
        elif charListLables[i] ==  1.0 :
          path = os.path.join(PATH, 'Ubuntu Mono')
          cv2.imwrite(os.path.join(path , str(i) + '.jpg'), img) 
        elif charListLables[i] ==  2.0 :
          path = os.path.join(PATH, 'Sweet Puppy')
          cv2.imwrite(os.path.join(path , str(i)+ '.jpg'), img)  


#--------------------------------------------------------------------


def loadTheData(trainPath, validationPath):

    charListImgsTrain = []
    charListLablesTrain = []
    charListImgsVal = []
    charListLablesVal = []
    
    preProcessData(trainPath, charListImgsTrain, charListLablesTrain)
    preProcessData(validationPath, charListImgsVal, charListLablesVal)
    
    writeToFolder(charListImgsTrain, charListLablesTrain, 'train')
    writeToFolder(charListImgsVal, charListLablesVal, 'validation')
    
    train_transform = transforms.Compose([

        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    validation_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    
    train_data = datasets.ImageFolder(os.path.join(dataPath, 'trainData'), transform=train_transform)
    validation_data = datasets.ImageFolder(os.path.join(dataPath, 'valData'), transform=validation_transform)
    
    torch.manual_seed(42)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=64, shuffle=True)
    
    
    return train_loader, validation_loader


#--------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, inputs, labels):
        'Initialization'
        self.labels = labels
        self.inputs = inputs
        self.make_data_np()
        # self.make_data_torch()

        self.length = self.inputs_np.shape[0]
    
 
  def make_data_np(self):
      self.inputs_np = np.array(self.inputs)
      self.inputs_np = np.transpose(self.inputs_np, axes = (0,3,1,2,))
      self.labels_np = np.array(self.labels)
      gc.collect()
  
  

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.inputs)

  def __getitem__(self, index):
        input = torch.tensor(self.inputs_np[index]).type(torch.cuda.FloatTensor)
        lable = torch.tensor(self.labels_np[index]).type(torch.cuda.FloatTensor)
        return input, lable

#--------------------------------------------------------------------

def useGpu():
    use_cuda = torch.cuda.is_available()
    # CUDA for PyTorch
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True


#--------------------------------------------------------------------

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 512, kernel_size=3 ,padding = (1,1))
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, padding = (1,1))
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, padding = (1,1))
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(32768, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 3)
        self.batchNorm1 = nn.BatchNorm2d(512)
        self.batchNorm2 = nn.BatchNorm2d(256)
        self.batchNorm3 = nn.BatchNorm2d(128)
        # self.noise = GaussianNoise()
    def forward(self, x):

        x = F.dropout(F.relu(self.batchNorm1(self.conv1(x))),0.25)
        x = F.max_pool2d(x , 2)
        x = F.dropout(F.relu(self.batchNorm2(self.conv2(x))),0.15)
        x = F.max_pool2d(x , 2)
        x = F.dropout(F.relu(self.batchNorm3(self.conv3(x))),0.3)
        x = F.max_pool2d(x , 2)
        x = x.view(-1, 32768)
        x = F.dropout(F.relu(self.fc1(x)),0.4)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x)
    



#--------------------------------------------------------------------


def train(network, epoch, train_loader, optimizer):
    #For analizing the model
    train_losses = []
    train_counter = []
    test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]


    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device='cuda')
        target = target.to(device='cuda')

        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target.long())
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
              epoch, batch_idx * len(data), len(train_loader.dataset),
              100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
            (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))

#--------------------------------------------------------------------

def valid(network, validation_loader):
    #For analizing the model
    validation_losses = []
    accuracy_counter = []
    
    network.eval()
    validation_loss = 0
    correct = 0
    with torch.no_grad():
      for data, target in validation_loader:
        data = data.to(device='cuda')
        target = target.to(device='cuda')
        output = network(data)
        validation_loss += F.nll_loss(output, target.long(), size_average=False).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
    validation_loss /= len(validation_loader.dataset)
    validation_losses.append(validation_loss)
    print('\nValidation set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
      validation_loss, correct, len(validation_loader.dataset),
      100. * correct / len(validation_loader.dataset)))
    accuracy_counter.append(100. * correct / len(validation_loader.dataset))

#--------------------------------------------------------------------

def trainTheModel(network, train_loader, validation_loader, optimizer):
    #Add the perproccesing of the data and network
    #Send all the network, loader and shit
    for epoch in range(1, n_epochs + 1):
      train(network,epoch, train_loader, optimizer)
      valid(network, validation_loader)
      torch.save(network.state_dict(), '/content/gdrive/MyDrive/Colab Notebooks/intro2ComputerVision/myTrainedModelSugar.h5')


def trainAndValidateModel(network, trainPath, validationPat):
    train_loader, validation_loader = loadTheData(trainPath, validationPath)
    if torch.cuda.is_available():
      network.cuda()
    optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate,
                          momentum=momentum)
    trainTheModel(network, train_loader, validation_loader, optimizer)
    
    
#----------------------------------------------------------------------------
#-------------Train and validation--------------------------------------------
#----------------------------------------------------------------------------


#-----------------------Utils fuctions----------------------------------------

def writeBatchToCSV(predictionsPerWord, counterCSV, pathToCSVfile, word, im):
  with open(pathToCSVfile, 'a', newline='') as file:
      writer = csv.writer(file)
      for k in range(len(predictionsPerWord)):
            counterCSV += 1
            # print(counterCSV)
          # print(word[k])
            if predictionsPerWord[k] ==  0:
                writer.writerow([counterCSV, im, word[k], "1.0", "0.0", "0.0"])
            elif predictionsPerWord[k] ==  1:
                writer.writerow([counterCSV, im, word[k], "0.0", "1.0", "0.0"])
            elif predictionsPerWord[k] ==  2:
                writer.writerow([str(counterCSV), im, word[k], "0.0", "0.0", "1.0"])
      return counterCSV



# Function to find most frequent element in a list 
def most_frequent(List): 
    return max(set(List), key = List.count) 


def validatePredictions(predictionsPerWord): 
  #Here i use the facr that any char in the word has the same font.
      most = most_frequent(predictionsPerWord)
      for i in range(len(predictionsPerWord)):
        if predictionsPerWord[i] != most:
          predictionsPerWord[i] = most
      return predictionsPerWord

def prdeictChar(network, input):
      with torch.no_grad():
        # make sure of the device type
        if torch.cuda.is_available():
            input = input.to(device='cuda')
            network.cuda()
        # print(input.shape)
        output = network(input)
        prediction = output.data.max(1, keepdim=True)[1]
        # print('output:' , output, 'pred: ', prediction)
        return prediction.item()

def loadToNetwork(batchChars, network):
      test_transform = transforms.Compose([
              transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225])
              ])
      # print('iam batchChars:', batchChars[-1].shape)
      predictions = []
      for char in batchChars:
          char = test_transform(char)
          char = char.view(1, 3, 128, 128)
          predictions.append(prdeictChar(network, char))
      return predictions
  
#-----------------------Flow fuction of the test--------------------------------

def testTheModel(network, testPath):
    file_name = testPath
    db = h5py.File(file_name, 'r')
    # The testImagesNamesList holds tuples of images
    testImagesNamesList = list(db['data'].keys())
    
    with open(pathToCSVfile, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["SN", "image", "Char", "b'Skylark'", "b'Sweet Puppy'", "b'Ubuntu Mono'"])
    
    network.load_state_dict(torch.load(pathToModel))
    network.eval()
    counterCSV = 0;
    for im in testImagesNamesList:
        
        counter = 0
        PER = 128 #An image of 128*128 pixels
        img = np.array(db['data'][im])# image
        # chars boxes
        char_bb = db['data'][im].attrs['charBB']
        ammountOfWordsInImage = db['data'][im].attrs['wordBB'].shape[2]
        wordsInImage = list(db['data'][im].attrs['txt'])
        
        for i in range(ammountOfWordsInImage):
          word = wordsInImage[i].decode('utf-8')
          batchChars = []
          
          for j in range(len(word)):
              # char image - perspective transform
              pts1 = np.float32([char_bb[:, :, counter].T[0], char_bb[:, :, counter].T[1], char_bb[:, :, counter].T[3], char_bb[:, :, counter].T[2]])
              pts2 = np.float32([[0, 0], [PER, 0], [0, PER], [PER, PER]])
    
              m = cv2.getPerspectiveTransform(pts1, pts2)
              dst = cv2.warpPerspective(img, m, (PER, PER))
              
              counter += 1
              batchChars.append(dst)

        #Predict a a font of one word from the image at a time.
          predictionsPerBatch = loadToNetwork(batchChars, network)
          validatedPredictions = validatePredictions(predictionsPerBatch)
          counterCSV = writeBatchToCSV(validatedPredictions, counterCSV, pathToCSVfile, word, im)

        
  
def main():
    # useGpu()
    network = Net()
    # trainAndValidateModel(network, trainPath, validationPath)
    testTheModel(network, testPath)
    
if __name__ == "__main__":
    # execute only if run as a script
    main()
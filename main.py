import PySimpleGUI as sg
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.transforms as transforms
from PIL import Image
#import cv2

import pytesseract
from PIL import Image

#pytesseract.pytesseract.tesseract_cmd = r'</Users/alexwood/PycharmProjects/camera/venv/lib/python3.10/site-packages/pytesseract>'

import numpy as np

#image = Image.open('Black.png')
#text = pytesseract.image_to_string(image)
#print(text)




import io
import os
import PySimpleGUI as sg
from PIL import Image






#neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = torch.relu(nn.MaxPool2d(2)(self.conv1(x)))
        x = torch.relu(nn.MaxPool2d(2)(self.conv2(x)))
        x = x.view(-1, 320)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
mnist_train = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
mnist_test = datasets.MNIST(root="./data", train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=64, shuffle=True)

model = Net()
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)


prediction1 = None
prediction2 = None




#train the model
def train_model():
    global model
    # Train the model

    #model.load_state_dict(torch.load("my_model.pth"))
    x = 1
    for epoch in range(150):
        train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=64, shuffle=True)
        print("working" + str(x))
        x = x + 1
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    #Save the model
    torch.save(model.state_dict(), "my_model.pth")

# Function to use the trained model
def use_model():
    global model
    global img
    global prediction1
    global prediction2
    file_types = [("JPEG (*.jpg)", "*.jpg"),
                  ("All files (*.*)", "*.*")]
    layout = [
        [sg.Image(key="-IMAGE-")],
        [
            sg.Text("Image File"),
            sg.Input(size=(25, 1), key="-FILE-"),
            sg.FileBrowse(file_types=file_types),
            sg.Button("Load Image"),
        ],
    ]

    window = sg.Window("Image Viewer", layout)
    filename = None

    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        if event == "Load Image":
            filename = values["-FILE-"]
            if os.path.exists(filename):
                image = Image.open(values["-FILE-"])
                image.thumbnail((400, 400))
                bio = io.BytesIO()
                # Actually store the image in memory in binary
                image.save(bio, format="PNG")
                # Use that image data in order to
                window["-IMAGE-"].update(data=bio.getvalue())

    window.close()
    img = image
    # Load the trained model
    #model.load_state_dict(torch.load("my_model.pth"))
    #print("Accuracy: %d %%" % (100 * correct / total))
    #return predicted
    img = img.resize((28, 28))

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),])



    #transform image
    img_tensor = transform(img)

    gray_img = torch.sum(img_tensor, dim=0, keepdim=True) / 3
    img_tensor = gray_img.reshape(1, 1, 28, 28)

    # Use the model to make a prediction
    model.load_state_dict(torch.load("my_model.pth"))
    model.eval()
    output = model(img_tensor)
    #print(output.data)
    _, predicted = torch.max(output.data, 1)

    return predicted




# interactive program
while True:
    layout = [[sg.Text("Do you want to train the model or use the trained model?")],
              [sg.Button("Train"), sg.Button("Save variable #1"), sg.Button("Save variable #2"), sg.Button("Calculate"), sg.Button("Exit")]]

    window = sg.Window("Model Selector", layout)
    event, values = window.read()
    window.close()




    if event == "Exit" or event == sg.WIN_CLOSED:
        break

    if event == "Train":
        train_model()
        sg.popup("Model trained successfully!")

    if event == "Save variable #1":
        prediction1 = use_model()

    if event == "Save variable #2":
        prediction2 = use_model()

    if event == "Calculate":
        if prediction1 != None and prediction2 != None:
            answer = 0
            while True:
                layout = [[sg.Text("Do you want to add subtract multiply or devide?")],
                          [sg.Button("+"), sg.Button("-"), sg.Button("X"),
                           sg.Button("/"), sg.Text(answer), sg.Button("Exit")]]

                window = sg.Window("Model Selector", layout)
                event, values = window.read()
                window.close()

                if event == "+":
                    answer = prediction1 + prediction2
                    print (prediction1)
                    print (prediction2)
                if event == "-":
                    answer = prediction1 - prediction2
                if event == "X":
                    answer = prediction1 * prediction2
                if event == "/":
                    answer = prediction1 / prediction2

                if event == "Exit" or event == sg.WIN_CLOSED:
                    break


            







import matplotlib.pyplot as plt
import numpy as np
import torch
import json
import torch.nn.functional as F
from torch import nn, optim
from torchvision import datasets, transforms, models
import torchvision.models as models
import seaborn as sns
from PIL import Image
from matplotlib.ticker import FormatStrFormatter
import argparse

def load_data(where  = "./flowers" ):
    data_dir = where
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms =transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    validation_transforms =transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    test_transforms =transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                 [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    test_data = datasets.ImageFolder(test_dir ,transform = test_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validationloader = torch.utils.data.DataLoader(validation_data, batch_size =32,shuffle = True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = 20, shuffle = True)

    return trainloader , validationloader , testloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def nn_setup(structure,features,Dropout,lr ):

    if structure == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif structure == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif structure == 'alexnet':
        model = models.alexnet(pretrained = True)
    else:
        print("Im sorry but {} is not a valid model.Did you mean vgg16,densenet121,or alexnet?".format(structure))
    for param in model.parameters():
        param.requires_grad = False
    
    features = list(model.classifier.children())[:-1]
    # number of filters in the bottleneck layer
    num_filters = model.classifier[len(features)].in_features
    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                                features.extend([
                                      nn.Dropout(),
                                      nn.Linear(num_filters, hidden_units),
                                      nn.ReLU(True),
                                      nn.Dropout(),
                                      nn.Linear(hidden_units, hidden_units),
                                      nn.ReLU(True),
                                      nn.Linear(hidden_units, num_labels),
                                      ])


    model.classifier = nn.Sequential(*features)
    model.to(device);
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)

    return model, criterion, optimizer




def train_network():
    epochs = 12
    steps = 0
    running_loss = 0
    print_every = 5
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if steps % print_every == 0:
                model.eval()
                test_loss = 0
                accuracy =0
                with torch.no_grad():
                    for inputs, labels in validationloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)



                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(testloader):.3f}.. "
                    f"Test accuracy: {accuracy/len(testloader):.3f}")

                running_loss = 0
                model.train()

def check_accuracy_on_test(testloader):
    correct = 0
    total = 0
    model.to(device);
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))



def save_checkpoint():
    model.class_to_idx = train_data.class_to_idx
    model.cpu
    checkpoint= {
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'class_to_idx':model.class_to_idx}
    torch.save(checkpoint, 'checkpoint.pth')

def load_model(path):
    checkpoint = torch.load('checkpoint.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    model.class_to_idx = checkpoint['class_to_idx']

    model.eval()
    model.train()
    load_model('checkpoint.pth')

def process_image(image_path):
    '''
    Scales, crops, and normalizes a PIL image for a PyTorch
    model, returns an Numpy array
    '''

    img = Image.open(image_path) # Open the image


    if img.size[0] > img.size[1]:
        img.thumbnail((10000, 256))
    else:
        img.thumbnail((256, 10000))


    left_margin = (img.width-224)/2
    right_margin = left_margin + 224


    bottom_margin = (img.height-224)/2
    top_margin = bottom_margin + 224

    img = img.crop((left_margin, bottom_margin, right_margin,
                       top_margin))

    img = np.array(img)/255


    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean)/std

    img = img.transpose((2, 0, 1))

    return img

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    if title:
        plt.title(title)

    image = image.transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    image = std * image + mean

    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax

model = model.to('cpu')

def predict(image_path, model, top_num=5):

    img = process_image(image_path)
    image_tensor = torch.from_numpy(img).type(torch.FloatTensor)

    model_input = image_tensor.unsqueeze(0)
    probs = torch.exp(model.forward(model_input))


    top_probs, top_labs = probs.topk(top_num)
    top_probs = top_probs.detach().numpy().tolist()[0]
    top_labs = top_labs.detach().numpy().tolist()[0]


    idx_to_class = {val: key for key, val in
                                      model.class_to_idx.items()}

    top_labels = [idx_to_class[lab] for lab in top_labs]
    top_flowers = [cat_to_name[idx_to_class[lab]] for lab in top_labs]

    return top_probs, top_labels, top_flowers

def plot_solution(image_path, model):

    plt.figure(figsize = (6,10))
    ax = plt.subplot(2,1,1)

    flower_num = image_path.split('/')[2]
    title_ = cat_to_name[flower_num]

    img = process_image(image_path)
    imshow(img, ax, title = title_);

    probs, labs, flowers = predict(image_path, model)

    plt.subplot(2,1,2)
    sns.barplot(x=probs, y=flowers, color=sns.color_palette()[0]);
    plt.show()

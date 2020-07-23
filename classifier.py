import matplotlib.pyplot as plt
import torch
import torchvision
import sys
import PIL
import json
import numpy as np

def load_image(relative_path):
    '''
    Loads a single image from drive, to be utilized in PyTorch Models
    '''
    print('[*]loading image from relative_path= {}'.format(relative_path))
    image_pil = PIL.Image.open(relative_path)
    image  = torchvision.transforms.ToTensor()((image_pil))
    image = image.unsqueeze(0)
    image = image.to('cuda')
    image.requires_grad = True
    return image

def printable_image(img):
    '''
    Converts PyTorch tensor to numpy array for representation of the image
    '''
    device = torch.device('cpu')
    img = img.squeeze(0)
    img = img.detach()
    img = img.to(device)
    npimg = img.numpy()
    img = np.transpose(npimg,(1,2,0))
    return img

def load_labels(path):
    with open(path) as f:
        imagenet_class_index = json.load(f)
        labels = [] 
        for i, _ in enumerate(imagenet_class_index):
            labels.append(imagenet_class_index[str(i)][1])
        return labels

def plot_image(image, title=''):
    '''
    Plots every image to be represnted
    '''
    plt.imshow(image)
    plt.title(title)
    plt.show()

if len(sys.argv) != 2:
    print('Usage: python {} <target_image>'.format(sys.argv[0]))
    print('Example: python {} giant_panda.jpg'.format(sys.argv[0]))
    exit()

labels = load_labels('imagenet_class_index.json')
device = torch.device("cuda")
model = torchvision.models.googlenet(pretrained=True).to(device)

image = load_image(sys.argv[1])

model.eval()
evaluation = int(model(image).argmax())
print('[*]the googlenet has predicted: {}'.format(labels[evaluation]))

img = printable_image(image)
plt.figure(num='The image {}'.format(sys.argv[1]))

plot_image(img,title= 'classified as {}'.format(labels[evaluation]))

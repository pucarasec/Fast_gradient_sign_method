import torch
import torchvision
import json
import PIL
import matplotlib.pyplot as plt
import numpy as np


# Loading labels of ImageNet Dataset
with open('imagenet_class_index.json') as f:
    imagenet_class_index = json.load(f)
    labels = [] 
    for i, _ in enumerate(imagenet_class_index):
        labels.append(imagenet_class_index[str(i)][1])

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


def add_ax_img(image, title, index, ax, invert=False):
    '''
    Adds an image to the argument Axes
    '''
    img = printable_image(image)
    if invert:
        aux = np.full((240,240,3), 1)
        img = aux - img*10
    ax.ravel()[index].imshow(img)
    ax.ravel()[index].set_title(title)
    ax.ravel()[index].set_axis_off()

def plot_results(original, altered_image, noise, original_label='', attack_image_label= ''):
    '''
    Plots every image to be represnted
    '''
    figure, ax = plt.subplots(ncols=3 )       
    add_ax_img(original, 'Original | Classified:{}'.format(original_label), 0, ax)
    add_ax_img(altered_image, 'Adversarial example | Classified:{}'.format(attack_image_label), 1, ax)
    add_ax_img(noise, 'Misclassification vector amplified', 2, ax)
    plt.tight_layout()
    plt.show()



# Loading Pytorch model
device = torch.device("cuda")
model = torchvision.models.googlenet(pretrained=True).to(device)

original = load_image("./giant_panda.jpg")

# Evaluation original image
model.eval()
original_evaluation = model(original)
correct_evaluation = int(original_evaluation.argmax())
print('[*]the googlenet has predicted: {}'.format(labels[correct_evaluation]))


# [*] FAST GRADIENT SIGN METHOD

# small enough so the misclassification vector it is not discernible for humans
magnitude_of_smallest_bit = 0.007 

# creation of the Misclassification Vector
loss = torch.nn.CrossEntropyLoss()
cost = loss(original_evaluation, torch.tensor([correct_evaluation]).to(device))
cost.backward()
model.zero_grad()
noise =   original.grad.sign()
noise = torch.clamp(noise, 0, 1)

## Creation of adversarial example
adversarial_example = original + magnitude_of_smallest_bit * noise
adversarial_example = torch.clamp(adversarial_example, 0, 1) # 

# Evaluation of adversarial example
adversarial_evaluation = int(model(adversarial_example).argmax())
print('[*]the googlenet has predicted: {}'.format(labels[adversarial_evaluation]))

# Plot results
plot_results(original, adversarial_example, noise, original_label=labels[correct_evaluation], attack_image_label= labels[adversarial_evaluation])
                                            


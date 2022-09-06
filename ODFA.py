import torch
from torch.autograd import Variable
from copy import deepcopy
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
#   Online Adversarial Defense Trainnig via ODFA.
#   https://github.com/layumi/U_turn/blob/master/README.md
def ODFA(model, img, rate = 16):
            model = deepcopy(model)
            model.eval()
            model.classifier.return_f = False
            n, c, h, w = img.size()
            inputs = Variable(img.cuda(), requires_grad=True)
            # ---------------------attack------------------
            # The input has been whiten.
            # So when we recover, we need to use a alpha
            alpha = 1.0 / (0.226 * 255.0)
            inputs_copy = Variable(inputs.data, requires_grad = False)
            diff = torch.FloatTensor(inputs.shape).zero_()
            diff = Variable(diff.cuda(), requires_grad = False)

            model.model.fc = nn.Sequential() #nn.Sequential(*L2norm)
            model.classifier.classifier = nn.Sequential()
            #model.classifier = nn.Sequential() PCB
            outputs = model(inputs)
            fnorm = torch.norm(outputs, p=2, dim=1, keepdim=True)
            outputs = outputs.div(fnorm.expand_as(outputs))
            outputs = outputs.view(outputs.size(0), -1)
            #print(outputs.shape)
            #feature_dim = outputs.shape[1]
            #batch_size = inputs.shape[0]
            #zero_feature = torch.zeros(batch_size,feature_dim)
            target = Variable(-outputs.data, requires_grad=False)
            criterion2 = nn.MSELoss()
            max_iter = round(min(1.25 * rate, rate+4))
            for iter in range( max_iter ):
                loss2 = criterion2(outputs, target)
                loss2.backward()
                diff += torch.sign(inputs.grad)
                mask_diff = diff.abs() > rate
                diff[mask_diff] = rate * torch.sign(diff[mask_diff])
                inputs = inputs_copy - diff * 1.0 * alpha
                inputs = clip(inputs,n)
                inputs = Variable(inputs.data, requires_grad=True)
                if iter == max_iter-1: break
                outputs = model(inputs)
                fnorm = torch.norm(outputs, p=2, dim=1, keepdim=True)
                outputs = outputs.div(fnorm.expand_as(outputs))
                outputs = outputs.view(outputs.size(0), -1)

            return inputs.detach()



def clip(inputs, batch_size):
    inputs = inputs.data
    for i in range(batch_size):
        inputs[i] = clip_single(inputs[i])
    inputs = Variable(inputs.cuda())
    return inputs

#######################################################################
# Creat Up bound and low bound
# Clip

data_transforms = transforms.Compose([
        transforms.Resize((256,128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

zeros = np.zeros((256,128,3),dtype=np.uint8)
zeros = Image.fromarray(zeros)
zeros = data_transforms(zeros)

ones = 255*np.ones((256,128,3), dtype=np.uint8)
ones = Image.fromarray(ones)
ones = data_transforms(ones)

zeros,ones = zeros.cuda(),ones.cuda()
def clip_single(input):
    low_mask = input<zeros
    up_mask = input>ones
    input[low_mask] = zeros[low_mask]
    input[up_mask] = ones[up_mask]
    return input


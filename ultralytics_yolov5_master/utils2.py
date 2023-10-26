from torchvision.io.image import read_file, read_image, write_jpeg
from torchvision.models import segmentation 
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image
from torch import tensor, uint8, argmin, argmax, device, sum, mul
from torch.nn import Sequential, Module, Softmax, functional
import numpy as np
import matplotlib.pyplot as plt
import os, datetime, cv2, glob, tqdm, time
from torch.utils.data import Dataset
from natsort import natsorted
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch import nn
from kornia.contrib import connected_components


class BinaryDiceLoss_blobPunish(Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean' ):
        super(BinaryDiceLoss_blobPunish, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"

        labels = connected_components(1*(predict>(predict.max()/2)).float(),num_iterations=200)
        blob_number_penalty = torch.numel(torch.unique(labels))/predict.shape[0]

        if blob_number_penalty < 1:
            blob_number_penalty = predict.shape[0]
        if blob_number_penalty > predict.shape[0]:
            blob_number_penalty = predict.shape[0]

        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = sum(mul(predict, target), dim=1) + self.smooth
        den = sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            loss = loss.mean()*blob_number_penalty
        elif self.reduction == 'sum':
            loss = loss.sum()*blob_number_penalty
        elif self.reduction == 'none':
            loss = loss*blob_number_penalty
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))
        
        return loss

class BinaryDiceLoss(Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean' ):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = sum(mul(predict, target), dim=1) + self.smooth
        den = sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

def del_dir_contents(path_to_dir):
    files = glob.glob(os.path.join(path_to_dir,'*'))
    for f in files:
        try:
            os.remove(f)
        except:
            print('Cant delete:', f)

def read_img_custom(path):
    if 'Tensor' in str(type(path)):
        return path
    else:
        if path[-3:] == 'png' or 'jpg':
            img = read_image(path).float()/255
        else:
            img = Image.open(path)
            img = np.asarray(img)
            img = np.moveaxis(img,2,0)
            img = tensor(img).float()/255
        return img

def read_all_images(paths, transforms = None, number_to_stop_at = None):
    out = []
    for i,this_path in enumerate(tqdm.tqdm(paths)):
        temp_img = read_img_custom(this_path)
        if transforms is not None:
            temp_img = transforms()(temp_img)
        out.append(temp_img)
        if number_to_stop_at is not None:
            if i >= number_to_stop_at:
                break
    return out

def normalize_img(input_img):
    input_img = input_img - input_img.min()
    input_img = input_img/input_img.max()
    return input_img

def s(img, title = None):
    if 'torch' in str(img.dtype):
        img = img.squeeze()
        if len(img.shape) > 2: # check RGB
            if argmin(tensor(img.shape)) == 0: # check if CHW 
                img = img.permute((1, 2, 0)) # change to HWC
        img = normalize_img(img)*255
        img = img.to('cpu').to(uint8)
    else:
        img = np.moveaxis(img,0,-1)
    plt.figure()
    plt.imshow(img)
    plt.title(title)

class SegmentationDataset(Dataset):
    def __init__(self, imagePaths, maskPaths = None, transforms = None, resize = None, device = device('cpu')):
        # store the image and mask filepaths, and augmentation
        # transforms
        self.imagePaths = imagePaths
        self.maskPaths = maskPaths
        self.transforms = transforms
        self.resize = resize
        self.device = device
    def __len__(self):
        # return the number of total samples contained in the dataset
        return len(self.imagePaths)
    def __getitem__(self, idx):
        # grab the image path from the current index
        imagePath = self.imagePaths[idx]
        # load the image from disk, swap its channels from BGR to RGB,
        # and read the associated mask from disk in grayscale mode
        image = read_img_custom(imagePath) # cv2.imread(imagePath)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.maskPaths is not None:
            mask = read_img_custom(self.maskPaths[idx])#cv2.imread(maskPath, 0)
        # check to see if we are applying any transformations
        if self.transforms is not None:
            # apply the transformations to both image and its mask
            # image = self.transforms(image)
            # if self.maskPaths is not None:
            #     mask = self.resize(mask)
            if self.maskPaths is not None:
                temp = self.transforms(image = image.squeeze().numpy(), mask = mask.squeeze().numpy())
                image = temp['image']
                mask = temp['mask'].unsqueeze(0)
            else:
                temp = self.transforms(image = image.squeeze().numpy())
                image = temp['image']

        if self.resize is not None:
            image = self.resize()(torch.tensor(image).unsqueeze(0))
            if self.maskPaths is not None:
                mask = self.resize()(mask)

        # return a tuple of the image and its mask
        if self.maskPaths is not None:
            image, mask = image.to(self.device), mask.to(self.device)
            return (image, mask)
        else:
            image = image.to(self.device)
            return (image)

def preprocess():
    preprocess_func = Sequential( 
        transforms.Resize([520,520],antialias=True),
        # transforms.Resize([384,384],antialias=True),
        transforms.Grayscale()
    )
    return preprocess_func

def write_outputs_to_images(inputs, outputs, output_path, i = 0, binary_threshold = 0.5):
    if 'Dict' in str(type(outputs)):
        outputs = outputs['out']
    
    scaled_toutputs = 1*(outputs/(outputs.max())).squeeze().unsqueeze(0)
    BW_toutputs = 1*(outputs>binary_threshold).squeeze().unsqueeze(0)

    Rscaled_toutputs = scaled_toutputs.clone()
    Rscaled_toutputs[BW_toutputs>0.5] = 1
    scaled_toutputs[BW_toutputs>0.5] = 0

    BW_toutputs = torch.concatenate((Rscaled_toutputs,scaled_toutputs ,scaled_toutputs ), dim = 0)

    this_tin_rgb = torch.concatenate((inputs,inputs,inputs),0).squeeze()
    norm_output = outputs/outputs.max()
    this_tin_rgb[0] = this_tin_rgb[0] + (norm_output)/2
    this_tin_rgb = torch.clip(this_tin_rgb,min=0.,max=1.)

    out1 = (this_tin_rgb*255).to(torch.uint8).cpu().squeeze()
    out2 = (BW_toutputs*255).to(torch.uint8).cpu().squeeze()
    out = torch.concatenate((out1,out2), dim = -1)

    try:
        write_jpeg(out,os.path.join(output_path,str(i) + '.jpg'),100)
    except:
        time.sleep(0.1)
        write_jpeg(out,os.path.join(output_path,str(i) + '.jpg'),100)

def get_model(model_size = 50, device = device('cpu'), freeze_layers = None, weights = True):

    from torch.nn import Parameter, Conv2d, Sigmoid
    from torch import mean as torch_mean

    if model_size == 50:
        if weights:
            model = segmentation.fcn_resnet50(weights = segmentation.FCN_ResNet50_Weights.DEFAULT)
        else:
            model = segmentation.fcn_resnet50(weights = None, aux_loss = False)
        # model = segmentation.deeplabv3_resnet50(weights = segmentation.DeepLabV3_ResNet50_Weights.DEFAULT)
    else:
        if weights:
            model = segmentation.fcn_resnet101(weights = segmentation.FCN_ResNet101_Weights.DEFAULT)
        else:
            model = segmentation.fcn_resnet101(weights = None, aux_loss = False)
        # model = segmentation.deeplabv3_resnet101(weights = segmentation.DeepLabV3_ResNet101_Weights.DEFAULT)

    if freeze_layers is not None:
        if freeze_layers == True:
            for param in model.parameters():
                param.requires_grad = False
        else:
            for param in model.parameters():
                param.requires_grad = True

    model.backbone.conv1 = Conv2d(1,64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False)
    model.backbone.conv1.requires_grad = True
    model.classifier._modules['4'] = Conv2d(512,1,kernel_size = (1,1),stride = (1,1), bias=True)
    model.classifier._modules['4'].requires_grad = True
    model.classifier._modules['5'] = Sigmoid()

    if weights:
        model.aux_classifier._modules['4'] = Conv2d(256,1,kernel_size = (1,1),stride = (1,1), bias=True)
        model.aux_classifier._modules['4'].requires_grad = True
        model.aux_classifier._modules['5'] = Sigmoid()

        input_weights = Parameter(torch_mean(model.backbone.conv1.weight,1).unsqueeze(1))
        model.backbone.conv1.weight = input_weights

    model = model.to(device)

    return model

def train_one_epoch(model, epoch, training_loader, optimizer, loss_fn, tb_writer = None):
    running_loss_total = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    stream = tqdm.tqdm(training_loader)
    for i, (inputs, labels) in enumerate(stream, start=1):

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)#['out']

        if 'Dict' in str(type(outputs)):
            outputs = outputs['out']

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss_total += loss.item()
        last_loss = running_loss_total / i # loss per batch

        stream.set_description("Epoch: {epoch}. Running LOSS: {metric_monitor}".format(epoch=epoch, metric_monitor=last_loss))

    return last_loss

def training_loop(model, EPOCHS, loss_fn, optimizer, training_loader, validation_loader, testing_loader, output_path = './output/', 
                  weights_outputs_path = os.getcwd(),best_vloss = 1000000, timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
                  epoch_number = 0, testing_binary_threshold = 0.5,
                  save_weights = True):
        
    for epoch in range(epoch_number,EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(model, epoch_number, training_loader, optimizer, loss_fn)#, writer)

        running_vloss_total = 0.0
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            stream = tqdm.tqdm(validation_loader)
            # for i, vdata in enumerate(tqdm.tqdm(validation_loader)):
            for i, (vinputs, vlabels) in enumerate(stream, start=1):
                voutputs = model(vinputs)#['out']
                if 'Dict' in str(type(voutputs)):
                    voutputs = voutputs['out']
                vloss = loss_fn(voutputs, vlabels)
                running_vloss_total += vloss 
                running_vloss = running_vloss_total / i
                stream.set_description("Epoch: {epoch}. Running VAL_LOSS: {metric_monitor}".format(epoch=epoch, metric_monitor=running_vloss))

        avg_vloss = running_vloss #/ (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:

            del_dir_contents(output_path)
            with torch.no_grad():
                for i, tdata in enumerate(tqdm.tqdm(testing_loader)):
                    tinputs = tdata
                    toutputs = model(tinputs)#['out']

                    write_outputs_to_images(tinputs, toutputs, output_path, i = i, binary_threshold = testing_binary_threshold)

            best_vloss = avg_vloss
            if save_weights == True:
                model_path = 'model_{}_{}'.format(timestamp, epoch_number) + '.pt'
                print('improved val_loss')
                torch.save(model.state_dict(), os.path.join(weights_outputs_path,model_path))
            if save_weights == False:
                print('improved val_loss')
            if save_weights == 'last':
                model_path = 'model_{}_{}'.format(timestamp, 0) + '.pt'
                print('improved val_loss')
                torch.save(model.state_dict(), os.path.join(weights_outputs_path,model_path))
        else:
            print('Did NOT improve - best val loss:', best_vloss)

        epoch_number += 1

    return model_path
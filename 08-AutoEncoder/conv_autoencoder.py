__author__ = 'SherlockLiao'

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import os
import heldkarp
from collections import Counter

output_dir = "dc_img128x128xTSP5_v2"
image_size = (128, 128)
if not os.path.exists('./%s' %output_dir):
    os.mkdir('./%s' %output_dir)


def to_img(x):
    x = torch.argmax(x, dim=1)
    print(torch.unique(x))
    print(Counter(torch.flatten(x).tolist()))

    x *= 126
    # x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.type(torch.FloatTensor)
    # print(x)
    # print(torch.max(x))
    # print(x.shape)
    # print(x.dtype)
    x = x.view(x.size(0), 1, image_size[0], image_size[1])
    # print(x.dtype)
    # print(torch.max(x))
    return x


num_epochs = 10000
if torch.cuda.is_available():
    batch_size = 128
else:
    batch_size = 1
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.ToTensor(),
])

class CustomDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        inp, out = heldkarp.generate_pair(5)
        inp = torch.as_tensor(inp, dtype=torch.int64)
        inp = F.one_hot(inp)
        inp = torch.transpose(inp, 1, 2)
        inp = torch.transpose(inp, 0, 1)
        inp = torch.squeeze(inp)
        inp = inp.type(torch.FloatTensor)
        # inp = inp.type(torch.FloatTensor)
        # inp.type(torch.LongTensor)

        out = torch.as_tensor(out, dtype=torch.int64)
        # out = F.one_hot(out)
        # out = torch.transpose(out, 0, 2)
        # out = torch.squeeze(out)
        out = out.type(torch.LongTensor)
        # out.type(torch.LongTensor)
        # if self.transform:
        #     inp = self.transform(inp)
        #     out = self.transform(out)
        # print(inp)
        # print(inp.shape)
        # inp, out = F.one_hot(inp), F.one_hot(out)
        test = torch.as_tensor(out, dtype=torch.int64)
        test = F.one_hot(test)
        test = torch.transpose(test, 1, 2)
        test = torch.transpose(test, 0, 1)
        test = torch.squeeze(test)
        test = test.type(torch.FloatTensor)
        return inp, out

# dataset = MNIST('./data', transform=img_transform, download=True)
dataset = CustomDataset(transform=img_transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Experiment 1 - Convolution and Deconvolution
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.block4T = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )
        self.block3T = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        self.block2T = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.block1T = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
        )
        self.pool4 = nn.Sequential(
            nn.Conv2d(512, 1, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(True),
            nn.Upsample(size=image_size, mode="bilinear"),
        )
        self.pool3 = nn.Sequential(
            nn.Conv2d(256, 1, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(True),
            nn.Upsample(size=image_size, mode="bilinear"),
        )
        self.pool2 = nn.Sequential(
            nn.Conv2d(128, 1, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(True),
            nn.Upsample(size=image_size, mode="bilinear"),
        )

    def forward(self, x):
        x_orig = x
        x = self.block1(x)
        x = self.block2(x)
        x2 = self.pool2(x)
        x = self.block3(x)
        x3 = self.pool3(x)
        x = self.block4(x)
        x4 = self.pool4(x)
        x = self.block4T(x)
        x = self.block3T(x)
        x = self.block2T(x)
        x = self.block1T(x)
        x = torch.cat([x2, x3, x4, x, x_orig[:, 1:2, :, :]], 1)
        x = nn.Sequential(
            nn.Conv2d(36, 3, 1, stride=1),
        ).cuda()(x)
        return x

if torch.cuda.is_available():
    model = autoencoder().cuda()
else:
    model = autoencoder()
print(model)
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform
model.apply(init_weights)
weights = torch.FloatTensor([1, 1, 1]).cuda()
criterion = nn.CrossEntropyLoss(weight=weights)
# criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
for epoch in range(num_epochs):
    total_loss = 0
    for data in dataloader:
        inp, out = data
        if torch.cuda.is_available():
            inp, out = inp.cuda(), out.cuda()
        # ===================forward=====================
        # print(inp.dtype)
        # print(inp.shape)
        output = model(inp)
        loss = criterion(output, out)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    total_loss += loss.data
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch+1, num_epochs, total_loss))
    if epoch % 10 == 0:
        print("Image written")
        pic = to_img(output.cpu().data)
        save_image(pic, './{}/image_{}.png'.format(output_dir, epoch))

torch.save(model.state_dict(), './conv_autoencoder.pth')

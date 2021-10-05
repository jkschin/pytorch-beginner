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

if not os.path.exists('./dc_img'):
    os.mkdir('./dc_img')


def to_img(x):
    x = torch.argmax(x, dim=1)
    x = x / 2
    # x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.type(torch.FloatTensor)
    # print(x)
    # print(torch.max(x))
    # print(x.shape)
    # print(x.dtype)
    x = x.view(x.size(0), 1, 100, 100)
    # print(x.dtype)
    # print(torch.max(x))
    return x


num_epochs = 100
batch_size = 128
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.ToTensor(),
])

class CustomDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        inp, out = heldkarp.generate_pair(10)
        inp = torch.as_tensor(inp, dtype=torch.int64)
        inp = F.one_hot(inp)
        inp = torch.transpose(inp, 1, 2)
        inp = torch.transpose(inp, 0, 1)
        inp = torch.squeeze(inp)
        inp = inp.type(torch.FloatTensor)
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
        return inp, out

# dataset = MNIST('./data', transform=img_transform, download=True)
dataset = CustomDataset(transform=img_transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.test_skip = nn.Conv2d(2, 16, 3, padding=1)
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            # nn.Tanh()
        )

    def forward(self, x):
        skip = self.test_skip(x)
        x = self.encoder(x)
        x = self.decoder(x)
        x = torch.cat([skip, x], 1)
        x = nn.Conv2d(17, 3, 3, padding=1)(x)
        # x = nn.Tanh()(x)
        return x


# model = autoencoder().cuda()
model = autoencoder()
print(model)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)
total_loss = 0
for epoch in range(num_epochs):
    for data in dataloader:
        inp, out = data
        # ===================forward=====================
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
        pic = to_img(output.cpu().data)
        save_image(pic, './dc_img/image_{}.png'.format(epoch))

torch.save(model.state_dict(), './conv_autoencoder.pth')

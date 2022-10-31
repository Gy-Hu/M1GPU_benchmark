import torch
from torchvision import transforms
import torchvision
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
import timm
import torch.nn as nn
from ranger21 import Ranger21
from torch.optim.lr_scheduler import CyclicLR
from tqdm import tqdm

def get_dataloader(batch_size):
    data_transform = {
        "train": transforms.Compose([transforms.Resize(96),
                                     transforms.ToTensor()]),
        "val": transforms.Compose([transforms.Resize(96),
                                   transforms.ToTensor()])
    }
    train_dataset = torchvision.datasets.CIFAR10('./p10_dataset', train=True, transform=data_transform["train"], download=True)
    test_dataset = torchvision.datasets.CIFAR10('./p10_dataset', train=False, transform=data_transform["val"], download=True)
    print('训练数据集长度: {}'.format(len(train_dataset)))
    print('测试数据集长度: {}'.format(len(test_dataset)))
    # DataLoader创建数据集
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader,test_dataloader

def show_pic(dataloader):#展示dataloader里的6张图片
    examples = enumerate(dataloader)  # 组合成一个索引序列
    batch_idx, (example_data, example_targets) = next(examples)
    classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        # plt.tight_layout()
        img = example_data[i]
        print('pic shape:',img.shape)
        img = img.swapaxes(0, 1)
        img = img.swapaxes(1, 2)
        plt.imshow(img, interpolation='none')
        plt.title(classes[example_targets[i].item()])
        plt.xticks([])
        plt.yticks([])
    plt.show()

def get_net(): #获得预训练模型并冻住前面层的参数
    net = timm.create_model('resnet18', pretrained=True, num_classes=10)
    #print(summary(net, input_size=(3, 224, 224)))
    '''Freeze all layers except the last layer(fc or classifier)'''
    for param in net.parameters():
        param.requires_grad = False
    # nn.init.xavier_normal_(model.fc.weight)
    # nn.init.zeros_(model.fc.bias)
    net.fc.weight.requires_grad = True
    net.fc.bias.requires_grad = True
    return net

def train(net, loss, train_dataloader, valid_dataloader, device, batch_size, num_epoch, lr, lr_min, optim='sgd', init=True, scheduler_type='Cosine'):
    def init_xavier(m):
        #if type(m) == nn.Linear or type(m) == nn.Conv2d:
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)

    if init:
        net.apply(init_xavier)

    print('training on:', device)
    net.to(device)

    if optim == 'sgd':
        optimizer = torch.optim.SGD((param for param in net.parameters() if param.requires_grad), lr=lr,
                                    weight_decay=0)
    elif optim == 'adam':
        optimizer = torch.optim.Adam((param for param in net.parameters() if param.requires_grad), lr=lr,
                                     weight_decay=0)
    elif optim == 'adamW':
        optimizer = torch.optim.AdamW((param for param in net.parameters() if param.requires_grad), lr=lr,
                                      weight_decay=0)
    elif optim == 'ranger':
        optimizer = Ranger21((param for param in net.parameters() if param.requires_grad), lr=lr,weight_decay=0,num_epochs=num_epoch,num_batches_per_epoch=len(train_dataloader),use_warmup=False,use_madgrad=False)
    if scheduler_type == 'Cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=lr_min)
    
    if scheduler_type == 'Cyclic':
        scheduler =CyclicLR(optimizer,base_lr=lr_min,max_lr=lr)

    train_losses = []
    train_acces = []
    eval_acces = []
    best_acc = 0.0
    for epoch in range(num_epoch):

        print("——————第 {} 轮训练开始——————".format(epoch + 1))

        # 训练开始
        net.train()
        train_acc = 0
        for batch in tqdm(train_dataloader):
            imgs, targets = batch
            imgs = imgs.to(device)
            targets = targets.to(device)
            output = net(imgs)

            Loss = loss(output, targets)
        
            optimizer.zero_grad()
            Loss.backward()
            optimizer.step()

            _, pred = output.max(1)
            num_correct = (pred == targets).sum().item()
            acc = num_correct / (batch_size)
            train_acc += acc
        scheduler.step()
        print("epoch: {}, Loss: {}, Acc: {}".format(epoch+1, Loss.item(), train_acc / len(train_dataloader)))
        train_acces.append(train_acc / len(train_dataloader))
        train_losses.append(Loss.item())

        # 测试步骤开始
        net.eval()
        eval_loss = 0
        eval_acc = 0
        with torch.no_grad():
            for imgs, targets in valid_dataloader:
                imgs = imgs.to(device)
                targets = targets.to(device)
                output = net(imgs)
                Loss = loss(output, targets)
                _, pred = output.max(1)
                num_correct = (pred == targets).sum().item()
                eval_loss += Loss
                acc = num_correct / imgs.shape[0]
                eval_acc += acc

            eval_losses = eval_loss / (len(valid_dataloader))
            eval_acc = eval_acc / (len(valid_dataloader))
            if eval_acc > best_acc:
                best_acc = eval_acc
                torch.save(net.state_dict(),'best_acc.pth')
            eval_acces.append(eval_acc)
            print("整体验证集上的Loss: {}".format(eval_losses))
            print("整体验证集上的正确率: {}".format(eval_acc))
    return train_losses, train_acces, eval_acces

def show_acces(train_losses, train_acces, valid_acces, num_epoch):#对准确率和loss画图显得直观
    plt.plot(1 + np.arange(len(train_losses)), train_losses, linewidth=1.5, linestyle='dashed', label='train_losses')
    plt.plot(1 + np.arange(len(train_acces)), train_acces, linewidth=1.5, linestyle='dashed', label='train_acces')
    plt.plot(1 + np.arange(len(valid_acces)), valid_acces, linewidth=1.5, linestyle='dashed', label='valid_acces')
    plt.grid()
    plt.xlabel('epoch')
    plt.xticks(range(1, 1 + num_epoch, 1))
    plt.legend()
    plt.show()

if __name__ == '__main__':
    train_dataloader, test_dataloader = get_dataloader(batch_size=1024)
    show_pic(train_dataloader)
    device = torch.device("mps")
    net = get_net()
    loss = nn.CrossEntropyLoss()
    train_losses, train_acces, eval_acces = train(net, loss, train_dataloader, test_dataloader, device, batch_size=1024, num_epoch=10, lr=0.1, lr_min=1e-5, optim='ranger',scheduler_type='Cyclic',init=False)
    show_acces(train_losses, train_acces, eval_acces, num_epoch=10)

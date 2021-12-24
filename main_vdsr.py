import argparse, os
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from vdsr import Net, BConv2d
from dataset import DatasetFromHdf5
import time

parser = argparse.ArgumentParser(description="PyTorch VDSR")
parser.add_argument("--batchSize", type=int, default=128, help="Training batch size")
parser.add_argument("--nEpochs", type=int, default=50, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.1, help="Learning Rate. Default=0.1")
parser.add_argument("--step", type=int, default=10, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
parser.add_argument("--cuda", default=True, help="Use cuda?")
parser.add_argument("--parallel", default=0, type=int, help = "number of GPUs for parallel training")
parser.add_argument("--resume", default="none", type=str, help="Path to checkpoi    nt (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--clip", type=float, default=0.4, help="Clipping Gradients. Default=0.4")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="Weight decay, Default: 1e-4")
parser.add_argument('--pretrained', default='none', type=str, help='path to pretrained model (default: none)')
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')

def main():
    global opt, model
    opt = parser.parse_args()
    print(opt)
    cuda = opt.cuda
    if cuda:
        print("cuda")
        print("=> use gpu id: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        if not torch.cuda.is_available():
                raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    # Sets the seed for generating random numbers.
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True
    print("===> Loading datasets")
    train_set = DatasetFromHdf5("data/train.h5")
    training_data_loader = DataLoader(dataset=train_set, batch_size=opt.batchSize, shuffle=True)

    print("===> Building model")
    model = Net()

    criterion = nn.MSELoss(reduction='sum')

    print("===> Setting GPU")
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    # optionally resume from a checkpoint  
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))
    else:
        print("opt.resume == none")
    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            model.load_state_dict(weights['model'].state_dict())
        else:
            print("=> no model found at '{}'".format(opt.pretrained))

    print("===> Setting Optimizer")
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    """
    Batch size is set to 16. Adam is utilized to optimize the network. The momentum parameter is set to 0.5, 
    weight decay is set to 2 × 10−4, and the initial learning rate is set to 1 × 10−4 and will be divided 
    a half every 200 epochs.
    """
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print("===> Training")
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        temperature = set_temperature(model, epoch)
        train(training_data_loader, optimizer, model, criterion, epoch, temperature)
        # writer.close()
        save_checkpoint(model, epoch)

def adjust_learning_rate(optimizer, epoch):
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr

def set_temperature(model, epochs):
    Temperature = torch.ones(1)
    for i, module in enumerate(model.modules()):
        if isinstance(module, BConv2d):
            if epochs != 0:
                module.update_temperature()
            Temperature = torch.Tensor([module.temperature])
    return Temperature

def train(training_data_loader, optimizer, model, criterion, epoch, temperature):
    lr = adjust_learning_rate(optimizer, epoch-1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    print("Epoch = {}, lr = {}".format(epoch, optimizer.param_groups[0]["lr"]))
    model.train()
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = Variable(batch[0]), Variable(batch[1], requires_grad=False)

        if opt.cuda:
            input = input.cuda()
            target = target.cuda()
        loss = criterion(model(input), target)

        if iteration % 100 == 0:
            output = "===> Epoch[{}]({}/{}): Loss: {:.10f} temperature: {:.3f} learning rate: {:.10f}".format(epoch, iteration, len(training_data_loader), loss.item(), temperature.item(), opt.lr)
            with open("loss.txt", "a+") as f:
                f.write(output + '\n')
                f.close

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
        optimizer.step()

        if iteration%100 == 0:
          print("===> Epoch[{}]({}/{}): Loss: {:.10f} temperature: {:.3f}".format(epoch, iteration, len(training_data_loader), loss.item(), temperature.item()))

def save_checkpoint(model, epoch):
    model_out_path = "checkpoint/" + "model_epoch_{}.pth".format(epoch)
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists("checkpoint/"):
        os.makedirs("checkpoint/")

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))

if __name__ == "__main__":
    main()


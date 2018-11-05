import os
import time
import torch.optim as optim
from tensorboardX import SummaryWriter
import model
from utils import *
from configuration import get_config
import dataloader

args = get_config()
device = args.device
torch.manual_seed(args.seed)

train_loader = dataloader.dataloader(args.data_directory, args.batch_size, True)
test_loader = dataloader.dataloader(args.data_directory, args.batch_size, False)

encoder = model.Encoder().to(device)

models = dict()
models['encoder.pt'] = encoder

if args.load_model != '000000000000':
    for model_name, model in models.items():
        model.load_state_dict(torch.load(os.path.join(args.log_directory, args.project, args.load_model, model_name)))
    args.time_stamp = args.load_model[:12]
    print('Model {} loaded.'.format(args.load_model))

def epoch(epoch_idx, is_train):
    epoch_start_time = time.time()
    start_time = time.time()
    mode = 'Train' if is_train else 'Test'
    epoch_loss = 0
    if is_train:
        for model in models.values():
            model.train()
        loader = train_loader
    else:
        for model in models.values():
            model.eval()
        loader = test_loader
    for batch_idx, sequence in enumerate(loader):
        batch_size = image.size()[0]
        optimizer.zero_grad()
        sequence = sequence.to(device)

        loss = F.cross_entropy(output, answer)
        if is_train:
            loss.backward()
            optimizer.step()
        epoch_loss += loss.item()
        pred = torch.max(output.data, 1)[1]
        correct = (pred == answer)
        if is_train:
            if batch_idx % args.log_interval == 0:
                print('Train Batch: {} [{}/{} ({:.0f}%)] Loss: {:.4f} / Time: {:.4f} / Acc: {:.4f}'.format(
                    epoch_idx,
                    batch_idx * batch_size, len(loader.dataset),
                    100. * batch_idx / len(loader),
                    loss.item() / batch_size,
                    time.time() - start_time,
                    correct.sum().item() / batch_size))
                idx = epoch_idx * len(loader) // args.log_interval + batch_idx // args.log_interval
                writer.add_scalar('Batch loss', loss.item() / batch_size, idx)
                writer.add_scalar('Batch accuracy', correct.sum().item() / batch_size, idx)
                writer.add_scalar('Batch time', time.time() - start_time, idx)
                start_time = time.time()

    print('====> {}: {} Average loss: {:.4f} / Time: {:.4f} / Accuracy: {:.4f}'.format(
        mode,
        epoch_idx,
        epoch_loss / len(loader.dataset),
        time.time() - epoch_start_time,
        sum(q_correct.values()) / len(loader.dataset)))
    writer.add_scalar('{} loss'.format(mode), epoch_loss / len(loader.dataset), epoch_idx)
    q_acc = {}
    for i in range(loader.dataset.q_size):
        q_acc['question {}'.format(str(i))] = q_correct[i] / q_num[i]
    q_corrects = list(q_correct.values())
    q_nums = list(q_num.values())
    writer.add_scalars('{} accuracy per question'.format(mode), q_acc, epoch_idx)
    writer.add_scalar('{} non-rel accuracy'.format(mode), sum(q_corrects[:3]) / sum(q_nums[:3]), epoch_idx)
    writer.add_scalar('{} rel accuracy'.format(mode), sum(q_corrects[3:]) / sum(q_nums[3:]), epoch_idx)
    writer.add_scalar('{} total accuracy'.format(mode), sum(q_correct.values()) / len(loader.dataset), epoch_idx)


if __name__ == '__main__':
    optimizer = optim.Adam([param for model in models.values() for param in list(model.parameters())], lr=args.lr)
    writer = SummaryWriter(args.log)
    for epoch_idx in range(args.start_epoch, args.start_epoch + args.epochs):
        # epoch(epoch_idx, True)
        epoch(epoch_idx, False)
        for model_name, model in models.items():
            torch.save(model.state_dict(), args.log + model_name)
        print('Model saved in ', args.log)
    writer.close()

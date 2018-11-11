import os
import time
import torch.optim as optim
from tensorboardX import SummaryWriter
import model
from utils import *
from configuration import get_config
import dataloader
import numpy as np
import torch
from sklearn.decomposition import PCA


args = get_config()

device = args.device
torch.manual_seed(args.seed)

train_loader = dataloader.dataloader(args.data_directory, args.batch_size, True)
test_loader = dataloader.dataloader(args.data_directory, args.batch_size, False)

c_encoder = model.Encoder(args.lstm_input, args.lstm_hidden, args.lstm_layers, args.code_size).to(device)
classifier = model.MLP([args.code_size] + [args.fc_hidden for i in range(args.fc_layers)] + [args.label_num]).to(device)
s_encoder = model.Encoder(args.lstm_input, args.lstm_hidden, args.lstm_layers, args.code_size).to(device)
decoder = model.Decoder(args.code_size * 2, args.lstm_hidden, args.lstm_layers, args.lstm_input, args.lstm_length).to(device)

first_models = dict()
first_models['c_encoder.pt'] = c_encoder
first_models['classifier.pt'] = classifier
second_models = dict()
second_models['s_encoder.pt'] = s_encoder
second_models['decoder.pt'] = decoder

if args.load_model != '000000000000':
    for model_name, model in first_models.items():
        model.load_state_dict(torch.load(os.path.join(args.log_directory, args.project, args.load_model, model_name)))
    for model_name, model in second_models.items():
        model.load_state_dict(torch.load(os.path.join(args.log_directory, args.project, args.load_model, model_name)))
    args.time_stamp = args.load_model[:12]
    print('Model {} loaded.'.format(args.load_model))


def first_epoch(epoch_idx, is_train):
    epoch_start_time = time.time()
    mode = 'Train' if is_train else 'Test'
    epoch_loss = 0
    total_correct = 0
    if is_train:
        for model in first_models.values():
            model.train()
        loader = train_loader
    else:
        for model in first_models.values():
            model.eval()
        loader = test_loader
    for batch_idx, (input_data, label) in enumerate(loader):
        first_optimizer.zero_grad()
        input_data = input_data.float().to(device)
        label = label.to(device)
        code = c_encoder(input_data)
        output = classifier(code)
        loss = F.cross_entropy(output, label)
        if is_train:
            loss.backward()
            first_optimizer.step()
        epoch_loss += loss.item()
        correct = (output.max(dim = 1)[1] == label)
        total_correct += correct.sum().item()
    print('====> {}: First {} Average loss: {:.4f} / Time: {:.4f} / Accuracy: {:.4f}'.format(
        mode,
        epoch_idx,
        epoch_loss / len(loader.dataset),
        time.time() - epoch_start_time,
        total_correct / len(loader.dataset)))
    writer.add_scalar('First {} loss'.format(mode), epoch_loss / len(loader.dataset), epoch_idx)
    writer.add_scalar('First {} total accuracy'.format(mode), total_correct / len(loader.dataset), epoch_idx)


def second_epoch(epoch_idx, is_train):
    epoch_start_time = time.time()
    mode = 'Train' if is_train else 'Test'
    epoch_loss = 0
    total_dis = 0
    total_recon = 0
    if is_train:
        for model in second_models.values():
            model.train()
        loader = train_loader
    else:
        for model in second_models.values():
            model.eval()
        loader = test_loader
    for batch_idx, (input_data, label) in enumerate(loader):
        second_optimizer.zero_grad()
        input_data = input_data.float().to(device)
        label = label.to(device)
        c_code = c_encoder(input_data).detach()
        s_code = s_encoder(input_data)
        output = classifier(s_code)
        discriminator_loss = F.cross_entropy(output, label)
        recon = decoder(c_code, s_code)
        reconstruction_loss = F.smooth_l1_loss(recon, input_data)
        loss = - discriminator_loss + reconstruction_loss
        if is_train:
            loss.backward()
            second_optimizer.step()
        epoch_loss += loss.item()
        total_dis += discriminator_loss.item()
        total_recon += reconstruction_loss.item()
    print('====> {}: Second {} Average loss: {:.4f} / Time: {:.4f}'.format(
        mode,
        epoch_idx,
        epoch_loss / len(loader.dataset),
        time.time() - epoch_start_time))
    writer.add_scalar('Second {} loss'.format(mode), epoch_loss / len(loader.dataset), epoch_idx)
    writer.add_scalar('Second {} discriminator loss'.format(mode), total_dis / len(loader.dataset), epoch_idx)
    writer.add_scalar('Second {} reconstruction loss'.format(mode), total_recon / len(loader.dataset), epoch_idx)


if __name__ == '__main__':
    first_optimizer = optim.Adam([param for model in first_models.values() for param in list(model.parameters())],
                                 lr=args.lr)
    second_optimizer = optim.Adam([param for model in second_models.values() for param in list(model.parameters())],
                                  lr=args.lr)
    writer = SummaryWriter(args.log)
    for epoch_idx in range(args.start_epoch, args.start_epoch + args.epochs1):
        first_epoch(epoch_idx, True)
        first_epoch(epoch_idx, False)
        for model_name, model in first_models.items():
            torch.save(model.state_dict(), os.path.join(args.log, model_name))
        print('Model saved in ', args.log)

    for epoch_idx in range(args.start_epoch, args.start_epoch + args.epochs2):
        second_epoch(epoch_idx, True)
        second_epoch(epoch_idx, False)
        for model_name, model in second_models.items():
            torch.save(model.state_dict(), os.path.join(args.log, model_name))
        print('Model saved in ', args.log)
    writer.close()

    #train data 클러스터링 확인
    data = dataloader.DailyStockPrice(args.data_directory, train=True)
    size = args.batch_size
    date = []
    s = []
    for batch_idx, (input_data, label) in enumerate(train_loader): #여기 loader도 바꿀
        input_data = input_data.float().to(device)
        s_code = s_encoder(input_data).detach()
        idx = np.asarray(data.get_date(idx=batch_idx*size, batch_size=size, mode=True))
        for i in range(s_code.shape[0]):
            s.append(np.asarray(s_code[i,:]))
            date.append(idx[i])

    d_s = {}
    d_s = dict(zip(date, s))

    pca = PCA(n_components=2)
    pca.fit(np.asarray(s))
    print(pca.explained_variance_ratio_)

    # special = [2010-03-26, 2010-11-23]
    print(max(date))

    import matplotlib.pyplot as plt
    x = pca.fit_transform(np.asarray(s))
    print(x.shape)
    plt.figure()
    plt.scatter(x[:,0], x[:,1], c = 'g')
    plt.show()



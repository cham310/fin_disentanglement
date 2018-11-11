import os
import torch
import argparse
import datetime

def get_config():
    parser = argparse.ArgumentParser(description='parser')

    model_arg = parser.add_argument_group('Model')
    model_arg.add_argument('--project', type=str, default='fin')
    model_arg.add_argument('--code-size', type=int, default=16)
    model_arg.add_argument('--lstm-input', type=int, default=6)
    model_arg.add_argument('--lstm-length', type=int, default=10)
    model_arg.add_argument('--lstm-hidden', type=int, default=32)
    model_arg.add_argument('--lstm-layers', type=int, default=2)
    model_arg.add_argument('--fc-hidden', type=int, default=32)
    model_arg.add_argument('--fc-layers', type=int, default=3)
    model_arg.add_argument('--label-num', type=int, default=10)

    data_arg = parser.add_argument_group('Data')
    data_arg.add_argument('--data-directory', type=str, default=os.path.join(os.getcwd(),'dataset'), metavar='N', help='directory of data')
    # data_arg.add_argument('--dataset', type=str, default='sortofclevr3')

    train_arg = parser.add_argument_group('Train')
    train_arg.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 128)')
    train_arg.add_argument('--epochs1', type=int, default=10, metavar='N', help='number of first step epochs to train (default: 10)')
    train_arg.add_argument('--epochs2', type=int, default=10, metavar='N', help='number of second step epochs to train (default: 10)')
    train_arg.add_argument('--lr', type=float, default=1e-4, metavar='N', help='learning rate (default: 2.5e-4)')
    train_arg.add_argument('--log-directory', type=str, default=os.path.join(os.getcwd(),'experiment'), metavar='N', help='log directory')
    train_arg.add_argument('--device', type=int, default=0, metavar='N', help='number of cuda')
    train_arg.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    train_arg.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    train_arg.add_argument('--time-stamp', type=str, default=datetime.datetime.now().strftime("%y%m%d%H%M%S"), metavar='N', help='time of the run(no modify)')
    train_arg.add_argument('--memo', type=str, default='default', metavar='N', help='memo of the model')
    train_arg.add_argument('--load-model', type=str, default='000000000000', metavar='N', help='load previous model')
    train_arg.add_argument('--start-epoch', type=int, default=0, metavar='N', help='start-epoch number')

    args, unparsed = parser.parse_known_args()

    if not torch.cuda.is_available():
        args.device = torch.device('cpu')
    else:
        torch.cuda.set_device(args.device)
        # args.device = torch.device(args.device)

    config_list = [args.project, args.batch_size, args.epochs1, args.epochs2, args.lr, args.device,
                   args.code_size, args.lstm_input, args.lstm_length, args.lstm_hidden, args.lstm_layers,
                   args.fc_hidden, args.fc_layers, args.label_num,
                   args.memo]

    args.config = '_'.join(map(str, config_list))
    args.log = os.path.join(args.log_directory, args.time_stamp + args.config)
    print("Config:", args.config)

    return args

import time
import logging
from data_builder import *
import argparse
from networks_for_CIFAR import *
from networks_for_ImageNet import *
from utils import accuracy, AvgrageMeter, save_checkpoint, get_model, create_para_dict, read_param, record_param, deletStrmodule, randomize_gate
import sys
sys.path.append("..")
from layers import *
from tensorboardX import SummaryWriter
from torch.cuda import amp
from schedulers import *
from Regularization import *
import random

####################################################
# args                                             #
#                                                  #
####################################################

def get_args():
    parser = argparse.ArgumentParser("Gated Spiking Neural Networks")
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--eval-resume', type=str, default='./raw/models', help='path for eval model')
    parser.add_argument('--train-resume', type=str, default='./raw/models', help='path for train model')
    parser.add_argument('--batch-size', type=int, default=72, help='batch size')
    parser.add_argument('--epochs', type=int, default=200, help='total epochs used in training SuperNet')
    parser.add_argument('--learning-rate', type=float, default=1e-1, help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight-decay', type=float, default=4e-5, help='weight decay')
    parser.add_argument('--seed', type=int, default=9, metavar='S', help='random seed (default: 9)')
    parser.add_argument('--auto-continue', default=False, action='store_true', help='report frequency')
    parser.add_argument('--display-interval', type=int, default=10, help='per display-interval batches to' + ' display model training')
    parser.add_argument('--save-interval', type=int, default=10, help='per save-interval epochs to save model')

    parser.add_argument('--dataset-path', type=str, default='./dataset/', help='path to dataset')
    parser.add_argument('--train-dir', type=str, default='./imagenet/train', help='path to ImageNet training dataset')
    parser.add_argument('--val-dir', type=str, default='./imagenet/val', help='path to ImageNet validation dataset')

    parser.add_argument('--tunable-lif', default=False, action='store_true', help='use different learning rate for gating factors')
    parser.add_argument('--amp', default=False, action='store_true', help='use amp')
    parser.add_argument('--modeltag', type=str, default='SNN', help='decide the name of the experiment, this name will also be used as the checkpoint name')

    # configure the GLIF
    parser.add_argument('--gate', type=float, default=[0.6, 0.8, 0.6], nargs='+', help='initial gate')
    parser.add_argument('--static-gate', default=False, action='store_true', help='use static_gate')
    parser.add_argument('--static-param', default=False, action='store_true', help='use static_LIF_param')
    parser.add_argument('--channel-wise', default=False, action='store_true', help='use channel-wise')
    parser.add_argument('--softsimple', default=False, action='store_true', help='experiments on coarsely fused LIF')

    parser.add_argument('--soft-mode', default=False, action='store_true', help='use soft_gate')
    parser.add_argument('--t', type=int, default=3, help='the length of time window')
    parser.add_argument('--randomgate', default=False, action='store_true', help='activate uniform-randomly intialized gates')

    #define a dataset, default: cifar10
    parser.add_argument('--imagenet', default=False, action='store_true', help='experiments on ImageNet')
    parser.add_argument('--cifar100', default=False, action='store_true', help='experiments on cifar100')

    # define a model
    parser.add_argument('--stand18', default=False, action='store_true', help='use resnet18_stand')
    parser.add_argument('--cifarnet', default=False, action='store_true', help='use cifarnet')
    parser.add_argument('--MS18', default=False, action='store_true', help='experiments on ResNet-18MS')
    parser.add_argument('--MS34', default=False, action='store_true', help='experiments on ResNet-34MS')
    #ResNet-19 is the default option for CIFAR.
    #ResNet-34 is the default option for ImageNet.
    #To use any of the two models above, just clarify the task and DO NOT input any model commands. e.g., --stand18.

    args = parser.parse_args()

    return args




####################################################
# trainer & tester                                 #
#                                                  #
####################################################
def train(args, model, device, train_loader, optimizer, epoch, writer, criterion, scaler=None):
    layer_cnt, gate_score_list = None, None
    t1 = time.time()
    Top1, Top5 = 0.0, 0.0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.long)
        optimizer.zero_grad()
        if scaler is not None:
            with amp.autocast():
                output = model(data)
                loss = criterion(output, target)

        else:
            output = model(data)
            loss = criterion(output, target)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            loss.backward()
            optimizer.step()

        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        Top1 += prec1.item() / 100
        Top5 += prec5.item() / 100

        if batch_idx % args.display_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTop-1 = {:.6f}\tTop-5 = {:.6f}\tTime = {:.6f}'.format(
                epoch, batch_idx * len(data / steps), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(),
                Top1 / args.display_interval, Top5 / args.display_interval, time.time() - t1
                    )
                )
            Top1, Top5 = 0.0, 0.0
    print('time used in the epoch:{}'.format(time.time() - t1))

def test(args, model, device, test_loader, epoch, writer, criterion, modeltag, dict_params, best= None):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    layer_cnt, gate_score_list = None, None
    model.eval()# inactivate BN
    t1 = time.time()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.long)
            output = model(data)
            loss = criterion(output, target)
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            n = data.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

        logInfo = 'TEST Epoch {}: loss = {:.6f},\t'.format(epoch, objs.avg) + \
                  'Top-1  = {:.6f},\t'.format(top1.avg / 100) + \
                  'Top-5  = {:.6f},\t'.format(top5.avg / 100) + \
                  'val_time = {:.6f}\n'.format(time.time() - t1)
        logging.info(logInfo)
        writer.add_scalar('Top1_of_arch_{}'.format(0), top1.avg / 100, epoch)
        writer.add_scalar('Top5_of_arch_{}'.format(0), top5.avg / 100, epoch)

        record_param(args, model, dict=dict_params, epoch=epoch, modeltag=modeltag)
        if best is not None:
            if top1.avg / 100 > best['acc']:
                best['acc'], best['epoch'] = top1.avg / 100, epoch
                print('saving...')
                save_checkpoint({
                    'state_dict': model.state_dict(),
                    }, epoch, tag=modeltag)#"./raw/models"

            print('best acc is {} found in epoch {}'.format(best['acc'], best['epoch']))

            if epoch % 20 == 0:
                print('saving...')
                save_checkpoint({
                    'state_dict': model.state_dict(),
                }, epoch, tag=modeltag)  # "./raw/models"
        record_param(args, model, dict=dict_params, epoch=epoch, modeltag=modeltag, store=True)

        writer.add_scalar('Test_Loss_/epoch', objs.avg, epoch)
        writer.add_scalar('Test_Acc_/epoch', top1.avg / 100, epoch)

def seed_all(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
def main():
    args = get_args()
    seed_all(args.seed)

    if torch.cuda.device_count() > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    # Log
    log_format = '[%(asctime)s] %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%d %I:%M:%S')
    t = time.time()
    local_time = time.localtime(t)
    if not os.path.exists('./log'):
        os.mkdir('./log')
    fh = logging.FileHandler(os.path.join('log/train-{}{:02}{}'.format(local_time.tm_year % 2000, local_time.tm_mon, t)))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    epochs = 1#已经迭代的次数
    initial_dict = {'gate': [0.6, 0.8, 0.6], 'param': [tau, Vth, linear_decay, conduct],
                   't': steps, 'static_gate': True, 'static_param': False, 'time_wise': True, 'soft_mode': False}
    initial_dict['gate'] = args.gate
    initial_dict['static_gate'] = args.static_gate
    initial_dict['static_param'] = args.static_param
    initial_dict['time_wise'] = False
    initial_dict['soft_mode'] = args.soft_mode
    if args.t != steps:
        initial_dict['t']=args.t

    # In case time step is too large, we intuitively recommend to use the following code to alleviate the linear decay
    # initial_dict['param'][2] = initial_dict['param'][1]/(initial_dict['t'] * 2)


    use_gpu = False
    if torch.cuda.is_available():
        use_gpu = True
    if args.imagenet:
        train_loader, val_loader, _ = build_data(use_cifar10=False, dataset='imagenet',
                                                 batch_size=args.batch_size, train_val_split=False, workers=32,
                                                 imagenet_train_dir=args.train_dir, imagenet_val_dir=args.val_dir)
    elif args.cifar100:
        train_loader, val_loader, _ = build_data(use_cifar10=False, dataset='CIFAR100', dpath=args.dataset_path,
                                                 batch_size=args.batch_size, train_val_split=False, workers=16)
    else:
        #use cifar10
        train_loader, val_loader, _ = build_data(use_cifar10=True, dpath=args.dataset_path,
                                              batch_size=args.batch_size, train_val_split=False, workers=16)
    print('load data successfully')

    print(initial_dict)
    #prepare the model
    if args.imagenet:
        if args.MS18:
            model = ResNet_18_stand_CW_MS(lif_param=initial_dict, input_size=224, n_class=1000)
        elif args.MS34:
            model = ResNet_34_stand_CW_MS(lif_param=initial_dict, input_size=224, n_class=1000)
        elif args.channel_wise:
            model = ResNet_34_stand_CW(lif_param=initial_dict, input_size=224, n_class=1000)
        else:
            model = ResNet_34_stand(lif_param=initial_dict, input_size=224, n_class=1000)
    elif args.cifar100:
        if args.cifarnet:
            model = CIFARNet(lif_param=initial_dict, input_size=32, n_class=100)
        elif args.stand18:
            if args.channel_wise:
                if args.softsimple:
                    model =ResNet_18_stand_CW_softsimple(lif_param=initial_dict, input_size=32, n_class=100)
                else:
                    model = ResNet_18_stand_CW(lif_param=initial_dict, input_size=32, n_class=100)
            else:
                model = ResNet_18_stand(lif_param=initial_dict, input_size=32, n_class=100)
        else:
            if args.channel_wise: #resnet -19
                if args.softsimple:
                    model =ResNet_19_stand_CW_softsimple(lif_param=initial_dict, input_size=32, n_class=100)
                else:
                    model = ResNet_19_cifar_CW(lif_param=initial_dict, input_size=32, n_class=100)
            else:
                model = ResNet_19_cifar(lif_param=initial_dict, input_size=32, n_class=100)
    else: #cifar10
        if args.stand18:
            if args.channel_wise:
                if args.softsimple:
                    model =ResNet_18_stand_CW_softsimple(lif_param=initial_dict, input_size=32, n_class=10)
                else:
                    model = ResNet_18_stand_CW(lif_param=initial_dict, input_size=32, n_class=10)
            else:
                model = ResNet_18_stand(lif_param=initial_dict, input_size=32, n_class=10)
        elif args.cifarnet:
            model = CIFARNet(lif_param=initial_dict, input_size=32, n_class=10)
        elif args.channel_wise: # resnet-19
            model = ResNet_19_cifar_CW(lif_param=initial_dict, input_size=32, n_class=10)
        else:
            model = ResNet_19_cifar(lif_param=initial_dict, input_size=32, n_class=10)

    if args.randomgate:
        randomize_gate(model)
        # model.randomize_gate
        print('randomized gate')

    modeltag = args.modeltag
    writer = SummaryWriter('./summaries/' + modeltag)
    print(model)
    dict_params = create_para_dict(args, model)
    # recording the initial GLIF parameters
    record_param(args, model, dict=dict_params, epoch=0, modeltag=modeltag)
    # classify GLIF-related params
    choice_param_name = ['alpha', 'beta', 'gamma']
    lifcal_param_name = ['tau', 'Vth', 'leak', 'conduct', 'reVth']
    all_params = model.parameters()
    lif_params = []
    lif_choice_params = []
    lif_cal_params = []

    for pname, p in model.named_parameters():
        if pname.split('.')[-1] in choice_param_name:
            lif_params.append(p)
            lif_choice_params.append(p)
        elif pname.split('.')[-1] in lifcal_param_name:
            lif_params.append(p)
            lif_cal_params.append(p)
    # fetch id
    params_id = list(map(id, lif_params))
    other_params = list(filter(lambda p: id(p) not in params_id, all_params))
    # optimizer & scheduler
    if args.tunable_lif:
        init_lr_diff = 10
        if args.imagenet:
            init_lr_diff = 1

        optimizer = torch.optim.SGD([
                {'params': other_params},
                {'params': lif_cal_params, "weight_decay": 0.},
                {'params': lif_choice_params, "weight_decay": 0., "lr":args.learning_rate / init_lr_diff}
            ],
                lr=args.learning_rate,
                momentum=0.9,
                weight_decay=args.weight_decay
            )
        scheduler = CosineAnnealingLR_Multi_Params_soft(optimizer,
                                                            T_max=[args.epochs, args.epochs, int(args.epochs)])
    else:
        optimizer = torch.optim.SGD([
            {'params': other_params},
            {'params': lif_params, "weight_decay": 0.}
        ],
            lr=args.learning_rate,
            momentum=0.9,
            weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    criterion = Loss(args)
    device = torch.device("cuda" if use_gpu else "cpu")
    #Distributed computation
    if torch.cuda.is_available():
        loss_function = criterion.cuda()
    else:
        loss_function = criterion.cpu()

    if args.auto_continue:
        lastest_model = get_model(modeltag)
        if lastest_model is not None:
            checkpoint = torch.load(lastest_model, map_location='cpu')
            epochs = checkpoint['epoch']
            if torch.cuda.device_count() > 1:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                checkpoint = deletStrmodule(checkpoint)
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            print('load from checkpoint, the epoch is {}'.format(epochs))
            dict_params = read_param(epoch=epochs, modeltag=modeltag)
            for i in range(epochs):
                scheduler.step()
            epochs += 1


    best = {'acc': 0., 'epoch': 0}

    if args.eval:
        lastest_model = get_model(modeltag, addr=args.eval_resume)
        if lastest_model is not None:
            epochs = -1
            checkpoint = torch.load(lastest_model, map_location='cpu')
            if args.imagenet:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                checkpoint = deletStrmodule(checkpoint)
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            if torch.cuda.device_count() > 1:
                device = torch.device(local_rank)
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                model = nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[local_rank],
                                                            output_device=local_rank,
                                                            find_unused_parameters=False)
            else:
                model = model.to(device)
            test(args, model, device, val_loader, epochs, writer, criterion=loss_function,
                 modeltag=modeltag, best=best, dict_params=dict_params)
        else:
            print('no model detected')
        exit(0)


    if torch.cuda.device_count() > 1:
        device = torch.device(local_rank)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[local_rank], output_device=local_rank,
                                                    find_unused_parameters=False)
    else:
        model = model.to(device)


    print('the random seed is {}'.format(args.seed))

    # amp
    if args.amp:
        scaler = amp.GradScaler()
    else:
        scaler = None

    while (epochs <= args.epochs):
        train(args, model, device, train_loader, optimizer, epochs, writer, criterion=loss_function,
              scaler=scaler)
        if epochs % 1 == 0:
            test(args, model, device, val_loader, epochs, writer, criterion=loss_function,
                 modeltag=modeltag, best=best, dict_params=dict_params)
        else:
            pass
        print('and lr now is {}'.format(scheduler.get_last_lr()))
        scheduler.step()
        epochs += 1
    writer.close()



if __name__ == "__main__":
    main()

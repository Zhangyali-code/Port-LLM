import time
from torch.utils.tensorboard import SummaryWriter
import torch

from model_prompt import *
from get_data_new_new import *
from LR import *
from calculate_erro import *

import pickle

import gc

gc.collect()

writer = SummaryWriter(log_dir='/home/edu/ZhangYali/MA_NEW_NEW/logs_new_less_para_600/prompt')

# obtain data
trainloader, testloader = get_data()
print('The length of datasets:', len(trainloader), len(testloader))

# model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model().to(device=device)

total = sum([param.nelement() for param in model.parameters()])
print("Number of parameter: %.5fM" % (total / 1e6))
total_learn = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of learnable parameter: %.5fM" % (total_learn / 1e6))

# loss function
criterion = nn.MSELoss(reduction='mean').to(device=device)
mae = nn.L1Loss(reduction='mean').to(device=device)

# optimizer
epochs = 600
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3, betas=(0.9, 0.999))
scheduler = WarmUpCosineAnnealingLR(optimizer=optimizer, T_warmup=100 * len(trainloader),
                                    T_max=epochs * len(trainloader), eta_min=1e-6)


def log_gradient_stats(model, writer, step):
    total_norm = 0.0  # Accumulates squared L2 norms of all gradients
    # grad_stats = {}    # Dictionary to store per-layer gradient stats

    for name, param in model.named_parameters():
        if param.grad is not None:
            # Calculate L2 norm of current parameter's gradients
            param_norm = param.grad.data.norm(2).item()
            total_norm += param_norm ** 2

            # Log gradient norms and distributions to TensorBoard
            # writer.add_scalar(f'Gradients/{name}_norm', param_norm, step)
            # writer.add_histogram(f'Gradients/{name}_hist', param.grad, step)

            # Collect detailed gradient statistics

    # Calculate and log total gradient norm
    total_norm = total_norm ** 0.5
    # writer.add_scalar('Gradients/Total_Norm', total_norm, step)

    return total_norm

# training
def train(input, target, model, optimizer):
    # the mode of trainning
    model.train()

    # initialize the grad of net
    model.zero_grad()

    # trainning
    # inputs:(B,T,2,100,50)
    # gen_target:(B,F,2,100,50)
    gen_target = model(input)

    # loss
    loss = criterion(gen_target, target) / criterion(target, torch.zeros_like(target))

    # backward
    loss.backward()

    #
    total_norm = log_gradient_stats(model, writer, iter_train)


    # update G
    optimizer.step()
    scheduler.step()

    # j=j+1

    return gen_target, loss.item(), optimizer.param_groups[0]['lr'], total_norm


##################
iter_train = 0
iter_test = 0

#
with torch.no_grad():
    test_input, _, _ = next(iter(trainloader))
    writer.add_graph(model, test_input.to(device))

# offline training+online testing
print('-------------Start training!-------------')
for epoch in range(1, epochs + 1):

    print('--------The{}Epoch---------'.format(epoch))
    print("start_time", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    for i, (input, target, target_ref) in enumerate(trainloader):
        #
        torch.cuda.empty_cache()
        # input:(B,T,2,100,50)
        # target:(B,F,2,100,50)
        # target_ref:(B,F,2)
        input, target, target_ref = input.to(device), target.to(device), target_ref.to(device)
        #
        gen_target, loss, lr, total_norm = train(input, target, model, optimizer)

        ##
        _, _, acc_loss = calculate_erro(target_ref, gen_target, target, criterion, device)

        #
        if i % 10 == 0:
            print('[Epoch %d/%d] [Batch %d/%d] [loss:%f] [acc_loss:%f] [total_norm:%.2e]' % (
                epoch, epochs, i, len(trainloader), loss, acc_loss.item(), total_norm))
        writer.add_scalar('Learning Rate During Training', lr, iter_train)
        writer.add_scalar('Loss During Training', loss, iter_train)
        writer.add_scalar('db During Training/dB', 10 * math.log10(loss), iter_train)
        writer.add_scalar('acc_loss During Training/%', 10 * math.log10(acc_loss.item()), iter_train)
        writer.add_scalar('Total_Norm During Training', total_norm, iter_train)
        ##############
        del input, target, target_ref, gen_target, loss, acc_loss, lr
        gc.collect()
        torch.cuda.empty_cache()
        ##############
        iter_train = iter_train + 1
    print("end_time: ", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), '\n')

    with torch.no_grad():
        print('testing!!')
        model.eval()
        for ii, (test_input, test_target, test_target_ref) in enumerate(testloader):
            #
            torch.cuda.empty_cache()
            #
            test_input, test_target, test_target_ref = test_input.to(device), test_target.to(
                device), test_target_ref.to(device)

            # fake_target: (B,F,2,100,50)
            fake_target = model(test_input.detach())

            # test_loss
            test_loss = criterion(fake_target, test_target) / criterion(test_target, torch.zeros_like(test_target))

            # acc
            acc = (1 - mae(fake_target, test_target) / mae(test_target, torch.zeros_like(test_target))) * 100

            #
            _, _, acc_loss = calculate_erro(test_target_ref, fake_target, test_target, criterion, device)
            #
            if ii % 10 == 0:
                print('[Epoch %d/%d] [Batch %d/%d] [test_loss:%f] [acc*100:%f] [acc_loss:%f]' % (
                    epoch, epochs, ii, len(testloader), test_loss.item(), acc, acc_loss.item()))

            writer.add_scalar('Loss During testing', test_loss.item(), iter_test)
            writer.add_scalar('dB During testing/dB', 10 * math.log10(test_loss.item()), iter_test)
            writer.add_scalar('Accurancy During Testing/%', acc, iter_test)
            writer.add_scalar('acc_loss During testing/%', 10 * math.log10(acc_loss.item()), iter_test)
            ##############
            del test_input, test_target, test_target_ref, fake_target, test_loss, acc, acc_loss
            gc.collect()
            torch.cuda.empty_cache()
            ##############
            iter_test = iter_test + 1

print('--------------Finish training!------------')

#
torch.save(model.state_dict(), '/home/edu/ZhangYali/MA_NEW_NEW/Model_new_less_para_600/prompt/model.pth')
print('model saved!')
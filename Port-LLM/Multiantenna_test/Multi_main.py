from ..train.model import Model
from Multi_utils import *
from Multi_calcu import *

import torch.nn as nn
import time

input, target, target_ref, _, _ = utils()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input = input.to(device)
target = target.to(device)
target_ref = target_ref.to(device)

# loss function
criterion = nn.MSELoss(reduction='mean').to(device=device)
mae = nn.L1Loss(reduction='mean').to(device=device)

# model
model = Model().to(device=device)
model.load_state_dict(torch.load('/home/edu/ZhangYali/MA_NEW_NEW/Model_acc_loss/model.pth'))  # load trained network parameters

# Initialize some parameters
db_loss = []
db_acc_loss = []
accuracy = []
gen_h = torch.zeros(40, 8, 2, 256)

with torch.no_grad():
    #
    model.eval()
    #10*8*2*100*50*16
    for i in range(input.shape[0]):
        print(f'BATCH {i}_th:',i)
        Loss = torch.zeros(1).to(device=device)
        Acc = torch.zeros(1).to(device=device)
        gen_multi_antenna_h = torch.zeros(1, 8, 2, 100, 50, 256).to(device=device)
        ####
        for j in range(256):
            start_time = time.perf_counter()  # record start time
            gen_target = model(input[i, :, :, :, :, j].unsqueeze(0))
            end_time = time.perf_counter()  # record end time
            inference_time = (end_time - start_time) * 1000  # c  onvert to milliseconds
            print(f'Inference time for sample {j}: {inference_time:.6f} ms')

            # lo
            test_loss = 10 * math.log10(criterion(gen_target, target[i, :, :, :, :, j].unsqueeze(0)) / criterion(target[i, :, :, :, :, j].unsqueeze(0), torch.zeros_like(target[i, :, :, :, :, j].unsqueeze(0))).item())
            Loss = Loss + test_loss
            # acc
            acc=(1 - mae(gen_target, target[i, :, :, :, :, j].unsqueeze(0)) / mae(
                target[i, :, :, :, :, j].unsqueeze(0), torch.zeros_like(target[i, :, :, :, :, j].unsqueeze(0)))) * 100
            Acc = Acc + acc

            #gen_target:B*F*2*100*50
            gen_multi_antenna_h[:, :, :, :, :, j] = gen_target

        px, py, acc_loss, gen_hh = calculate_erro(target_ref[i, :, :, :].unsqueeze(0), gen_multi_antenna_h, target[i, :, :, :, :, :].unsqueeze(0), criterion, device)
        db_acc_loss.append(acc_loss)
        db_loss.append(Loss.item() / 256)
        accuracy.append(Acc.item() / 256)
        gen_h[i, :, :, :] = gen_hh  #the channel corresponding to the predicted port

print('finished!')

log_db_loss = open(r'/home/edu/ZhangYali/MA_NEW_NEW/SE_results/32_8/V150/our_model/db_loss.txt', mode='a', encoding='utf-8')
print(db_loss, file=log_db_loss)
log_db_loss.close()  # close file

log_acc = open(r'/home/edu/ZhangYali/MA_NEW_NEW/SE_results/32_8/V150/our_model/accuracy.txt', mode='a', encoding='utf-8')
print(accuracy, file=log_acc)
log_acc.close()

log_db_acc_loss = open(r'/home/edu/ZhangYali/MA_NEW_NEW/SE_results/32_8/V150/our_model/db_acc_loss.txt', mode='a', encoding='utf-8')
print(db_acc_loss, file=log_db_acc_loss)
log_db_acc_loss.close()

log_gen_h = open(r'/home/edu/ZhangYali/MA_NEW_NEW/SE_results/32_8/V150/our_model/gen_h.txt', mode='a', encoding='utf-8')
print(gen_h, file=log_gen_h)
log_gen_h.close()
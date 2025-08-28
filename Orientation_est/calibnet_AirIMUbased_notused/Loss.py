import torch
import math


def log_cosh_loss(pred, targ):
    # log(cosh(x)) with small epsilon to avoid numerical instability
    # x = pred - target
    eps = 1e-12
    pred = pred / (pred.norm(dim=1, keepdim=True) + eps)
    targ = targ / (targ.norm(dim=1, keepdim=True) + eps)

    dot_product = torch.abs(torch.sum(pred * targ, dim=1))  # (B,), quaternion dot -> angle dist
    cosangle = torch.clamp(dot_product, -1.0 + eps, 1.0 - eps)

    angle = 2 * torch.acos(cosangle) * 180.0 / torch.pi
    loss_interval = angle.sum(dim=1)
    return loss_interval


def multiscaleloss(preds, targ, n):
    device = preds.device
    window_size = preds.shape[2]
    loss_interval = 32

    # n이 2의 거듭제곱인지 확인
    if not (n > 0 and (n & (n - 1)) == 0): # bit 연산자 활용한 거듭제곱 판별
        print("n이 2의 거듭제곱이 아님")
        return None

    #L = torch.tensor(0., device=device)
    L = 0

    loss_window_seq = int(window_size / loss_interval)
    for j in range(loss_window_seq):
        start = j * 32
        mid = start + 16
        end = (j+1) * 32
        L16_1 = log_cosh_loss(preds[:,:,start:mid], targ[:,:,start:mid])
        L16_2 = log_cosh_loss(preds[:,:,mid:end], targ[:,:,mid:end])
        L32 = log_cosh_loss(preds[:,:,start:end], targ[:,:,start:end])

        L = L + L16_1 + L16_2 + L32


    '''levels = int(math.log2(n)) # n = 2, levels = 1
    for j in range(levels+1):
        segments = n // (2 ** j) # segments = 2,1
        for i in range(segments):  # [0:2dt] +[0:dt] + [dt:2dt]
            start = int(i * 2 * dt / (segments))
            end = int((i + 1) * 2 * dt / (segments))
            #print(start, end, i, j, segments)
            L = L + log_cosh_loss(preds[:,:,start:end], targ[:,:,start:end])
            print(log_cosh_loss(preds[start:end], targ[start:end]), "loss", j, i)'''

    return L


'''def loss(preds, targ):
    L = 0
    for i in range(preds.size(0)):
        L += np.log(np.cosh(preds[i] - targ[i]))
    return L

def multiscaleloss(preds, targ, dt):
    n = preds.size(0) / dt
    if np.log2(n).type != int:
        print("n이 2의 거듭제곱이 아님")
        return None
    else:
        L = 0
        for j in range(np.log2(n)):
            for i in range(n / 2**j):
                L += loss(preds[i*dt:(i+1)*dt],targ[i*dt:(i+1)*dt])
        return L'''
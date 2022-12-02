def triangle_func(epoch):
    max_epoch = 50
    max_lr_epoch = int(max_epoch / 2)  # 삼각형의 꼭짓점 (높이 1)
    grad = 1 / max_lr_epoch  # 기울기
    if max_lr_epoch > epoch:
        return grad * epoch
    else:
        return max(0, 1 - grad * (epoch - max_lr_epoch))

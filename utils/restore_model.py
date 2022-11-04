import torch


def restore_model(generate, kernel):
    premodel = torch.load("./orl.pkl")
    idx = len(kernel)
    i = 0
    x = generate.state_dict()
    for now, pre in zip(generate.state_dict(), premodel):
        generate.state_dict()[now].copy_(premodel[pre])
        y = generate.state_dict()
        i += 1
        if i == idx*2:
            break

    return generate

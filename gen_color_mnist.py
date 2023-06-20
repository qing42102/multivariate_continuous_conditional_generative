import torchvision
import torch
import numpy as np
import torchvision.transforms as transforms
import tqdm
import argparse
import os


def save_images(transform, transform_name, data_path, dir_name):
    fulltrainset = torchvision.datasets.MNIST(
        root=data_path, train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        fulltrainset, batch_size=2000, shuffle=False, pin_memory=True
    )

    test_set = torchvision.datasets.MNIST(
        root=data_path, train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        test_set, batch_size=2000, shuffle=False, pin_memory=True
    )

    color_data_x, color_data_y = gen_fgbgcolor_data(
        trainloader, img_size=(3, 28, 28), cpr=args.cpr, noise=10.0
    )
    train_mask = get_mask(trainloader)
    np.save(dir_name + "/colored_mnist_" + transform_name + "_train.npy", color_data_x)
    np.save(dir_name + "/colored_mnist_target_train.npy", color_data_y)
    np.save(
        dir_name + "/colored_mnist_" + transform_name + "_mask_train.npy", train_mask
    )

    color_data_x, color_data_y = gen_fgbgcolor_data(
        testloader, img_size=(3, 28, 28), cpr=None, noise=10.0
    )
    test_mask = get_mask(testloader)
    np.save(dir_name + "/colored_mnist_" + transform_name + "_test.npy", color_data_x)
    np.save(dir_name + "/colored_mnist_target_test.npy", color_data_y)
    np.save(dir_name + "/colored_mnist_" + transform_name + "_mask_test.npy", test_mask)


def get_mask(loader):
    mask_array = []
    for batch_x, batch_y in loader:
        for x in batch_x:
            mask = np.zeros(x.shape)
            mask[x > 0] = 1
            mask_array.append(mask)

    return np.array(mask_array, dtype=np.float32)


# generate color codes
def get_color_codes(cpr):
    C = np.random.rand(len(cpr), nb_classes, 3)
    C = C / np.max(C, axis=2)[:, :, None]
    print(C.shape)
    return C


def gen_fgbgcolor_data(loader, img_size=(3, 28, 28), cpr=[0.5, 0.5], noise=10.0):
    if cpr is not None:
        assert sum(cpr) == 1, "--cpr must be a non-negative list which sums to 1"
        Cfg = get_color_codes(cpr)
        Cbg = get_color_codes(cpr)
    else:
        Cfg = get_color_codes([1])
        Cbg = get_color_codes([1])
    tot_iters = len(loader)
    for i in tqdm.tqdm(range(tot_iters), total=tot_iters):
        x, targets = next(iter(loader))
        assert (
            len(x.size()) == 4
        ), "Something is wrong, size of input x should be 4 dimensional (B x C x H x W; perhaps number of channels is degenrate? If so, it should be 1)"
        bs = targets.shape[0]

        x = (x * 255).type("torch.FloatTensor")
        x_rgb = torch.ones(x.size(0), 3, x.size()[2], x.size()[3]).type(
            "torch.FloatTensor"
        )
        x_rgb = x_rgb * x
        x_rgb_fg = 1.0 * x_rgb

        color_choice = (
            np.argmax(np.random.multinomial(1, cpr, targets.shape[0]), axis=1)
            if cpr is not None
            else 0
        )
        c_fg = (
            Cfg[color_choice, targets]
            if cpr is not None
            else Cfg[color_choice, np.random.randint(nb_classes, size=targets.shape[0])]
        )
        c_fg = c_fg.reshape(-1, 3, 1, 1)
        c_fg = torch.from_numpy(c_fg).type("torch.FloatTensor")
        x_rgb_fg[:, 0] = x_rgb_fg[:, 0] * c_fg[:, 0]
        x_rgb_fg[:, 1] = x_rgb_fg[:, 1] * c_fg[:, 1]
        x_rgb_fg[:, 2] = x_rgb_fg[:, 2] * c_fg[:, 2]

        bg = 255 - x_rgb
        # c = C[targets] if np.random.rand()>cpr else C[np.random.randint(C.shape[0], size=targets.shape[0])]
        color_choice = (
            np.argmax(np.random.multinomial(1, cpr, targets.shape[0]), axis=1)
            if cpr is not None
            else 0
        )
        c_bg = (
            Cbg[color_choice, targets]
            if cpr is not None
            else Cbg[color_choice, np.random.randint(nb_classes, size=targets.shape[0])]
        )
        c_bg = c_bg.reshape(-1, 3, 1, 1)
        c_bg = torch.from_numpy(c_bg).type("torch.FloatTensor")
        bg[:, 0] = bg[:, 0] * c_bg[:, 0]
        bg[:, 1] = bg[:, 1] * c_bg[:, 1]
        bg[:, 2] = bg[:, 2] * c_bg[:, 2]

        x_rgb = x_rgb_fg + bg
        x_rgb = x_rgb + torch.tensor((noise) * np.random.randn(*x_rgb.size())).type(
            "torch.FloatTensor"
        )
        x_rgb = torch.clamp(x_rgb, 0.0, 255.0)

        targets = torch.cat(
            (torch.unsqueeze(targets, dim=1), torch.squeeze(c_bg), torch.squeeze(c_fg)),
            dim=1,
        )
        targets = targets.numpy()

        if i == 0:
            color_data_x = np.zeros((bs * tot_iters, *img_size))
            color_data_y = np.zeros((bs * tot_iters, targets.shape[1]))
        color_data_x[i * bs : (i + 1) * bs] = x_rgb / 255.0
        color_data_y[i * bs : (i + 1) * bs] = targets

        color_data_x = color_data_x.astype(np.float32)
        color_data_y = color_data_y.astype(np.float32)

    return color_data_x, color_data_y


if "__main__" == __name__:
    data_path = "mnist/"

    np.random.seed(0)
    torch.manual_seed(0)

    parser = argparse.ArgumentParser(description="Generate colored MNIST")

    # Hyperparams
    parser.add_argument(
        "--cpr",
        nargs="+",
        type=float,
        default=[0.5, 0.5],
        help="color choice is made corresponding to a class with these probability",
    )
    args = parser.parse_args()

    so2_transform = transforms.Compose(
        [transforms.RandomAffine(90, translate=(0.0, 0.0)), transforms.ToTensor()]
    )

    se2_transform = transforms.Compose(
        [transforms.RandomAffine(90, translate=(0.25, 0.25)), transforms.ToTensor()]
    )
    vanilla_transform = transforms.Compose([transforms.ToTensor()])

    nb_classes = 10

    dir_name = (
        data_path
        + "cmnist/"
        + "fgbg_cmnist_cpr"
        + "-".join(str(p) for p in args.cpr)
        + "/"
    )
    print(dir_name)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    save_images(vanilla_transform, "init", data_path, dir_name)
    # save_images(so2_transform, "so2", data_path, dir_name)
    # save_images(se2_transform, "se2", data_path, dir_name)

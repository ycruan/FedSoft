import numpy as np
from torch.utils.data import ConcatDataset

from clients import Client


def create_clients_mnist_and_mnist_rotation_90(client_solver):
    from datawrappers.mnist_rotation import (MnistRotationLocal90, MnistRotationDev90, MnistRotationSampler)
    from datawrappers.mnist import MnistLocal, MnistSampler, MnistDev
    num_clients = 100
    num_samples_min = 100
    num_samples_max = 201

    mnist_sampler = MnistSampler()
    mnist_rotation_sampler = MnistRotationSampler()
    client_vec = []
    mixture_vec = []
    for i in range(num_clients):
        num_samples = np.random.randint(num_samples_min, num_samples_max)
        # num_samples_mnist = int(np.random.rand() * num_samples)
        # num_samples_mnist = int((0.3 if i < (num_clients / 2) else 0.7) * num_samples)
        num_samples_mnist = int((1.0 * i / num_clients + 0.5 / num_clients) * num_samples)
        num_samples_mnist_rotation = num_samples - num_samples_mnist
        mnist_ds = MnistLocal(mnist_sampler.sample(class_vec=range(10), num_samples=num_samples_mnist))
        mnist_rotation_ds = MnistRotationLocal90(
            mnist_rotation_sampler.sample(class_vec=range(10), num_samples=num_samples_mnist_rotation))
        tag = 'Mixture of mnist and mnist-rotation-90 data. {} full mnist, {} full mnist rotation, total {} data points.'.format(
            num_samples_mnist, num_samples_mnist_rotation, num_samples)
        client = Client(ID=i, ds=ConcatDataset([mnist_ds, mnist_rotation_ds]), solver=client_solver, tag=tag)
        client_vec.append(client)
        mixture_vec.append([num_samples_mnist, num_samples_mnist_rotation, num_samples])

    test_mnist_ds = MnistDev()
    test_mnist_rotation_ds = MnistRotationDev90()

    param_dict = {
        'ds_description': '(2) Full mnist and full mnist-rotation-90',
        'num_clusters': 2,
        'num_classes': 10,
        'num_clients': num_clients,
        'num_samples_min': num_samples_min,
        'num_samples_max': num_samples_max,
        'num_samples_test_each': len(test_mnist_ds),
        'client_tags': {},
        'mixture': {}
    }
    for k, client in enumerate(client_vec):
        param_dict['client_tags']['client_' + str(client.ID)] = client.tag
        param_dict['mixture']['client_' + str(client.ID)] = mixture_vec[k]

    return client_vec, [test_mnist_ds, test_mnist_rotation_ds], param_dict


def create_clients_cifar_and_cifar_rotation_90(client_solver):
    from datawrappers.cifar_rotation import (CifarRotationLocal90, CifarRotationDev90, CifarRotationSampler)
    from datawrappers.cifar import CifarLocal, CifarSampler, CifarDev
    num_clients = 20
    num_samples_min = 3000
    num_samples_max = 3501

    cifar_sampler = CifarSampler()
    cifar_rotation_sampler = CifarRotationSampler()
    client_vec = []
    mixture_vec = []
    for i in range(num_clients):
        num_samples = np.random.randint(num_samples_min, num_samples_max)
        # num_samples_cifar = int(np.random.rand() * num_samples)
        num_samples_cifar = int((.3 if i < (num_clients / 2) else .7) * num_samples)
        # num_samples_cifar = int((1.0 * i / num_clients + 0.5 / num_clients) * num_samples)
        num_samples_cifar_rotation = num_samples - num_samples_cifar
        cifar_ds = CifarLocal(cifar_sampler.sample(class_vec=range(10), num_samples=num_samples_cifar))
        cifar_rotation_ds = CifarRotationLocal90(
            cifar_rotation_sampler.sample(class_vec=range(10), num_samples=num_samples_cifar_rotation))
        tag = 'Mixture of cifar and cifar-rotation-90 data. {} full cifar, {} full cifar rotation, total {} data points.'.format(
            num_samples_cifar, num_samples_cifar_rotation, num_samples)
        client = Client(ID=i, ds=ConcatDataset([cifar_ds, cifar_rotation_ds]), solver=client_solver, tag=tag)
        client_vec.append(client)
        mixture_vec.append([num_samples_cifar, num_samples_cifar_rotation, num_samples])

    test_cifar_ds = CifarDev()
    test_cifar_rotation_ds = CifarRotationDev90()

    param_dict = {
        'ds_description': '(2) Full cifar and full cifar-rotation-90',
        'num_clusters': 2,
        'num_classes': 10,
        'num_clients': num_clients,
        'num_samples_min': num_samples_min,
        'num_samples_max': num_samples_max,
        'num_samples_test_each': len(test_cifar_ds),
        'client_tags': {},
        'mixture': {}
    }
    for k, client in enumerate(client_vec):
        param_dict['client_tags']['client_' + str(client.ID)] = client.tag
        param_dict['mixture']['client_' + str(client.ID)] = mixture_vec[k]

    return client_vec, [test_cifar_ds, test_cifar_rotation_ds], param_dict


def create_clients_mnist_rotation_4set(client_solver):
    from datawrappers.mnist_rotation import (MnistRotationLocal90, MnistRotationLocal180, MnistRotationLocal270,
                                             MnistRotationDev90, MnistRotationDev180, MnistRotationDev270,
                                             MnistRotationSampler)
    from datawrappers.mnist import MnistLocal, MnistSampler, MnistDev
    num_clients = 100
    num_samples_min = 100
    num_samples_max = 201

    mnist_sampler = MnistSampler()
    mnist_rotation_sampler = MnistRotationSampler()
    client_vec = []
    for i in range(num_clients):
        num_samples = np.random.randint(num_samples_min, num_samples_max)
        frac_vec = [0., 1.]
        for _ in range(3):
            frac_vec.append(np.random.rand())
        frac_vec.sort()
        frac_vec = [frac_vec[j] - frac_vec[j - 1] for j in [1, 2, 3, 4]]
        num_samples_mnist0 = int(frac_vec[0] * num_samples)
        num_samples_mnist90 = int(frac_vec[1] * num_samples)
        num_samples_mnist180 = int(frac_vec[2] * num_samples)
        num_samples_mnist270 = num_samples - num_samples_mnist0 - num_samples_mnist90 - num_samples_mnist180
        mnist0_ds = MnistLocal(mnist_sampler.sample(class_vec=range(10), num_samples=num_samples_mnist0))
        mnist90_ds = MnistRotationLocal90(
            mnist_rotation_sampler.sample(class_vec=range(10), num_samples=num_samples_mnist90))
        mnist180_ds = MnistRotationLocal180(
            mnist_rotation_sampler.sample(class_vec=range(10), num_samples=num_samples_mnist180))
        mnist270_ds = MnistRotationLocal270(
            mnist_rotation_sampler.sample(class_vec=range(10), num_samples=num_samples_mnist270))
        tag = 'Mixture of mnist and 3 other mnist rotation data. {} full mnist-0, {} full mnist-90, {} full mnist-180. {} full mnist-270 total {} data points.'.format(
            num_samples_mnist0, num_samples_mnist90, num_samples_mnist180, num_samples_mnist270, num_samples)
        client = Client(ID=i, ds=ConcatDataset([mnist0_ds, mnist90_ds, mnist180_ds, mnist270_ds]), solver=client_solver,
                        tag=tag)
        client_vec.append(client)

    test_mnist0_ds = MnistDev()
    test_mnist90_ds = MnistRotationDev90()
    test_mnist180_ds = MnistRotationDev180()
    test_mnist270_ds = MnistRotationDev270()

    param_dict = {
        'ds_description': '(4) Full mnist rotations -0 -90 -180 -270, sizes allocated randomly',
        'num_clusters': 4,
        'num_classes': 10,
        'num_clients': num_clients,
        'num_samples_min': num_samples_min,
        'num_samples_max': num_samples_max,
        'num_samples_test_each': len(test_mnist0_ds),
        'client_tags': {}
    }
    for client in client_vec:
        param_dict['client_tags']['client_' + str(client.ID)] = client.tag

    return client_vec, [test_mnist0_ds, test_mnist90_ds, test_mnist180_ds, test_mnist270_ds], param_dict


def create_clients_letters_lower_and_upper(client_solver):
    from datawrappers.letters_lower import LettersLowerLocal, LettersLowerSampler, LettersLowerDev, \
        LettersLowerDevSampler
    from datawrappers.letters_upper import LettersUpperLocal, LettersUpperSampler, LettersUpperDev, \
        LettersUpperDevSampler
    num_clients = 100  # total number of clients
    num_samples_min = 100
    num_samples_max = 201
    num_dev_samples = 20000

    lower_sampler = LettersLowerSampler()
    upper_sampler = LettersUpperSampler()
    client_vec = []
    mixture_vec = []
    for i in range(num_clients):
        num_samples = np.random.randint(num_samples_min, num_samples_max)
        # num_samples_lower = int(np.random.rand() * num_samples)
        num_samples_lower = int((.3 if i < (num_clients / 2) else .7) * num_samples)
        # num_samples_lower = int((1.0 * i / num_clients + 0.5 / num_clients) * num_samples)
        num_samples_upper = num_samples - num_samples_lower
        lower_ds = LettersLowerLocal(lower_sampler.sample(class_vec=range(26), num_samples=num_samples_lower))
        upper_ds = LettersUpperLocal(upper_sampler.sample(class_vec=range(26), num_samples=num_samples_upper))
        tag = 'Mixture of lower and upper case alphabet letters. {} full 26 lowercase letters, {} full 26 uppercase letters, total {} data points.'.format(
            num_samples_lower, num_samples_upper, num_samples)
        client = Client(ID=i, ds=ConcatDataset([lower_ds, upper_ds]), solver=client_solver, tag=tag)
        client_vec.append(client)
        mixture_vec.append([num_samples_lower, num_samples_upper, num_samples])

    lower_dev_sampler = LettersLowerDevSampler()
    upper_dev_sampler = LettersUpperDevSampler()
    test_lower_ds = LettersLowerDev(lower_dev_sampler.sample(range(26), num_dev_samples))
    test_upper_ds = LettersUpperDev(upper_dev_sampler.sample(range(26), num_dev_samples))

    param_dict = {
        'ds_description': '(2) 26 lowercase and 26 uppercase alphabet letters',
        'num_clusters': 2,
        'num_classes': 26,
        'num_clients': num_clients,
        'num_samples_min': num_samples_min,
        'num_samples_max': num_samples_max,
        'num_samples_test': 'lower = {}, upper = {}'.format(num_dev_samples, num_dev_samples),
        'client_tags': {},
        'mixture': {}
    }
    for k, client in enumerate(client_vec):
        param_dict['client_tags']['client_' + str(client.ID)] = client.tag
        param_dict['mixture']['client_' + str(client.ID)] = mixture_vec[k]

    return client_vec, [test_lower_ds, test_upper_ds], param_dict


def create_clients_letters_rotation_8set(client_solver):
    from datawrappers.letters_lower import (LettersLowerLocal, LettersLowerLocal90, LettersLowerLocal180,
                                            LettersLowerLocal270, LettersLowerDev, LettersLowerDev90,
                                            LettersLowerDev180, LettersLowerDev270, LettersLowerSampler,
                                            LettersLowerDevSampler)
    from datawrappers.letters_upper import (LettersUpperLocal, LettersUpperLocal90, LettersUpperLocal180,
                                            LettersUpperLocal270, LettersUpperDev, LettersUpperDev90,
                                            LettersUpperDev180, LettersUpperDev270, LettersUpperSampler,
                                            LettersUpperDevSampler)
    from utils import generate_probability_partition

    num_clients = 100
    num_samples_min = 100
    num_samples_max = 201
    num_dev_samples = 20000

    lower_sampler = LettersLowerSampler()
    upper_sampler = LettersUpperSampler()
    client_vec = []
    mixture = []
    for k in range(num_clients):
        num_samples = np.random.randint(num_samples_min, num_samples_max)
        frac_vec = generate_probability_partition(8)
        num_samples_vec = [int(frac_vec[s] * num_samples) for s in range(8)]
        num_samples_vec[-1] = num_samples - int(np.sum(num_samples_vec[:-1]))
        lower0_ds = LettersLowerLocal(lower_sampler.sample(range(26), num_samples=num_samples_vec[0]))
        lower90_ds = LettersLowerLocal90(lower_sampler.sample(range(26), num_samples=num_samples_vec[1]))
        lower180_ds = LettersLowerLocal180(lower_sampler.sample(range(26), num_samples=num_samples_vec[2]))
        lower270_ds = LettersLowerLocal270(lower_sampler.sample(range(26), num_samples=num_samples_vec[3]))
        upper0_ds = LettersUpperLocal(upper_sampler.sample(range(26), num_samples=num_samples_vec[4]))
        upper90_ds = LettersUpperLocal90(upper_sampler.sample(range(26), num_samples=num_samples_vec[5]))
        upper180_ds = LettersUpperLocal180(upper_sampler.sample(range(26), num_samples=num_samples_vec[6]))
        upper270_ds = LettersUpperLocal270(upper_sampler.sample(range(26), num_samples=num_samples_vec[7]))
        ds_vec = [lower0_ds, lower90_ds, lower180_ds, lower270_ds, upper0_ds, upper90_ds, upper180_ds, upper270_ds]

        tag = 'Mixture of 8 letters distributions with mixture {}, total {} data points.'.format(
            num_samples_vec, num_samples)
        client = Client(ID=k, ds=ConcatDataset(ds_vec), solver=client_solver, tag=tag)
        client_vec.append(client)
        num_samples_vec.append(num_samples)
        mixture.append(num_samples_vec)

    lower_dev_sampler = LettersLowerDevSampler()
    upper_dev_sampler = LettersUpperDevSampler()
    test_lower0_ds = LettersLowerDev(lower_dev_sampler.sample(range(26), num_dev_samples))
    test_lower90_ds = LettersLowerDev90(lower_dev_sampler.sample(range(26), num_dev_samples))
    test_lower180_ds = LettersLowerDev180(lower_dev_sampler.sample(range(26), num_dev_samples))
    test_lower270_ds = LettersLowerDev270(lower_dev_sampler.sample(range(26), num_dev_samples))
    test_upper0_ds = LettersUpperDev(upper_dev_sampler.sample(range(26), num_dev_samples))
    test_upper90_ds = LettersUpperDev90(upper_dev_sampler.sample(range(26), num_dev_samples))
    test_upper180_ds = LettersUpperDev180(upper_dev_sampler.sample(range(26), num_dev_samples))
    test_upper270_ds = LettersUpperDev270(upper_dev_sampler.sample(range(26), num_dev_samples))

    test_ds_vec = [test_lower0_ds, test_lower90_ds, test_lower180_ds, test_lower270_ds, test_upper0_ds, test_upper90_ds,
                   test_upper180_ds, test_upper270_ds]

    param_dict = {
        'ds_description': '(8) mixture of 8 letters rotattion datasets',
        'num_clusters': 8,
        'num_classes': 26,
        'num_clients': num_clients,
        'num_samples_min': num_samples_min,
        'num_samples_max': num_samples_max,
        'num_samples_test_each': num_dev_samples,
        'client_tags': {},
        'mixture': {}
    }
    for k, client in enumerate(client_vec):
        param_dict['client_tags']['client_' + str(client.ID)] = client.tag
        param_dict['mixture']['client_' + str(client.ID)] = mixture[k]

    return client_vec, test_ds_vec, param_dict


def create_clients_letters_rotation_4set(client_solver):
    from datawrappers.letters_lower import (LettersLowerLocal, LettersLowerLocal90, LettersLowerDev, LettersLowerDev90,
                                            LettersLowerSampler, LettersLowerDevSampler)
    from datawrappers.letters_upper import (LettersUpperLocal, LettersUpperLocal90, LettersUpperDev, LettersUpperDev90,
                                            LettersUpperSampler, LettersUpperDevSampler)
    from utils import generate_probability_partition

    num_clients = 100
    num_samples_min = 100
    num_samples_max = 201
    num_dev_samples = 20000

    lower_sampler = LettersLowerSampler()
    upper_sampler = LettersUpperSampler()
    client_vec = []
    mixture = []
    for k in range(num_clients):
        num_samples = np.random.randint(num_samples_min, num_samples_max)
        frac_vec = generate_probability_partition(4)
        num_samples_vec = [int(frac_vec[s] * num_samples) for s in range(4)]
        num_samples_vec[-1] = num_samples - int(np.sum(num_samples_vec[:-1]))
        lower0_ds = LettersLowerLocal(lower_sampler.sample(range(26), num_samples=num_samples_vec[0]))
        lower90_ds = LettersLowerLocal90(lower_sampler.sample(range(26), num_samples=num_samples_vec[1]))
        upper0_ds = LettersUpperLocal(upper_sampler.sample(range(26), num_samples=num_samples_vec[2]))
        upper90_ds = LettersUpperLocal90(upper_sampler.sample(range(26), num_samples=num_samples_vec[3]))
        ds_vec = [lower0_ds, lower90_ds, upper0_ds, upper90_ds]

        tag = 'Mixture of 4 letters distributions with mixture {}, total {} data points.'.format(
            num_samples_vec, num_samples)
        client = Client(ID=k, ds=ConcatDataset(ds_vec), solver=client_solver, tag=tag)
        client_vec.append(client)
        num_samples_vec.append(num_samples)
        mixture.append(num_samples_vec)

    lower_dev_sampler = LettersLowerDevSampler()
    upper_dev_sampler = LettersUpperDevSampler()
    test_lower0_ds = LettersLowerDev(lower_dev_sampler.sample(range(26), num_dev_samples))
    test_lower90_ds = LettersLowerDev90(lower_dev_sampler.sample(range(26), num_dev_samples))
    test_upper0_ds = LettersUpperDev(upper_dev_sampler.sample(range(26), num_dev_samples))
    test_upper90_ds = LettersUpperDev90(upper_dev_sampler.sample(range(26), num_dev_samples))

    test_ds_vec = [test_lower0_ds, test_lower90_ds, test_upper0_ds, test_upper90_ds]

    param_dict = {
        'ds_description': '(4) mixture of 4 letters rotattion datasets',
        'num_clusters': 4,
        'num_classes': 26,
        'num_clients': num_clients,
        'num_samples_min': num_samples_min,
        'num_samples_max': num_samples_max,
        'num_samples_test_each': num_dev_samples,
        'client_tags': {},
        'mixture': {}
    }
    for k, client in enumerate(client_vec):
        param_dict['client_tags']['client_' + str(client.ID)] = client.tag
        param_dict['mixture']['client_' + str(client.ID)] = mixture[k]

    return client_vec, test_ds_vec, param_dict


def create_clients_lr2(client_solver):
    from datawrappers.lr import LR2ALocal, LR2BLocal, LR2ADev, LR2BDev, LR2Sampler, LR_DIM, LR2_DEV_SIZE

    num_clients = 100
    num_samples_min = 100
    num_samples_max = 201

    client_vec = []
    mixture = []
    sampler = LR2Sampler()
    for i in range(num_clients):
        num_samples = np.random.randint(num_samples_min, num_samples_max)
        num_samples_1 = int(np.random.rand() * num_samples)
        # num_samples_1 = int((.3 if i < (num_clients / 2) else .7) * num_samples)
        # num_samples_1 = int((1.0 * i / num_clients + 0.5 / num_clients) * num_samples)
        num_samples_2 = num_samples - num_samples_1

        lr_ds1 = LR2ALocal(sampler.sample(num_samples_1))
        lr_ds2 = LR2BLocal(sampler.sample(num_samples_2))
        tag = 'Mixture of two set lr2 data. {} samples for dist A, {} samples for dist B, total {} data points.'.format(
            num_samples_1, num_samples_2, num_samples)
        client = Client(ID=i, ds=ConcatDataset([lr_ds1, lr_ds2]), solver=client_solver, tag=tag)
        client_vec.append(client)
        mixture.append([num_samples_1, num_samples_2, num_samples])

    test_lr_ds1 = LR2ADev()
    test_lr_ds2 = LR2BDev()

    param_dict = {
        'ds_description': '(2) Two LR ds',
        'num_clusters': 2,
        'num_clients': num_clients,
        'num_samples_min': num_samples_min,
        'num_samples_max': num_samples_max,
        'num_samples_test_each': LR2_DEV_SIZE,
        'dim': LR_DIM,
        'client_tags': {},
        'mixture': {}
    }
    for k, client in enumerate(client_vec):
        param_dict['client_tags']['client_' + str(client.ID)] = client.tag
        param_dict['mixture']['client_' + str(client.ID)] = mixture[k]

    return client_vec, [test_lr_ds1, test_lr_ds2], param_dict


def create_clients_lr2_div(client_solver, weight_var):
    from datawrappers.lr import LRLocal, LR_DIM
    from utils import generate_probability_partition

    num_clients = 100
    num_samples_min = 100
    num_samples_max = 201
    num_dev_samples = 5000

    mixture_size = 2
    noise_var = 1.
    weight_vec = []
    for _ in range(mixture_size):
        weight_vec.append(np.random.multivariate_normal(np.zeros((LR_DIM,)), weight_var * np.eye(LR_DIM)).tolist())

    client_vec = []
    mixture = []
    for k in range(num_clients):
        num_samples = np.random.randint(num_samples_min, num_samples_max)

        frac_vec = generate_probability_partition(mixture_size)
        num_samples_vec = [int(frac * num_samples) for frac in frac_vec]
        num_samples_vec[-1] = num_samples - int(np.sum(num_samples_vec[:-1]))

        lr_ds_vec = [LRLocal(weight=weight_vec[s], noise_var=noise_var, size=num_samples_vec[s]) for s in
                     range(mixture_size)]

        tag = 'Mixture of {} set lr10 data with respectively {} samples, total {} samples.'.format(
            mixture_size, num_samples_vec, num_samples)
        client = Client(ID=k, ds=ConcatDataset(lr_ds_vec), solver=client_solver, tag=tag)
        client_vec.append(client)
        num_samples_vec.append(num_samples)
        mixture.append(num_samples_vec)

    text_ds_vec = []
    for s in range(mixture_size):
        text_ds_vec.append(LRLocal(weight=weight_vec[s], noise_var=noise_var, size=num_dev_samples))

    param_dict = {
        'ds_description': '({}) LR ds with input weight_var'.format(mixture_size),
        'num_clusters': mixture_size,
        'num_clients': num_clients,
        'num_samples_min': num_samples_min,
        'num_samples_max': num_samples_max,
        'num_samples_test_each': num_dev_samples,
        'dim': LR_DIM,
        'weight_vec': weight_vec,
        'weight_var': weight_var,
        'noise_var': noise_var,
        'client_tags': {},
        'mixture': {}
    }
    for k, client in enumerate(client_vec):
        param_dict['client_tags']['client_' + str(client.ID)] = client.tag
        param_dict['mixture']['client_' + str(client.ID)] = mixture[k]

    return client_vec, text_ds_vec, param_dict

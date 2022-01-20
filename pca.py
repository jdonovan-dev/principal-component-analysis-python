from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt


def load_and_center_dataset(filename):
    x = np.load(filename)
    x = x - np.mean(x, axis=0)
    return x


def get_covariance(dataset):
    return np.dot(np.transpose(dataset), dataset) / (dataset.shape[0] - 1)


def get_eig(S, m):
    x = eigh(S, eigvals=(S.shape[0] - m, S.shape[0] - 1))
    return np.diag(x[0][::-1]), np.fliplr(x[1])


def get_eig_perc(S, perc):
    x = eigh(S)
    comp = x[0].sum() * perc
    temp = []
    for v in x[0]:
        if v > comp:
            temp.append(v)

    return np.diag(temp[::-1]), np.fliplr(np.delete(x[1], np.s_[0:len(x[0]) - len(temp)], axis=1))


def project_image(img, U):
    return np.dot(U, np.dot(np.transpose(U), img))


def display_image(orig, proj):
    img, dims = plt.subplots(nrows=1, ncols=2)

    dims[0].set_title('Original')
    v1 = dims[0].imshow(np.transpose(
        np.reshape(orig, (32, 32))), aspect='equal')
    img.colorbar(v1, ax=dims[0])

    dims[1].set_title('Projection')
    v2 = dims[1].imshow(np.transpose(
        np.reshape(proj, (32, 32))), aspect='equal')
    img.colorbar(v2, ax=dims[1])

    plt.show()
    return None

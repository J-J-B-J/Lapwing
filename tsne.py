import argparse

import numpy
from tqdm import tqdm
import cv2
import torch
import random
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from datasets import Dataset, collate_skip_empty
from resnet import ResNet101


def fix_random_seeds():
    """Make the random seed very random"""
    seed = 10
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_features(dataset, batch, num_images, data_type: str):
    """Get the features of a set of images"""
    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # initialize our implementation of ResNet
    model = ResNet101()
    model.eval()
    model.to(device)

    # read the dataset and initialize the data loader
    dataset = Dataset(dataset, data_type, num_images)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch,
                                             collate_fn=collate_skip_empty,
                                             shuffle=True)

    # we'll store the features as NumPy array of size num_images x feature_size
    features = None

    # we'll also store the image labels and paths to visualize them later
    labels = []
    image_paths = []

    for batch in tqdm(dataloader, desc='Running the model inference'):
        images = batch['image'].to(device)
        labels += batch['label']
        image_paths += batch['image_path']

        with torch.no_grad():
            output = model.forward(images)

        current_features = output.cpu().numpy()
        if features is not None:
            features = np.concatenate((features, current_features))
        else:
            features = current_features

    return features, labels, image_paths


# scale and move the coordinates, so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range


def scale_image(image, max_image_size):
    image_height, image_width, _ = image.shape

    scale = max(1, image_width / max_image_size, image_height / max_image_size)
    image_width = int(image_width / scale)
    image_height = int(image_height / scale)

    image = cv2.resize(image, (image_width, image_height))
    return image


def draw_rectangle_by_class(image, label):
    image_height, image_width, _ = image.shape

    # get the color corresponding to image class
    image = cv2.rectangle(image, (0, 0), (image_width - 1, image_height - 1),
                          color=[255, 255, 255], thickness=1)

    return image


def compute_plot_coordinates(image, x, y, image_centers_area_size, offset):
    image_height, image_width, _ = image.shape

    # compute the image center coordinates on the plot
    center_x = int(image_centers_area_size * x) + offset

    # in matplotlib, the y axis is directed upward
    # to have the same here, we need to mirror the y coordinate
    center_y = int(image_centers_area_size * (1 - y)) + offset

    # knowing the image center, compute the coordinates of the top left and
    # bottom right corner
    tl_x = center_x - int(image_width / 2)
    tl_y = center_y - int(image_height / 2)

    br_x = tl_x + image_width
    br_y = tl_y + image_height

    return tl_x, tl_y, br_x, br_y


def visualize_tsne_images(tx, ty, images, labels, data_type: str, plot_size=1000, max_image_size=1000):
    # we'll put the image centers in the central area of the plot
    # and use offsets to make sure the images fit the plot
    offset = max_image_size // 2
    image_centers_area_size = plot_size - 2 * offset

    tsne_plot = 255 * np.ones((plot_size, plot_size, 3), np.uint8)

    # now we'll put a small copy of every image to its corresponding T-SNE coordinate
    for image_path, label, x, y in tqdm(
            zip(images, labels, tx, ty),
            desc='Building the T-SNE plot',
            total=len(images)
    ):
        image = cv2.imread(image_path)

        # scale the image to put it to the plot
        image = scale_image(image, max_image_size)

        # draw a rectangle with a color corresponding to the image class
        image = draw_rectangle_by_class(image, label)

        # compute the coordinates of the image on the scaled plot visualization
        tl_x, tl_y, br_x, br_y = compute_plot_coordinates(image, x, y, image_centers_area_size, offset)

        # put the image to its TSNE coordinates using numpy subarray indices
        tsne_plot[tl_y:br_y, tl_x:br_x, :] = image

    plt.title(data_type)

    plt.imshow(tsne_plot[:, :, ::-1])
    plt.show()


def visualize_tsne(tsne, images, labels, data_type: str, plot_size=1000, max_image_size=100):
    # extract x and y coordinates representing the positions of the images
    # on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]

    # scale and move the coordinates, so they fit [0; 1] ranges
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    # visualize the plot: samples as images
    visualize_tsne_images(tx, ty, images, labels, data_type, plot_size=plot_size,
                          max_image_size=max_image_size)


def image_visualisation(dir_path: str, data_type: str, dpi: int, figsize: int):
    """Visualise images on a similarity diagram"""
    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams['figure.figsize'] = figsize

    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type=str, default=dir_path)
    parser.add_argument('--batch', type=int, default=126)
    parser.add_argument('--num_images', type=int, default=500)
    args = parser.parse_args()

    fix_random_seeds()

    features, labels, image_paths = get_features(
        dataset=args.path,
        data_type=data_type,
        batch=args.batch,
        num_images=args.num_images
    )

    if len(image_paths) <= 30:
        tsne = TSNE(n_components=2, perplexity=len(image_paths)-1).fit_transform(features)
    else:
        tsne = TSNE(n_components=2).fit_transform(features)

    visualize_tsne(tsne, image_paths, labels, data_type)

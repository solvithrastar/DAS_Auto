"""
A collection of image processing functions which can be used
to process DAS images.
The tuning parameters of the functions will vary between DAS setups.
"""

import numpy as np
from typing import Tuple, Union, List
import cv2, pathlib
import matplotlib as mpl

from skimage import io
from skimage.filters import threshold_multiotsu, threshold_otsu
from skimage.morphology import remove_small_objects
from skimage.measure import label, regionprops
from skimage.color import rgb2gray

from scipy.ndimage.filters import gaussian_filter


def crop_image(
    image: np.ndarray, x_range: Tuple[int, int], y_range: Tuple[int, int]
) -> np.ndarray:
    """
    If the image has been plotted with matplotlib, it will contain some frame
    and maybe even axes and a colorbar. This function is used to cut those
    out of the image.

    :param image: The numpy array containing the values in the image
    :type image: np.ndarray
    :param x_range: A tuple saying from which x pixel to which x pixel you want
        to keep the image
    :type x_range: Tuple[int, int]
    :param y_range: A tuple saying from which y pixel to which y pixel you want
        to keep the image
    :type y_range: Tuple[int, int]
    :return: A numpy array which contains the cropped image
    :rtype: np.ndarray
    """
    if len(image.shape) == 2:
        return image[y_range[0] : y_range[1], x_range[0] : x_range[1]]
    else:
        return image[y_range[0] : y_range[1], x_range[0] : x_range[1], :]


def read_image_from_file(
    image: Union[str, pathlib.Path], as_gray: bool = True
) -> np.ndarray:
    """
    When the image has been saved in a file, this will convert it to a numpy
    array. If you want to use the other algorithms, you make it gray

    :param image: Path to the image file
    :type image: Union[str, pathlib.Path]
    :param as_gray: Convert it to grayscale, defaults to True
    :type as_gray: bool, optional
    :return: Numpy array including the image information
    :rtype: np.ndarray
    """
    if as_gray:
        return 1.0 - io.imread(fname=image, as_gray=as_gray)
    else:
        return io.imread(fname=image)


def filter_image(
    image: np.ndarray, sigma: float = 1.5, mode: str = "wrap"
) -> np.ndarray:
    """
    Filtering of the image, this can be customized but it currently
    applies a Gaussian filter to it

    :param image: A numpy array containing an image
    :type image: np.ndarray
    :param sigma: The standard deviation of the filtering, defaults to 1.5
    :type sigma: float, optional
    :param mode: How the edges are handled, defaults to "wrap"
    :type mode: str, optional
    :return: A filtered image
    :rtype: np.ndarray
    """
    return gaussian_filter(input=image, sigma=sigma, mode=mode)


def data_2_grayscale(
    image: np.ndarray, clip: Union[float, int] = 400
) -> np.ndarray:
    """
    This function takes data, plots in on the seismic colorscale within
    matplotlib and converts it into an inverted grayscale image.
    This works well in pipelines that performs detections directly on the
    data without the plotting.

    :param image: A numpy array containing data
    :type image: np.ndarray
    :param clip: A value where data is clipped, used to normalize the colorbar,
        the data will be clipped to the same range. defaults to 400
    :type clip: Union[float, int], optional
    :return: A grayscale image
    :rtype: np.ndarray
    """
    # defining the correct colorscale
    norm = mpl.colors.Normalize(vmin=-clip, vmax=clip, clip=True)
    mapper = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.seismic)
    # make sure the data fits the colorscale
    image = np.clip(image, -clip, clip)

    # map data to seismic colorscale
    image = mapper.to_rgba(image)
    # map seismic colorscale into grayscale
    image = (
        0.2125 * image[:, :, 0]
        + 0.7154 * image[:, :, 1]
        + 0.0721 * image[:, :, 2]
    )
    # invert the grayscale
    return 1.0 - image


def remove_median_brightness(image: np.ndarray) -> np.ndarray:
    """
    To reduce the noise level in the image, the brightness can be reduced
    row by row as the noise levels are row dependent. This removes the
    median brightness level per row.

    :param image: The numpy array containing the image
    :type image: np.ndarray
    :return: An array where the brightness level has been reduced
    :rtype: np.ndarray
    """
    medians = np.median(image, axis=1).reshape(image.shape[0], 1)
    return np.clip(image - medians, 0.0, 1.0)


def remove_mean_brightness(image: np.ndarray) -> np.ndarray:
    """
    To reduce the noise level in the image, the brightness can be reduced
    row by row as the noise levels are row dependent. This removes the
    mean brightness level per row.

    :param image: The numpy array containing the image
    :type image: np.ndarray
    :return: An array where the brightness level has been reduced
    :rtype: np.ndarray
    """
    means = np.mean(image, axis=1).reshape(image.shape[0], 1)
    return np.clip(image - means, 0.0, 1.0)


def compute_brightness_thresholds(image: np.ndarray, classes=4) -> list:
    """
    Compute the threshold values which are used to separate the image into
    True/False pixels. The function will always return a list, even if the
    classes are only two, then it will be a list of length 1.
    It can be useful to test a few different numbers of classes to see
    at which point the relevant signal is True and when it is False.

    :param image: The numpy array containing the image
    :type image: np.ndarray
    :param classes: How many classes to separate the image to, needs to be
        2 or higher. High numbers take longer to compute, defaults to 4
    :type classes: int, optional
    :return: A list of brightness values used to separate the image to classes
    :rtype: list
    """
    if classes < 2:
        raise ValueError("Classes can not be fewer than two")
    elif classes == 2:
        return list(threshold_otsu(image=image))
    else:
        return threshold_multiotsu(image=image, classes=classes)


def apply_brightness_threshold(
    image: np.ndarray, threshold: float
) -> np.ndarray:
    """
    This function takes a grayscale image and a brightness threshold value
    and separates the image into a binary True/False image.

    :param image: A numpy array containing the image
    :type image: np.ndarray
    :param threshold: The brightness threshold value
    :type threshold: float
    :return: A binary image represented in a numpy array
    :rtype: np.ndarray
    """
    return np.where(image < threshold, False, True)


def coherency_thresholding(
    image: np.ndarray, min_cluster_size: int = 64
) -> np.ndarray:
    """
    Put any pixel to False which is not a part of a True cluster over a
    certain limit

    :param image: A numpy array containing the image
    :type image: np.ndarray
    :param min_cluster_size: The minimum True cluster size in pixels,
        defaults to 64
    :type min_cluster_size: int, optional
    :return: A binary image with incoherent Trues as False
    :rtype: np.ndarray
    """
    return remove_small_objects(ar=image, min_size=min_cluster_size)


def create_shaped_template(
    vertical_length: int, horizontal_length: int
) -> np.ndarray:
    """
    Created a template shape to move through the array.
    In principle this can be any shape, currently we only give direct support
    to rectangular templates, but OpenCV provides other options so the
    function can be customized to fit anyones needs.

    :param vertical_length: vertical length of template in pixels
    :type vertical_length: int
    :param horizontal_length: horizontal length of template in pixels
    :type horizontal_length: int
    :return: An array describing the template
    :rtype: np.ndarray
    """
    template_structure = cv2.getStructuringElement(
        cv2.MORPH_RECT, (horizontal_length, vertical_length)
    )
    return template_structure


def remove_template_shape(
    image: np.ndarray, template: Union[List[np.ndarray], np.ndarray]
) -> np.ndarray:
    """
    A template moves through the images and detects fitting True shapes and
    makes them False.

    :param image: A numpy array containing the image, the array needs to be
        imported as type np.float32
    :type image: np.ndarray
    :param template: Either a single template or a list of templates
    :type template: Union[list[np.ndarray], np.ndarray]
    :return: A boolean image where the templates have been removed
    :rtype: np.ndarray
    """
    shapes = np.copy(image)
    if isinstance(template, list):
        for _i, temp in enumerate(template):
            if _i > 0:
                shapes = np.copy(image)
            shapes = cv2.erode(shapes, temp)
            shapes = cv2.dilate(shapes, temp)
            image = image - shapes
    else:
        shapes = cv2.erode(shapes, template)
        shapes = cv2.dilate(shapes, template)
        image = image - shapes

    return image.astype(bool)


def get_regionprops(
    image: np.ndarray,
) -> List:
    """
    Count positive regions in a binary image and get properties of those
    regions.

    :param image: A boolean numpy array containing the image
    :type image: np.ndarray
    :return: A list of region properties
    :rtype: List
    """
    labels = label(input=image)
    return regionprops(labels)

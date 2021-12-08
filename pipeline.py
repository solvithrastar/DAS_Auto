"""
A file where a pipeline can be assembled,

The pipeline can for stack up multiple of the image processing
methods and run them on a stack of either images or datafiles.

A data reading function will not be provided here as they depend on
DAS interrogators and can include proprietary information.
"""
import image_processing as ip
import pathlib
from typing import Tuple, Union, List
import numpy as np


def earthquake_in_image(
    filename: Union[str, pathlib.Path],
    x_range: Tuple[int, int],
    y_range: Tuple[int, int],
    median: bool,
    mean: bool,
    classes: int,
    threshold: int,
    min_cluster_size: Union[int, List[int]],
    templates: Union[Tuple[int, int], List[Tuple[int, int]]] = None,
) -> bool:
    """
    A pipeline which indicates whether there is any signal of interest
    within an image.

    :param filename: Name of the .png or .jpg file with the image
    :type filename: Union[str, pathlib.Path]
    :param x_range: Pixels from/to which to crop the image horizontally
    :type x_range: Tuple[int, int]
    :param y_range: Pixels from/to which to crop the image vertically
    :type y_range: Tuple[int, int]
    :param median: Boolean value deciding whether to remove median noise
    :type median: bool
    :param mean: Boolean value deciding whether to remove mean noise
    :type mean: bool
    :param classes: How many classes to separate the brightness into
    :type classes: int
    :param threshold: Which of those classes to pick as the True/False
        boundary. This refers to the index of the threshold list.
        If there are only 2 classes, give 0 here.
    :type threshold: int
    :param min_cluster_size: Minimum coherency threshold as pixels
    :type min_cluster_size: Union[int, List[int]]
    :param templates: Shapes of templates to remove from image, if
        you do not want to remove any shapes, pass None, defaults to None
    :type templates: Union[Tuple[int, int], List[Tuple[int, int]]], optional
    :return: True/False telling whether there is an earthquake in
        the image
    :rtype: bool
    """
    image = ip.read_image_from_file(image=filename, as_gray=True)
    image = ip.crop_image(image=image, x_range=x_range, y_range=y_range)
    if median:
        image = ip.remove_median_brightness(image)
    if mean:
        image = ip.remove_mean_brightness(image)
    brightness_thresholds = ip.compute_brightness_thresholds(
        image=image, classes=classes
    )
    image = ip.apply_brightness_threshold(
        image=image, threshold=brightness_thresholds[threshold]
    )
    if isinstance(min_cluster_size, list):
        first_min_cluster = min_cluster_size[0]
        second_min_cluster = min_cluster_size[1]
    else:
        first_min_cluster = min_cluster_size
        second_min_cluster = min_cluster_size

    image = ip.coherency_thresholding(
        image=image, min_cluster_size=first_min_cluster
    )
    template_shapes = []
    if templates is not None:
        if not isinstance(templates, list):
            templates = [templates]
        for template in templates:
            template_shapes.append(
                ip.create_shaped_template(template[0], template[1])
            )
        image = ip.remove_template_shape(
            image=image.astype(np.float32), template=template_shapes
        )
    if templates is None and second_min_cluster == first_min_cluster:
        pass
    else:
        image = ip.coherency_thresholding(
            image=image, min_cluster_size=second_min_cluster
        )
    props = ip.get_regionprops(image=image)
    if len(props) > 0:
        return True
    else:
        return False


def earthquake_in_data(
    data: np.ndarray,
    clip: Union[int, float],
    median: bool,
    mean: bool,
    classes: int,
    threshold: int,
    min_cluster_size: Union[int, List[int]],
    sigma: float = None,
    templates: Union[Tuple[int, int], List[Tuple[int, int]]] = None,
) -> bool:
    """
    A pipeline which indicates whether there is an earthquake in
    the passed data chunk.
    The function assumes the data from the interrogator has been read,
    converted to nanostrain, potentially filtered, and is passed as
    a numpy array

    :param data: Numpy array containing the data
    :type data: np.ndarray
    :param clip: Values at which to clip the data and normalize colorbar
    :type clip: Union[int, float]
    :param median: Boolean value deciding whether to remove median noise
    :type median: bool
    :param mean: Boolean value deciding whether to remove mean noise
    :type mean: bool
    :param classes: How many classes to separate the brightness into
    :type classes: int
    :param threshold: Which of those classes to pick as the True/False
        boundary. This refers to the index of the threshold list.
        If there are only 2 classes, give 0 here.
    :type threshold: int
    :param min_cluster_size: Minimum coherency threshold as pixels
    :type min_cluster_size: Union[int, List[int]]
    :param sigma: Standard deviation of Gaussian filter,
        if None, image will not be filtered, defaults to None
    :type sigma: float, optional
    :param templates: Shapes of templates to remove from image, if
        you do not want to remove any shapes, pass None, defaults to None
    :type templates: Union[Tuple[int, int], List[Tuple[int, int]]], optional
    :return: True/False telling whether there is an earthquake in
        the data chunk
    :rtype: bool
    """
    data = np.clip(data, a_min=-clip, a_max=clip)

    if sigma is not None:
        data = ip.filter_image(image=data, sigma=sigma)
    data = ip.data_2_grayscale(image=data, clip=clip)

    if median:
        data = ip.remove_median_brightness(image=data)
    if mean:
        data = ip.remove_mean_brightness(image=data)

    brightness_thresholds = ip.compute_brightness_thresholds(
        image=data, classes=classes
    )
    data = ip.apply_brightness_threshold(
        image=data, threshold=brightness_thresholds[threshold]
    )

    if isinstance(min_cluster_size, list):
        first_min_cluster = min_cluster_size[0]
        second_min_cluster = min_cluster_size[1]
    else:
        first_min_cluster = min_cluster_size
        second_min_cluster = min_cluster_size

    data = ip.coherency_thresholding(
        image=data, min_cluster_size=first_min_cluster
    )
    template_shapes = []
    if templates is not None:
        if not isinstance(templates, list):
            templates = [templates]
        for template in templates:
            template_shapes.append(
                ip.create_shaped_template(template[0], template[1])
            )
        data = ip.remove_template_shape(
            image=data.astype(np.float32), template=template_shapes
        )
    if templates is None and second_min_cluster == first_min_cluster:
        pass
    else:
        data = ip.coherency_thresholding(
            image=data, min_cluster_size=second_min_cluster
        )
    props = ip.get_regionprops(image=data)
    if len(props) > 0:
        return True
    else:
        return False

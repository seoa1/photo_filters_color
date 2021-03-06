#!/usr/bin/env python3

import math

from PIL import Image

#lab 0 functions
def get_pixel(image, x, y):
    if x < 0:
        return get_pixel(image, 0 , y)
    elif x >= image['height']:
        return get_pixel(image, image['height'] - 1, y)
    elif y < 0:
        return get_pixel(image, x, 0)
    elif y >= image['width']:
        return get_pixel(image, x , image['width'] - 1)
    return image['pixels'][x * image['width'] + y]


def set_pixel(image, x, y, c):
    idx = x * image['width'] + y
    if idx < len(image['pixels']):
        image['pixels'][idx] = c
    else:
        image['pixels'].append(c)


def apply_per_pixel(image, func):
    result = {
        'height': image['height'],
        'width': image['width'],
        'pixels': []
    }
    for x in range(image['height']):
        for y in range(image['width']):
            color = get_pixel(image, x, y)
            newcolor = func(color)
            set_pixel(result, x, y, newcolor)
    return result


def inverted(image):
    return apply_per_pixel(image, lambda c: 255-c)


# HELPER FUNCTIONS

def correlate(image, kernel):
    """
    Compute the result of correlating the given image with the given kernel.

    The output of this function should have the same form as a 6.009 image (a
    dictionary with 'height', 'width', and 'pixels' keys), but its pixel values
    do not necessarily need to be in the range [0,255], nor do they need to be
    integers (they should not be clipped or rounded at all).

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.

    DESCRIBE YOUR KERNEL REPRESENTATION HERE
    My kernels will be 2D arrays, to account for the possibility of a kernel 
    that is not a perfect square.
    """
    new_image = {'height': image['height'], 'width': image['width'], 'pixels': []}
    kernel_col_len = len(kernel)
    kernel_row_len = len(kernel[0])
    image_len = len(image['pixels'])
    for i in range(image_len):
        row = i // image['width']
        col = i % image['width']
        color_sum = 0
        for j in range(kernel_col_len):
            for k in range(kernel_row_len):
                color_sum += kernel[j][k] * get_pixel(image, row + (j - kernel_col_len // 2), col + (k - kernel_row_len // 2))
        new_image['pixels'].append(color_sum)
    return new_image


def round_and_clip_image(image):
    """
    Given a dictionary, ensure that the values in the 'pixels' list are all
    integers in the range [0, 255].

    All values should be converted to integers using Python's `round` function.

    Any locations with values higher than 255 in the input should have value
    255 in the output; and any locations with values lower than 0 in the input
    should have value 0 in the output.
    """
    for i in range(len(image['pixels'])):
        num = image['pixels'][i]
        if num > 255:
            image['pixels'][i] = 255
        elif num < 0:
            image['pixels'][i] = 0
        else:
            image['pixels'][i] = round(num)
    return image

# FILTERS

def blurred(image, n):
    """
    Return a new image representing the result of applying a box blur (with
    kernel size n) to the given input image.

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.
    """
    # first, create a representation for the appropriate n-by-n kernel (you may
    # wish to define another helper function for this)
    kernel = blurred_kernel(n)

    # then compute the correlation of the input image with that kernel
    new_image = correlate(image, kernel)

    # and, finally, make sure that the output is a valid image (using the
    # helper function from above) before returning it.
    tested_image = round_and_clip_image(new_image)
    return tested_image

def blurred_kernel(n):
    unit_value = 1 / (n * n)
    return [ ([unit_value] * n) for i in range(n)]

def sharpened(image, n):
    blurred_image = correlate(image, blurred_kernel(n))
    sharpened_image = {'height': image['height'], 'width': image['width'], 'pixels': []}
    for i in range(len(blurred_image['pixels'])):
        sharpened_image['pixels'].append(2 * image['pixels'][i] - blurred_image['pixels'][i])
    return round_and_clip_image(sharpened_image)

def edges(image):
    kernel_x = [[-1,0,1],
                [-2,0,2],
                [-1,0,1]]
    kernel_y = [[-1,-2,-1],
                [0,0,0],
                [1,2,1]]
    correlated_x = correlate(image, kernel_x)
    correlated_y = correlate(image, kernel_y)
    edge_image = {'height': image['height'], 'width': image['width'], 'pixels': []}
    for i in range(len(image['pixels'])):
        edge_image['pixels'].append(round(math.sqrt(correlated_x['pixels'][i] ** 2 + correlated_y['pixels'][i] ** 2)))
    return round_and_clip_image(edge_image)
# HELPER FUNCTIONS FOR LOADING AND SAVING IMAGES

def load_greyscale_image(filename):
    """
    Loads an image from the given file and returns a dictionary
    representing that image.  This also performs conversion to greyscale.

    Invoked as, for example:
       i = load_image('test_images/cat.png')
    """
    with open(filename, 'rb') as img_handle:
        img = Image.open(img_handle)
        img_data = img.getdata()
        if img.mode.startswith('RGB'):
            pixels = [round(.299 * p[0] + .587 * p[1] + .114 * p[2])
                      for p in img_data]
        elif img.mode == 'LA':
            pixels = [p[0] for p in img_data]
        elif img.mode == 'L':
            pixels = list(img_data)
        else:
            raise ValueError('Unsupported image mode: %r' % img.mode)
        w, h = img.size
        return {'height': h, 'width': w, 'pixels': pixels}


def save_greyscale_image(image, filename, mode='PNG'):
    """
    Saves the given image to disk or to a file-like object.  If filename is
    given as a string, the file type will be inferred from the given name.  If
    filename is given as a file-like object, the file type will be determined
    by the 'mode' parameter.
    """
    out = Image.new(mode='L', size=(image['width'], image['height']))
    out.putdata(image['pixels'])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()

# VARIOUS FILTERS


def color_filter_from_greyscale_filter(filt):
    """
    Given a filter that takes a greyscale image as input and produces a
    greyscale image as output, returns a function that takes a color image as
    input and produces the filtered color image.
    """
    def color_function(color_image):
        red_greyscale_img = {'height': color_image['height'], 'width': color_image['width'], 'pixels': []}
        green_greyscale_img = {'height': color_image['height'], 'width': color_image['width'], 'pixels': []}
        blue_greyscale_img = {'height': color_image['height'], 'width': color_image['width'], 'pixels': []}
        color_filtered_img = {'height': color_image['height'], 'width': color_image['width'], 'pixels': []}
        for pixel in color_image['pixels']:
            red_greyscale_img['pixels'].append(pixel[0])
            green_greyscale_img['pixels'].append(pixel[1])
            blue_greyscale_img['pixels'].append(pixel[2])
        red_filtered_img = filt(red_greyscale_img)
        green_filtered_img = filt(green_greyscale_img)
        blue_filtered_img = filt(blue_greyscale_img)
        for i in range(len(color_image['pixels'])):
            color_filtered_img['pixels'].append((red_filtered_img['pixels'][i], green_filtered_img['pixels'][i], blue_filtered_img['pixels'][i]))
        return color_filtered_img
    return color_function


def make_blur_filter(n):
    def blur_filter(image):
        return blurred(image, n)
    return blur_filter

def make_sharpen_filter(n):
    def sharpen_filter(image):
        return sharpened(image, n)
    return sharpen_filter

def filter_cascade(filters):
    """
    Given a list of filters (implemented as functions on images), returns a new
    single filter such that applying that filter to an image produces the same
    output as applying each of the individual ones in turn.
    """
    def cascaded_filter(image):
        for filt in filters:
            image = filt(image)
        return image
    return cascaded_filter

# SEAM CARVING

# Main Seam Carving Implementation

def seam_carving(image, ncols):
    """
    Starting from the given image, use the seam carving technique to remove
    ncols (an integer) columns from the image.
    """
    grey_image = greyscale_image_from_color_image(image)
    energy_img = compute_energy(grey_image)
    cum_energy = cumulative_energy_map(energy_img)
    new_image = image.copy()
    for i in range(ncols):
        seam = minimum_energy_seam(cum_energy)
        new_image = image_without_seam(new_image, seam)
        cum_energy = image_without_seam(cum_energy, seam)
    return new_image

# Optional Helper Functions for Seam Carving

def greyscale_image_from_color_image(image):
    """
    Given a color image, computes and returns a corresponding greyscale image.

    Returns a greyscale image (represented as a dictionary).
    """
    greyscale_image = {'height': image['height'], 'width': image['width'], 'pixels': []}
    for pixel in image['pixels']:
        greyscale_image['pixels'].append(round(.299 * pixel[0] + .587 * pixel[1] + .114 * pixel[2]))
    return greyscale_image
        


def compute_energy(grey):
    """
    Given a greyscale image, computes a measure of "energy", in our case using
    the edges function from last week.

    Returns a greyscale image (represented as a dictionary).
    """
    return edges(grey)


def cumulative_energy_map(energy):
    """
    Given a measure of energy (e.g., the output of the compute_energy function),
    computes a "cumulative energy map" as described in the lab 1 writeup.

    Returns a dictionary with 'height', 'width', and 'pixels' keys (but where
    the values in the 'pixels' array may not necessarily be in the range [0,
    255].
    """
    cum_energy = {'height': 1, 'width': energy['width'], 'pixels': energy['pixels'][0 : energy['width']]}
    row_num = 1
    while cum_energy['height'] < energy['height']:
        for i in range(cum_energy['width']):
            minim = 0
            if row_num > 0:
                if i > 0 and i < cum_energy['width'] - 1:
                    minim = min(get_pixel(cum_energy, row_num - 1, j) for j in range(i - 1, i + 2))
                elif i == 0:
                    minim = min(get_pixel(cum_energy, row_num - 1, j) for j in range(0, 2))
                else:
                    minim = min(get_pixel(cum_energy, row_num - 1, j) for j in range(cum_energy['width'] - 2, cum_energy['width']))
            cum_energy['pixels'].append(get_pixel(energy, row_num, i) + minim)
        cum_energy['height'] += 1
        row_num += 1
    return cum_energy

def minimum_energy_seam(c):
    """
    Given a cumulative energy map, returns a list of the indices into the
    'pixels' list that correspond to pixels contained in the minimum-energy
    seam (computed as described in the lab 1 writeup).
    """
    seam = []
    first_row_idx = len(c['pixels']) - c['width']
    min_energy = c['pixels'][first_row_idx]
    start_idx = first_row_idx
    for i in range(first_row_idx, len(c['pixels'])):
        if c['pixels'][i] < min_energy:
            min_energy = c['pixels'][i]
            start_idx = i
    seam.append(start_idx)
    for row_num in reversed(range(c['height'])):
        if row_num > 0:
            top_left = c['pixels'][start_idx - c['width'] - 1]
            top_middle = c['pixels'][start_idx - c['width']]
            top_right = c['pixels'][start_idx - c['width'] + 1]
            if start_idx > row_num * c['width'] - 1:
                if start_idx < (row_num + 1) * c['width']:
                    if top_left <= top_middle and top_left <= top_right:
                        start_idx = start_idx - c['width'] - 1
                    elif top_middle <= top_left and top_middle <= top_right:
                        start_idx = start_idx- c['width']
                    else:
                        start_idx = start_idx - c['width'] + 1
                else:
                    if top_left <= top_middle:
                        start_idx = start_idx - c['width'] - 1
                    else:
                        start_idx = start_idx - c['width']
            else:
                if top_middle <= top_right:
                    start_idx = start_idx - c['width']
                else:
                    start_idx = start_idx - c['width'] + 1
            seam.append(start_idx)
    return seam

def image_without_seam(im, s):
    """
    Given a (color) image and a list of indices to be removed from the image,
    return a new image (without modifying the original) that contains all the
    pixels from the original image except those corresponding to the locations
    in the given list.
    """
    new_image = {'height': im['height'], 'width': im['width'] - 1, 'pixels': []}
    for i in range(len(im['pixels'])):
        if i not in s:
            new_image['pixels'].append(im['pixels'][i])
    return new_image


# HELPER FUNCTIONS FOR LOADING AND SAVING COLOR IMAGES

def load_color_image(filename):
    """
    Loads a color image from the given file and returns a dictionary
    representing that image.

    Invoked as, for example:
       i = load_color_image('test_images/cat.png')
    """
    with open(filename, 'rb') as img_handle:
        img = Image.open(img_handle)
        img = img.convert('RGB')  # in case we were given a greyscale image
        img_data = img.getdata()
        pixels = list(img_data)
        w, h = img.size
        return {'height': h, 'width': w, 'pixels': pixels}


def save_color_image(image, filename, mode='PNG'):
    """
    Saves the given color image to disk or to a file-like object.  If filename
    is given as a string, the file type will be inferred from the given name.
    If filename is given as a file-like object, the file type will be
    determined by the 'mode' parameter.
    """
    out = Image.new(mode='RGB', size=(image['width'], image['height']))
    out.putdata(image['pixels'])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()


if __name__ == '__main__':
    # code in this block will only be run when you explicitly run your script,
    # and not when the tests are being run.  this is a good place for
    # generating images, etc.
    image = load_color_image('test_images/cat.png')
    new_image = seam_carving(image, 5)
    save_color_image(new_image, 'seamed_cat.png')

import cv2
from cv2 import circle
import numpy as np
import imutils
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from math import floor, degrees, atan, cos, sin, radians
import logging
mpl.use('Agg')
logger = logging.getLogger(__name__)


from .calc_angle_spacing import calc_angle_spacing
from Smartscope.lib.image_manipulations import convert_centers_to_boxes, save_image, to_8bits, auto_contrast



def find_contours(im, thresh):
    thresh = cv2.threshold(im, thresh, 255, cv2.THRESH_BINARY)[1]
    t = cv2.convertScaleAbs(thresh)
    cnts = cv2.findContours(t.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    return cnts, t


def gauss(x, mu, sigma, A):
    return A * np.exp(- ((x - mu) ** 2) / (2 * sigma ** 2))


def fit_gauss(blur, min=40, max=255):
    bins = max - min
    flat_blur = blur.flatten()
    flat_blur = flat_blur[(flat_blur > min) & (flat_blur < max)]
    y, x, _ = plt.hist(flat_blur, bins=bins)
    peak = np.argmax(y)
    amax = np.max(y)
    std = [-1, -1]
    for i in range(peak, int(bins), 1):
        if y[i] <= amax * 0.25:
            std[1] = i - peak
            break
    for i in range(peak, 0, -1):
        if y[i] <= amax * 0.25:
            std[0] = peak - i
            break

    std = np.mean(np.array([abs(i) for i in std if i >= 0]))

    expected = (x[peak], std, amax)
    try:
        params, cov = curve_fit(gauss, x[:-1], y, expected)
        return params, True
    except Exception:
        print('Could not fit gaussian, passing expected params')
        return expected, False


def plot_hist_gauss(image, thresh, mu=None, sigma=None, a=None, size=254):
    mydpi = 300
    fig = plt.figure(figsize=(5, 5), dpi=mydpi)
    ax = fig.add_subplot(111)
    flat = image.flatten()
    ax.hist(flat, bins=200, label='Distribution')
    x = np.linspace(0, 255, 100)
    if all([mu is not None, sigma is not None, a is not None]):
        ax.plot(x, gauss(x, mu, sigma, a), color='red', lw=3, label='gaussian')

    if mu is None:
        mu = np.mean(flat)
    ax.axvline(mu, c='orange', label='mean')
    ax.axvline(thresh, c='green', label='threshold')
    ax.title.set_text('Histogram')
    ax.set_xlabel('Pixel intensity')
    ax.set_ylabel('Counts')
    ax.legend()
    fig.canvas.draw()
    hist = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    hist = hist.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    hist = imutils.resize(hist, height=size)
    plt.close(fig='all')
    return hist


def find_squares(montage, threshold=30):
    if 'threshold' in montage.__dict__:
        threshold = montage.threshold

    blurred = cv2.GaussianBlur(montage.montage, (5, 5), 0)

    cnts, _ = find_contours(blurred, threshold)
    cnts = [cnt for cnt in cnts if (montage.area_threshold[0] * 0.5 < cv2.contourArea(cnt))]

    print(f'{montage._id}, {len(cnts)} targets found')
    return cnts, True, 'SquareTarget', None


def find_square(image):
    thresh = np.mean(image)
    hist = plot_hist_gauss(image, thresh, size=image.shape[0])
    contours, _ = find_contours(image, thresh)
    contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return contour, np.array([cX, cY]), hist

def check_aspect(rect, ratio=1.4):
    x, y, w, h = rect
    if max(w, h) / min(w, h) < ratio:
        return True
    return False

def rect_to_box(rect):
    x, y, w, h = rect
    return np.array([x, y, x + w, y + h])

def find_targets_binary(image, minsize=20, maxsize=500):
    """
    Finds holes by applying a binary threshold on the image.
    The threshold is automatically evaluated based on the gaussian curve
    fitting on the pixel intensity histogram.
    """
    square_mask = create_square_mask(image)
    image = auto_contrast(image*square_mask)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    result = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)
    done = False

    (mu, sigma, a), is_fit = fit_gauss(blurred)
    if mu < 100:
        sig = 5
    else:
        sig = 3

    while not done:
        threshold = mu + sigma * sig

        cnts, t = find_contours(blurred, threshold)
        cnts = [cv2.boundingRect(cnt) for cnt in cnts]
        cnts = [cnt for cnt in cnts if check_aspect(cnt)]
        median_size = np.median([cnt[2]*cnt[3] for cnt in cnts])
        logger.info(f'Found {len(cnts)} targets with mean size {median_size}')
        cnts = [rect_to_box(cnt) for cnt in cnts if (minsize < cnt[2]*cnt[3] < maxsize)]
        # cnts = [cnt for cnt in cnts if (50 < cv2.contourArea(cnt) < 500)]
 
        for cnt in cnts:
            cv2.rectangle(result, cnt[:2],cnt[2:], (0, 255, 0), 2)

        if len(cnts) < 90 and sig > 2:
            sig -= 0.5
        else:
            done = True

        cv2.imwrite('test.png', result)     

    if len(cnts) < 30:
        return None, False, dict()

    return cnts, True, dict()


def binary_finder(montage):
    pixel_size = montage.pixel_size
    average_hole_size = 1 / (pixel_size / 10000)
    return find_targets_binary(montage.image, minsize=average_hole_size*0.2, maxsize=average_hole_size*5)


def fourrier_filter(im, ang, coords):
    f = np.fft.fft2(im)
    fshift = np.fft.fftshift(f)
    fang = np.angle(fshift)

    fft_test = np.zeros(im.shape, dtype=np.uint8)
    center = np.floor(np.array(fft_test.shape) / 2).astype(int)
    try:
        i, j = coords
        dist = sqrt(i**2 + j**2)
    except:
        dist = coords

    for ind in range(1, 5, 1):
        angle = ang + (90 * ind)
        x = int(round(dist * cos(radians(angle))))
        y = int(round(dist * sin(radians(angle))))
        fft_test[center[0] + y, center[1] + x] = 255
    F = fft_test * np.exp(1j * fang)
    reversed = to_8bits(np.real(np.fft.ifft2(np.fft.ifftshift(F))))
    rev = abs(reversed / 255)
    rev[rev < 0.7] = 0
    rev[rev >= 0.7] = 1
    return rev


def fft_method(montage, diameter_in_um=1.2):
    """ Finds the spacing and angle by finding peaks in the 2D power spectrum of the image. """
    orientation, spacing, square_cont, ratio = calc_angle_spacing(montage.image)
    square_angle = square_cont[0] - square_cont[1]
    square_angle = degrees(atan(square_angle[0] / square_angle[1]))
    bit8_montage = auto_contrast(montage.image)
    bit8_color = cv2.cvtColor(bit8_montage, cv2.COLOR_GRAY2RGB)
    square, _, _ = find_square(bit8_montage)
    mask = np.zeros(bit8_montage.shape, dtype="uint8")
    cv2.drawContours(mask, [square], -1, 255, cv2.FILLED)
    cv2.drawContours(bit8_color, [square], -1, (255, 0, 0), 20)
    dist_pix = 1 / spacing / ratio
    dist = dist_pix / (montage.pixel_size / 10000)
    orientation = 90 - square_angle - orientation
    logger.info(f'Found holey pattern of {round(dist,2)} \u03BCm at {round(orientation,2)}\u00B0')
    pattern_filter = fourrier_filter(montage.image, orientation, 1 / dist_pix)
    product = pattern_filter * mask
    cnts, t = find_contours(product, 254)
    logger.debug(f'Adding {len(cnts)} holes to square')
    outputs = []
    radius_in_pix = int(diameter_in_um * 10000 / montage.pixel_size // 2)
    for cnt in cnts:
        center, radius = cv2.minEnclosingCircle(cnt)
        if radius_in_pix * 0.7 < radius < radius_in_pix * 1.5:
            logger.debug(f'{center}, {radius}')
            outputs.append(convert_centers_to_boxes(
                center,
                montage.pixel_size,
                montage.shape_x,
                montage.shape_y,
                diameter_in_um=diameter_in_um
            ))
            cv2.circle(bit8_color, np.array(center).astype(int), \
                int(radius), (0, 255, 0), cv2.FILLED)
    save_image(bit8_color, 'fft_method', destination=montage.directory, resize_to=512)
    return outputs, True, dict()


def regular_pattern(montage, spacing_in_um=3, diameter_in_um=1.2, **kwargs):
    """ Applies a regular pattern of targets on the image. """
    pixel_size = montage.pixel_size/10000
    radius_in_pix = int(diameter_in_um / pixel_size // 2)
    pixel_spacing = int(spacing_in_um / pixel_size)
    bit8_montage = auto_contrast(montage.image)
    square, _, _ = find_square(bit8_montage)
    output = []
    total_positions = 0
    mask = np.zeros([montage.shape_x,montage.shape_y] , dtype="uint8")
    cv2.drawContours(mask, [square], -1, 255, cv2.FILLED)
    x = radius_in_pix
    while x <= montage.shape_y:
        y = radius_in_pix
        while y <= montage.shape_x:
            if mask[y,x] == 255:
                output.append(convert_centers_to_boxes(
                    np.array([x, y]),
                    montage.pixel_size,
                    montage.shape_x,
                    montage.shape_y,
                    diameter_in_um=diameter_in_um
                ))
            y+= pixel_spacing
            total_positions+=1
        x+= pixel_spacing
    logger.info(f'Filtered a total of {len(output)} targets' + \
        f'from {total_positions} positions within the square')
    return output, True, dict()


def find_square_center(img):
    img = auto_contrast(img)
    thresh = np.mean(img)
    # hist = plot_hist_gauss(montage.montage, thresh, size=montage.montage.shape[0])
    contours, _ = find_contours(img, thresh)
    areas = [cv2.contourArea(cnt) for cnt in contours]
    largest_contour = contours[areas.index(max(areas))]
    M = cv2.moments(largest_contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return np.array([cX, cY])

def create_square_mask(image):
    cnts, center, _ = find_square(image)
    mask = np.zeros(image.shape)
    cv2.drawContours(mask,[cnts],-1,1,cv2.FILLED)
    return mask

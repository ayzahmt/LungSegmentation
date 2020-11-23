import pydicom
import os
import numpy as np
import datetime


#
# Load dicom images from path
#
def load_images(path):
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.InstanceNumber))

    return slices


#
# Convert to HU values
#
def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    image = image.astype(np.int16)

    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)

    return np.array(image, dtype=np.int16)


#
# Save images as an array
#
def save_images_array(path, images):
    x = datetime.datetime.now()
    dateformat = str(x.year) + str(x.month) + str(x.day)

    np.save(path + "fullimages_%s.npy" % (dateformat), images)


#
# Load images (as an array) from npy file
#
def load_images_array(path, yyyymmdd):
    full_path = path + "fullimages_%d.npy" % yyyymmdd
    return np.load(full_path).astype(np.float64)


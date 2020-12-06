import matplotlib.pyplot as plt
import numpy as np


#
# Show HU values on graphic.
#
def show_hu_values(images):
    plt.hist(images.flatten(), bins=50, color='c')
    plt.xlabel("Hounsfield Units Values")
    plt.ylabel("Frequency")
    plt.show()


def show_one_slice(slice):
    plt.imshow(slice, cmap='gray')
    plt.show()


def plot_dicom(images):
    ps = images[0].PixelSpacing
    ss = images[0].SliceThickness
    ax_aspect = ps[1] / ps[0]
    sag_aspect = ps[1] / ss
    cor_aspect = ss / ps[0]

    masked_lung = []

    for img in images:
        masked_lung.append(img)

    plot_dicom_internal(masked_lung, ax_aspect, sag_aspect, cor_aspect)


def plot_dicom_internal(images, ax_aspect, sag_aspect, cor_aspect):
    img_shape = list(images[0].pixel_array.shape)
    img_shape.append(len(images))
    img3d = np.zeros(img_shape)

    for i, s in enumerate(images):
        img2d = s.pixel_array
        img3d[:, :, i] = img2d

    a1 = plt.subplot(2, 2, 1)
    plt.imshow(img3d[:, :, img_shape[2] // 2])
    a1.set_aspect(ax_aspect)

    a2 = plt.subplot(2, 2, 2)
    plt.imshow(img3d[:, img_shape[1] // 2, :])
    a2.set_aspect(sag_aspect)

    a3 = plt.subplot(2, 2, 3)
    plt.imshow(img3d[img_shape[0] // 2, :, :].T)
    a3.set_aspect(cor_aspect)

    plt.show()


def plot_nifti(images):
    img_shape = list(images[0].shape)
    img_shape.append(len(images))
    img3d = np.zeros(img_shape)

    for i, s in enumerate(images):
        img3d[:, :, i] = s

    image_index = img_shape[2] // 2

    a1 = plt.subplot(1, 1, 1)
    plt.imshow(img3d[:, :, image_index])
    a1.set_aspect(1)

    plt.show()


def plot_dicom_and_nifti(dicom_images, nifti_images, segmented_lungs):

    masked_lung = []

    for img in dicom_images:
        masked_lung.append(img)

    img_shape_dicom = list(masked_lung[0].pixel_array.shape)
    img_shape_dicom.append(len(masked_lung))
    img3d_dicom = np.zeros(img_shape_dicom)

    for i, s in enumerate(masked_lung):
        img2d = s.pixel_array
        img3d_dicom[:, :, i] = img2d

    image_index = img_shape_dicom[2] // 2

    a1 = plt.subplot(2, 2, 1)
    plt.imshow(img3d_dicom[:, :, image_index])
    a1.set_aspect(1)

    img_shape_nifti = list(nifti_images[0].shape)
    img_shape_nifti.append(len(nifti_images))
    img3d_nifti = np.zeros(img_shape_nifti)

    for i, s in enumerate(nifti_images):
        img3d_nifti[:, :, i] = s

    a2 = plt.subplot(2, 2, 2)
    plt.imshow(img3d_nifti[:, :, image_index])
    a2.set_aspect(1)

    a3 = plt.subplot(2, 2, 3)
    plt.imshow(segmented_lungs[image_index], cmap='gray')
    a3.set_aspect(1)

    plt.show()




import numpy as np
from skimage import measure
import dicomimages


def create_mask(image):
    # piksel değerlerini 1 ve 2 yapıyoruz
    # backround olarak default 0 değeri set edilir
    binary_image = np.array(image > -300, dtype=np.int8) + 1

    # label backround değerini default 0 alıyor
    # piksel değerlerini etiketliyor (1 den başlıyor)
    labels = measure.label(binary_image)

    # backround için en köşeden seçilir
    background_label = labels[0, 0, 0]

    binary_image[background_label == labels] = 2

    # Measure.label fonksiyonu ile en büyük doku belirlenir ve geri kalanı doldurulur (0 ya da 1)
    for i, single_slice in enumerate(binary_image):
        # pikselleri 1 ve 2 yapmıştık (+1 eklemiştik). Şimdi eski haline çeviriyoruz.
        single_slice = single_slice - 1
        labeling = measure.label(single_slice)
        # backround = 0. Eğer 0 dan büyük değer varsa etiket değerini dön
        l_max = largest_label_volume(labeling, bg=0)

        if l_max is not None:  # akciğer hücresi var demek. akciger = 2
            binary_image[i][labeling != l_max] = 1  # akciğer etiketli olmayanları 1 yap

    # pikselleri 1 ve 2 yapmıştık (+1 eklemiştik). Şimdi eski haline çeviriyoruz.
    binary_image -= 1
    binary_image = 1 - binary_image  # 1 değeri akcigeri gösteriyor.

    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    # backround = 0. Eğer 0 dan büyük değer varsa etiket değerini dön
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None:
        binary_image[labels != l_max] = 0  # akciğer etiketli olmayanları 0 yap

    return binary_image


def largest_label_volume(im, bg=-1):
    values, counts = np.unique(im, return_counts=True)

    counts = counts[values != bg]
    values = values[values != bg]

    if len(counts) > 0:
        return values[np.argmax(counts)]
    else:
        return None


def segment_lung(dicom_path):
    imgs = dicomimages.load_images(dicom_path)
    imgs_with_hu = dicomimages.get_pixels_hu(imgs)

    masks = create_mask(imgs_with_hu)

    imgs_segmented_lung = imgs_with_hu * masks

    return imgs_segmented_lung



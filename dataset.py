import numpy as np
import json

window_size = 25
core_size = 5

LABEL_THRESHOLD = 0.85

CROPPED_IMAGE_INDEX = 0

dataset_cropped_images_path = "C:\\Users\\Ahmet\\Desktop\\tez_verileri\\dataset\\images\\"
dataset_cropped_images_info_path = "C:\\Users\\Ahmet\\Desktop\\tez_verileri\\dataset\\dataset.json"

DATASET = []


#
# Create dataset
#
def create_dataset(segmented_lungs, images_with_label, patient, test):
    # 512*512*57 to convert 57*512*512
    images_with_label = np.transpose(images_with_label, (2, 1, 0))

    for i in range(len(segmented_lungs)):

        labeled_image = images_with_label[i]

        image = segmented_lungs[i]

        x = labeled_image.shape[0]
        y = labeled_image.shape[1]

        # doktor tarafından etiketlenmiş değerleri içeriyorsa
        if 1 in labeled_image or 2 in labeled_image or 3 in labeled_image:
            for m in range(0, x, 3):
                for n in range(0, y, 4):
                    if labeled_image[m][n] != 0:
                        path, label = crop_image(image, labeled_image, m, n)
                        if label != 0:
                            DATASET.append({
                                #'patient_id': patient.id,
                                #'patient_name': patient.name,
                                #'patient_surname': patient.surname,
                                'image_path': path,
                                'label': str(label)
                            })

    with open(dataset_cropped_images_info_path, 'w') as file:
        file.write(json.dumps(DATASET, indent=4, sort_keys=True))


#
# Crop image by defined size (window_size)
#
def crop_image(image, labeled_image, row, col):
    middle_index = int(window_size / 2)

    x_begin = row - middle_index
    x_end = row + middle_index + 1

    y_begin = col - middle_index
    y_end = col + middle_index + 1

    cropped_labeled_img = labeled_image[x_begin:x_end, y_begin:y_end]

    cropped_image = image[x_begin:x_end, y_begin:y_end]

    label = get_most_used_label(cropped_labeled_img, middle_index)

    global CROPPED_IMAGE_INDEX
    CROPPED_IMAGE_INDEX = CROPPED_IMAGE_INDEX + 1

    path = dataset_cropped_images_path + str(CROPPED_IMAGE_INDEX) + ".txt"

    if label != 0:
        np.savetxt(path, cropped_image)
    return path, label


def get_most_used_label(labeled_image, middle_index):
    img = []

    x = labeled_image.shape[0]  # = windows size
    y = labeled_image.shape[1]  # = windows size

    max_index = middle_index + core_size
    min_index = middle_index - core_size

    for i in range(x):
        for j in range(y):
            if min_index < i < max_index and min_index < j < max_index:
                img.append(labeled_image[i][j])

    # sıfırların sayısını bul
    zero_count = np.count_nonzero((np.array(img) == 0))
    # sıfırlar LABEL_THRESHOLD değerinden fazla ise direkt 0 değerini dön
    if zero_count / len(img) > LABEL_THRESHOLD:
        return 0

    # etikekleri (values) ve sayılarını (counts) hesapla
    (values, counts) = np.unique(np.array(img), return_counts=True)
    # sayısı fazla olanın indisini dön
    ind = np.argmax(counts)
    if values[ind] == 0:
        ind = 1 + np.argmax(counts[1:])

    label = values[ind]

    return label

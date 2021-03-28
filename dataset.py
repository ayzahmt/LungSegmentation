import numpy as np
import json
from random import shuffle
from tqdm import tqdm

window_size = 25
core_size = 5

LABEL_THRESHOLD = 0.85

CROPPED_IMAGE_INDEX = 0

dataset_cropped_images_path = "C:\\Users\\Ahmet\\Desktop\\tez_verileri\\dataset\\images3\\"
dataset_cropped_images_info_path = "C:\\Users\\Ahmet\\Desktop\\tez_verileri\\dataset\\dataset3.json"
images_info_path = "C:\\Users\\Ahmet\\Desktop\\tez_verileri\\dataset\\images3.npy"
labels_info_path = "C:\\Users\\Ahmet\\Desktop\\tez_verileri\\dataset\\labels3.npy"

train_dataset_info_path = "C:\\Users\\Ahmet\\Desktop\\tez_verileri\\dataset\\train_dataset3.json"
test_dataset_info_path = "C:\\Users\\Ahmet\\Desktop\\tez_verileri\\dataset\\test_dataset3.json"

test_info_path = "C:\\Users\\Ahmet\\Desktop\\tez_verileri\\dataset\\test175.json"
test_image_info_path = "C:\\Users\\Ahmet\\Desktop\\tez_verileri\\dataset\\testimages175.npy"
test_label_info_path = "C:\\Users\\Ahmet\\Desktop\\tez_verileri\\dataset\\testlabels175.npy"

DATASET = []


#
# Create dataset
#
def create_dataset(segmented_lungs, images_with_label, test):
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
                                # 'patient_id': patient.id,
                                # 'patient_name': patient.name,
                                # 'patient_surname': patient.surname,
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


#
# Create train and test dataset
#
def create_train_and_test_dataset():
    # crop edilmiş görüntülerin bilgileri alınır
    with open(dataset_cropped_images_info_path, "r") as file:
        info_datas = json.load(file)
        # bilgiler karıştırılır
        shuffle(info_datas)

        train_rate = round(0.80 * len(info_datas))

        train_dataset = info_datas[:train_rate]
        train_file = open(train_dataset_info_path, "w")
        train_file.write(json.dumps(train_dataset, indent=4, sort_keys=True))
        train_file.close()

        test_dataset = info_datas[train_rate:]
        test_file = open(test_dataset_info_path, "w")
        test_file.write(json.dumps(test_dataset, indent=4, sort_keys=True))
        test_file.close()


def create_images_and_labels():
    data = json.load(open(dataset_cropped_images_info_path, "r"))
    images = []
    labels = []
    for row in tqdm(data):
        img = np.loadtxt(row['image_path'])
        images.append(img.reshape(25, 25, 1))

        if row['label'] == '1':
            labels.append([1, 0, 0])
        elif row['label'] == '2':
            labels.append([0, 1, 0])
        elif row['label'] == '3':
            labels.append([0, 0, 1])

    np.save(images_info_path, images)
    np.save(labels_info_path, labels)


def get_data():
    images = np.load(images_info_path).astype(np.float64)
    labels = np.load(labels_info_path).astype(np.float64)

    return images, labels


def get_test_mini_data():
    images = np.load(test_image_info_path).astype(np.float64)
    labels = np.load(test_label_info_path).astype(np.float64)

    return images, labels


def create_test_mini_images_and_labels():
    data = json.load(open(test_info_path, "r"))
    images = []
    labels = []
    for row in tqdm(data):
        img = np.loadtxt(row['image_path'])
        images.append(img.reshape(25, 25, 1))

        if row['label'] == '1':
            labels.append([1, 0, 0])
        elif row['label'] == '2':
            labels.append([0, 1, 0])
        elif row['label'] == '3':
            labels.append([0, 0, 1])

    np.save(test_image_info_path, images)
    np.save(test_label_info_path, labels)
    #return np.array(images), np.array(labels)


if __name__ == '__main__':
    create_test_mini_images_and_labels();


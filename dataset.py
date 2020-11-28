
#
# Create dataset
#
def create_dataset(segmented_lungs, images_with_label):
    for i in range(len(segmented_lungs)):
        labeled_image = images_with_label[i]

        x = labeled_image.shape[0]
        y = labeled_image.shape[1]

        # doktor tarafından etiketlenmiş değerleri içeriyorsa
        if 1 in labeled_image or 2 in labeled_image or 3 in labeled_image:
            for m in range(0, x, 3):
                for n in range(0, y, 3):
                    if labeled_image[m][n] != 0:
                        return None
                    #düzenlenecek


import dicomimages
import visualization
import segmentation
import dataset
from patientContract import Patient
from patientContract import PatientPath
import numpy as np

#######
# Initialize parameter and run methods
#######
data_path = "C:\\Users\\Ahmet\\Desktop\\tez_verileri\\tez\\dataset\\"

p1_dicom_path = "C:\\Users\\Ahmet\\Desktop\\tez_verileri\\msc\\scans\\scans\\Adem Acar\\12-12-2016 bt\\DICOM\\ST000000\\SE000003\\"
p1_nifti_path = "C:\\Users\\Ahmet\\Desktop\\tez_verileri\\msc\\isaretlemeler\\HRCT 2.nii"

p2_nifti_path = "C:\\Users\\Ahmet\\Desktop\\tez_verileri\\msc\\isaretlemeler\\HRCT.nii"
p2_dicom_path = "C:\\Users\\Ahmet\\Desktop\\tez_verileri\\msc\\scans\\scans\\Adem Acar\\11-5-2016 bt\\DICOM\\ST000000\\SE000001"

p3_nifti_path = "C:\\Users\\Ahmet\\Desktop\\tez_verileri\\msc\\isaretlemeler\\jakvalid.nii"
p3_dicom_path = "C:\\Users\\Ahmet\\Desktop\\tez_verileri\\msc\\scans\\scans\\Jak Valid Sevindiren\\DICOM\\S00001\\SER00002"

p4_nifti_path = "C:\\Users\\Ahmet\\Desktop\\tez_verileri\\msc\\isaretlemeler\\3a.nii"
p4_dicom_path = "C:\\Users\\Ahmet\\Desktop\\tez_verileri\\msc\\\scans\\scans\\Fatma Demirsoy"


paths = [PatientPath(p1_dicom_path, p1_nifti_path)
         #,PatientPath(p2_dicom_path, p2_nifti_path)
         #,PatientPath(p3_dicom_path, p3_nifti_path)
         #,PatientPath(p4_dicom_path, p4_nifti_path)
         ]

# imgs = dicomimages.get_pixels_hu(patient)
# dicomimages.save_images_array(data_path, imgs)

# imgs = dicomimages.load_images_array(data_path, 2020109)
# imgs = segmentation.create_mask(imgs)

dices = []
def dice_metric(segmented_images, labeled_images):
    labeled_images_transpose = np.transpose(labeled_images, (2, 1, 0))

    for i in range(0, len(segmented_lungs)):
        dice = segmentation.dice_metric_coeffecient(segmented_images[i], labeled_images_transpose[i])
        dices.append(dice)
        print(dice)

    return dices


for path in paths:

    images = dicomimages.load_images(path.dicom)
    #patient = Patient(images[0].PatientID, images[0].PatientName.given_name, images[0].PatientName.family_name)

    images_with_hu = dicomimages.get_pixels_hu(images)

    labeled_images = dicomimages.load_nifti_file(path.nifti)

    segmented_lungs = segmentation.segment_lung(images_with_hu)

    #visualization.show_masked_lung(images)

    #visualization.show_nifti_lung(labeled_images)

    visualization.plot_dicom_and_nifti(images, labeled_images, segmented_lungs)

    #dice = dice_metric(segmented_lungs, labeled_images)

    #print("Dice metrix average for patient:", sum(dice) / len(dice))

    #dataset.create_dataset(segmented_lungs, labeled_images, patient, test=False)


    #visualization.show_one_slice(segmented_lungs[30])

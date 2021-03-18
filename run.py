import dicomimages
import visualization
import segmentation
import dataset
from patientContract import Patient
from patientContract import PatientPath
import numpy as np
import os
import json

#######
# Initialize parameter and run methods
#######
data_path = "C:\\Users\\Ahmet\\Desktop\\tez_verileri\\tez\\dataset\\"
files_path = os.path.join(os.getcwd(), 'files')
dice_result_path = os.path.join(files_path, 'dice_max_with_morphology.json')


p1_dicom_path = "C:\\Users\\Ahmet\\Desktop\\tez_verileri\\msc\\scans\\scans\\Adem Acar\\12-12-2016 bt\\DICOM\\ST000000\\SE000003\\"
p1_nifti_path = "C:\\Users\\Ahmet\\Desktop\\tez_verileri\\msc\\isaretlemeler\\HRCT 2.nii"

p2_nifti_path = "C:\\Users\\Ahmet\\Desktop\\tez_verileri\\msc\\isaretlemeler\\HRCT.nii"
p2_dicom_path = "C:\\Users\\Ahmet\\Desktop\\tez_verileri\\msc\\scans\\scans\\Adem Acar\\11-5-2016 bt\\DICOM\\ST000000\\SE000001"

p3_nifti_path = "C:\\Users\\Ahmet\\Desktop\\tez_verileri\\msc\\isaretlemeler\\jakvalid.nii"
p3_dicom_path = "C:\\Users\\Ahmet\\Desktop\\tez_verileri\\msc\\scans\\scans\\Jak Valid Sevindiren\\DICOM\\S00001\\SER00002"

p4_nifti_path = "C:\\Users\\Ahmet\\Desktop\\tez_verileri\\msc\\isaretlemeler\\3a.nii"
p4_dicom_path = "C:\\Users\\Ahmet\\Desktop\\tez_verileri\\msc\\\scans\\scans\\Fatma Demirsoy"

p5_nifti_path = "C:\\Users\\Ahmet\\Desktop\\tez_verileri\\msc\\isaretlemeler\\7b.nii"
p5_dicom_path = "C:\\Users\\Ahmet\\Desktop\\tez_verileri\\msc\\\scans\\scans\\Sabih Tansal\\KASIM 2016 BT\\DICOM\\ST000000\\SE000001"

''' 
Doktor tarafında neredeyse çok az bi kısım işaretlenmiş, 
p5_nifti_path = "C:\\Users\\Ahmet\\Desktop\\tez_verileri\\msc\\isaretlemeler\\5c.nii"
p5_dicom_path = "C:\\Users\\Ahmet\\Desktop\\tez_verileri\\msc\\\scans\\scans\\"
'''

paths = [PatientPath(p1_dicom_path, p1_nifti_path)
         ,PatientPath(p2_dicom_path, p2_nifti_path)
         ,PatientPath(p3_dicom_path, p3_nifti_path)
         ,PatientPath(p4_dicom_path, p4_nifti_path)
         ,PatientPath(p5_dicom_path, p5_nifti_path)
         ]

# imgs = dicomimages.get_pixels_hu(patient)
# dicomimages.save_images_array(data_path, imgs)

# imgs = dicomimages.load_images_array(data_path, 2020109)
# imgs = segmentation.create_mask(imgs)

dices = []
dice_results = []
def dice_metric(segmented_images, labeled_images, path):

    middle_slice_index = int(len(segmented_images) // 2)
    begin = middle_slice_index_per_5_begin = middle_slice_index - int(middle_slice_index * 10 / 100)
    end = middle_slice_index_per_5_end = middle_slice_index + int(middle_slice_index * 10 / 100)

    slice_count = 0
    i = middle_slice_index
    dice = segmentation.dice_metric_coeffecient(segmented_images[i], labeled_images[i])
    dice_results.append({
        'patient_nifti': path.nifti,
        'slice': i,
        'dice_rate': round(dice, 2)
    })

    with open(dice_result_path, 'w') as file:
        file.write(json.dumps(dice_results, indent=4))

'''
    for i in range(begin, end+1):
        dice = segmentation.dice_metric_coeffecient(segmented_images[i], labeled_images[i])
        dices.append(dice)
        print(str(i) + ".slice - " + str(dice))
        slice_count = slice_count + 1

        dice_results.append({
            'patient_nifti': path.nifti,
            'slice': i,
            'dice_rate': round(dice, 2)
        })
        

    print(str(slice_count) + " tane slice için Dice metrix ortalama:", sum(dices) / len(dices))
    print()
    '''




for path in paths:

    images = dicomimages.load_images(path.dicom)
    #patient = Patient(images[0].PatientID, images[0].PatientName.given_name, images[0].PatientName.family_name)

    images_with_hu = dicomimages.get_pixels_hu(images)

    labeled_images = dicomimages.load_nifti_file(path.nifti)

    segmented_lungs = segmentation.segment_lung(images_with_hu)

    #segmented_lungs = images_with_hu * segmented_lungs

    #visualization.show_masked_lung(images)

    #visualization.show_nifti_lung(labeled_images)

    #visualization.plot_dicom_and_nifti(images, labeled_images, segmented_lungs)

    #visualization.plot_nifti(labeled_images)

    dice_metric(segmented_lungs, labeled_images, path)

    #print("Dice metrix average for patient:", sum(dice) / len(dice))

    #dataset.create_dataset(segmented_lungs, labeled_images, patient, test=False)


    #visualization.show_one_slice(segmented_lungs[30])

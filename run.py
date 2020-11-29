import dicomimages
import visualization
import segmentation
import dataset
from patientContract import Patient

#######
# Initialize parameter and run methods
#######
p1_dicom_path = "C:\\Users\\Ahmet\\Desktop\\tez_verileri\\msc\\scans\\scans\\Adem Acar\\12-12-2016 bt\\DICOM\\ST000000\\SE000003\\"
data_path = "C:\\Users\\Ahmet\\Desktop\\tez_verileri\\tez\\dataset\\"
p1_nifti_path = "C:\\Users\\Ahmet\\Desktop\\tez_verileri\\msc\\isaretlemeler\\HRCT 2.nii"


# imgs = dicomimages.get_pixels_hu(patient)
# dicomimages.save_images_array(data_path, imgs)

# imgs = dicomimages.load_images_array(data_path, 2020109)
# imgs = segmentation.create_mask(imgs)


imgs = dicomimages.load_images(p1_dicom_path)
patient = Patient(imgs[0].PatientID, imgs[0].PatientName.given_name, imgs[0].PatientName.family_name)

imgs_with_hu = dicomimages.get_pixels_hu(imgs)

labeled_imgs = dicomimages.load_nifti_file(p1_nifti_path)
segmented_lungs = segmentation.segment_lung(imgs_with_hu)
dataset.create_dataset(segmented_lungs, labeled_imgs, patient)


#visualization.show_one_slice(imgs[30])

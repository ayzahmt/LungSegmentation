import dicomimages
import visualization


#######
# Initialize parameter and run methods
#######
p1_dicom_path = "C:\\Users\\Ayaz\\Desktop\\msc\\scans\\scans\\Adem Acar\\12-12-2016 bt\\DICOM\\ST000000\\SE000003\\"
data_path = "C:\\Users\\Ayaz\\Desktop\\tez\\dataset\\"

patient = dicomimages.load_images(p1_dicom_path)
imgs = dicomimages.get_pixels_hu(patient)
dicomimages.save_images_array(data_path, imgs)
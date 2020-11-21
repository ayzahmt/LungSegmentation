import dicomimages
import visualization
import segmentation

#######
# Initialize parameter and run methods
#######
p1_dicom_path = "C:\\Users\\Ahmet\\Desktop\\tez_verileri\\msc\\scans\\scans\\Adem Acar\\12-12-2016 bt\\DICOM\\ST000000\\SE000003\\"
data_path = "C:\\Users\\Ahmet\\Desktop\\tez_verileri\\tez\\dataset\\"

# patient = dicomimages.load_images(p1_dicom_path)
# imgs = dicomimages.get_pixels_hu(patient)
# dicomimages.save_images_array(data_path, imgs)

# imgs = dicomimages.load_images_array(data_path, 2020109)
# imgs = segmentation.create_mask(imgs)

imgs = segmentation.segment_lung(p1_dicom_path)

visualization.show_one_slice(imgs[30])

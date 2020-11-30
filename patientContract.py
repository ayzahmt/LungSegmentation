class Patient:

    def __init__(self, id, name, surname):
        self.id = id
        self.name = name
        self.surname = surname


class PatientPath:

    def __init__(self, dicom, nifti):
        self.dicom = dicom
        self.nifti = nifti

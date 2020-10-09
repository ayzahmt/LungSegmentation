import matplotlib.pyplot as plt


#
# show HU values on graphic.
#
def show_hu_values(images):
    plt.hist(images.flatten(), bins=50, color='c')
    plt.xlabel("Hounsfield Units (HU)")
    plt.ylabel("Frequency")
    plt.show()


def show_one_slice(slice):
    plt.imshow(slice, cmap='gray')
    plt.show()

import matplotlib.pyplot as plt


#
# Show HU values on graphic.
#
def show_hu_values(images):
    plt.hist(images.flatten(), bins=50, color='c')
    plt.xlabel("Hounsfield Units Values")
    plt.ylabel("Frequency")
    plt.show()


def show_one_slice(slice):
    plt.imshow(slice, cmap='gray')
    plt.show()

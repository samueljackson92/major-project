import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_multiple_images(images):
    fig = plt.figure()

    num_images = len(images)
    for i, image in enumerate(images):
        axis = fig.add_subplot(1, num_images, i+1)
        plt.imshow(image, cmap=cm.Greys_r)

    plt.show()

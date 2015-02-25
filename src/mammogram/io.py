"""
Various IO utility functions.
"""

import os

def iterate_directory(directory, img_ext=".png"):
	""" Iterate of a directory of images

	:param directory: the directory to iterate over.
	:param img_ext: expected extension of the images.
	:returns: iterator to the image paths in the directory
	"""
	check_is_directory(directory)
	for img_name in os.listdir(directory):
		img_path = os.path.join(directory,img_name)
		check_is_image(img_path, img_ext)
		yield img_path


def check_is_directory(directory):
	"""Check that the specified path is a directory

	:param directory: path to check if it is a directory
	:raises: ValueError
	"""
	if not os.path.isdir(directory):
		raise ValueError("%s is not a directory" % directory)


def check_is_image(img_path, ext):
	"""Check that the specified path is an image with the expected extension

	:param directory: path to check if it is a image
	:raises: ValueError
	"""
	if not os.path.isfile(img_path):
		raise ValueError("%s is not a file" % img_path)

	if not img_path.endswith(ext):
		raise ValueError("%s does not have the expected file extension"
						  % img_path)

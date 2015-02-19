import os.path
from skimage import io

location = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_FOLDER = os.path.join(location, "./test_data")

def load_file(file_name):
    path = os.path.join(TEST_DATA_FOLDER, file_name)
    return io.imread(path, as_grey=True)

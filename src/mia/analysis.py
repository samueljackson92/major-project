""" Analysis module for examining the results of running feature detection on
a dataset of images.
"""

import functools
import logging
import numpy as np
import pandas as pd

from sklearn import manifold, preprocessing

logger = logging.getLogger(__name__)


def _handle_data_frame(func):
    @functools.wraps(func)
    def inner(feature_matrix, **kwargs):
        if isinstance(feature_matrix, pd.DataFrame):
            fit_output = func(feature_matrix.as_matrix(), **kwargs)
            return pd.DataFrame(fit_output, index=feature_matrix.index)
        else:
            return func(feature_matrix)
    return inner


@_handle_data_frame
def tSNE(feature_matrix, **kwargs):
    """Run the t-SNE algorithm on the feature space of a collection of images

    :param feature_matrix: matrix of features use with the t-SNE
    :returns: 2darray -- lower dimensional mapping of the t-SNE algorithm
    """
    feature_matrix = standard_scaler(feature_matrix)
    tSNE = manifold.TSNE(**kwargs)
    fit_output = tSNE.fit_transform(feature_matrix)
    return fit_output


@_handle_data_frame
def standard_scaler(feature_matrix):
    scalar = preprocessing.StandardScaler()
    feature_matrix = scalar.fit_transform(feature_matrix)
    return feature_matrix


@_handle_data_frame
def normalize_data_frame(feature_matrix):
    return preprocessing.normalize(feature_matrix)


def measure_closeness(data_frame, labels):
    groups = data_frame.groupby(labels)
    ds = [_cluster_measure(frame) for index, frame in groups]
    return pd.Series(ds, index=labels.unique())


def _cluster_measure(group):
    points = group[[0, 1]]
    centroid = points.sum() / group.size
    distances = ((centroid - points)**2).sum(axis=1)
    distances = distances.apply(np.sqrt)
    return distances.mean()


# def feature_statistics(raw_detections):
#     """Create blob features from a file of blob detections
#
#     :param csv_file: file containing the detected blobs
#     :param output_file: name to output the resulting features to.
#     """
#     image_names = raw_detections['image_name'].unique()
#
#     info_df = _create_info_data_frame(raw_detections, image_names)
#     feature_matrix = _create_feature_matrix(raw_detections, image_names)
#
#     feature_matrix = pd.concat([feature_matrix, info_df], axis=1)
#     return feature_matrix
#
#
# def load_synthetic_meta_data(file_name):
#     labels = pd.DataFrame.from_csv(file_name)
#
#     regex_string = r"Output-test_Mix_DPerc(\d+)_c-02_26_2015-\d+_\d+_\d+"
#     name_regex = re.compile(regex_string)
#     group_names = [l for l in labels.index]
#     group_ids = [int(re.match(name_regex, name).group(1))
#                  for name in group_names]
#     labels.index = group_ids
#     return labels
#
#
# def create_meta_data_for_synthetic_mammogram(data_frame, labels):
#     ids = [[group_id for name in data_frame.img_name
#             if "DPerc%d" % group_id in name]
#            for group_id in labels.index]
#
#     ids_by_group = pd.Series([i for row in ids for i in row],
#                              index=data_frame.img_name)
#     cls_by_group = pd.Series([labels.BIRADS[i] for row in ids for i in row],
#                              index=data_frame.img_name)
#
#     return pd.DataFrame({'group_id': ids_by_group, 'class': cls_by_group})
#
#
# def _create_info_data_frame(raw_detections, index_names):
#     info_columns = ['patient_id', 'view', 'side', 'class']
#     info_df = raw_detections[info_columns].drop_duplicates()
#     info_df.index = index_names
#     return info_df
#
#
# def _create_feature_matrix(raw_detections, index_names):
#     feature_matrix = pd.DataFrame()
#     for index, frame in raw_detections.groupby('image_name'):
#         shape_props = blob_props(frame)
#         # int_props = intensity_props(frame)
#         row = shape_props
#         # row = pd.concat([shape_props, int_props], axis=1)
#         feature_matrix = pd.concat([feature_matrix, row], ignore_index=True)
#
#     feature_matrix.index = index_names
#     return feature_matrix

# def _make_image_data_frame(img_name, df):
#     info_df = _make_image_info_data_frame(img_name)
#     info_df = pd.concat([info_df]*df.shape[0], ignore_index=True)
#     frame = pd.concat([df, info_df], axis=1)
#     return frame
#
#
# def _make_image_info_data_frame(img_name):
#     _, ext = os.path.splitext(img_name)
#
#     if ext == '.dcm':
#         return _make_synthetic_info_data_frame(img_name)
#     else:
#         return _make_mammogram_info_data_frame(img_name)
#
#
# def _make_synthetic_info_data_frame(img_name):
#     return pd.DataFrame([img_name], columns=['img_name'])
#
#
# def _make_mammogram_info_data_frame(img_name):
#     name_regex = re.compile(r'p(\d{3}-\d{3}-\d{5})-([a-z]{2})\.png')
#
#     def find_image_info(name):
#         m = re.match(name_regex, name)
#         patient_id = m.group(1).replace('-', '')
#         view, side = list(m.group(2))
#         return [img_name, patient_id, view, side]
#
#     image_info = find_image_info(img_name)
#     return pd.DataFrame([image_info], columns=['image_name', 'patient_id',
#                                                'view', 'side'])
#
#
# def add_BIRADS_class(feature_matrix, class_labels_file):
#     """Add the BIRADS classes to the data frame given a file with the class
#     labels
#
#     :param feature_matrix: DataFrame for features where the index is the imag
#e
#                            names
#     :param class_labels_file: csv file containg the class labels
#     :returns: DataFrame with the class labels added under the column 'class'
#     """
#     class_labels = pd.DataFrame().from_csv(class_labels_file)
#     class_labels = class_labels['BI-RADS']
#
#     class_hash = {}
#     for img, c in class_labels.iteritems():
#         class_hash[img] = c
#
#     img_classes = [class_hash[int(name)]
#                    for name in feature_matrix['patient_id']]
#     feature_matrix['class'] = pd.Series(img_classes,
#                                         index=feature_matrix.index)
#     return feature_matrix

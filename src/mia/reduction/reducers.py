import logging
import numpy as np
import os.path
import pandas as pd

from mia.reduction.multi_processed_reduction import MultiProcessedReduction
from mia.features.blobs import detect_blobs
from mia.features.linear_structure import detect_linear
from mia.features.intensity import detect_intensity
from mia.features.texture import detect_texture
from mia.utils import preprocess_image

logger = logging.getLogger(__name__)


class BlobFeaturesReduction(MultiProcessedReduction):

    def process_image(self, image_path, mask_path):
        img_name = os.path.basename(image_path)

        img, msk = preprocess_image(image_path, mask_path)
        blob_props = detect_blobs(img, msk)
        blob_props.index = pd.Series([img_name] * blob_props.shape[0])
        blob_props['breast_area'] = np.count_nonzero(msk)

        logger.info("%d blobs found in image %s"
                    % (blob_props.shape[0], img_name))

        return blob_props


class LineFeaturesReduction(MultiProcessedReduction):

    def process_image(self, image_path, mask_path):
        img, msk = preprocess_image(image_path, mask_path)
        img_name = os.path.basename(image_path)

        line_props, _ = detect_linear(img, msk)
        line_props.index = pd.Series([img_name] * line_props.shape[0])
        line_props['breast_area'] = np.count_nonzero(msk)

        logger.info("%d blobs found in image %s"
                    % (line_props.shape[0], img_name))

        return line_props


class PatchIntensityFeaturesReduction(MultiProcessedReduction):

    def process_image(self, img_path, msk_path, patch_frame):
        img, msk = preprocess_image(img_path, msk_path)
        img_name = os.path.basename(img_path)

        patch = patch_frame.loc[[img_name]]
        logger.info("Detecting intensity features in %d patches"
                    % patch.shape[0])

        intensity_props = detect_intensity(img, patch)
        intensity_props.index = \
            pd.Series([img_name] * intensity_props.shape[0])
        return intensity_props


class PatchTextureFeaturesReduction(MultiProcessedReduction):

    def process_image(self, img_path, msk_path, patch_frame):
        img, msk = preprocess_image(img_path, msk_path)
        img_name = os.path.basename(img_path)

        patch = patch_frame.loc[[img_name]]
        logger.info("Detecting texture features in %d patches"
                    % patch.shape[0])

        texture_props = detect_texture(img, patch)
        texture_props.index = pd.Series([img_name] * texture_props.shape[0])
        return texture_props

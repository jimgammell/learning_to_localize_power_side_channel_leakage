from common import *
from utils.download_unzip import download, unzip
from .base_model import GenericZaidNet, GenericWoutersNet
from .keras_to_pytorch_utils import *

urls = [
    r'https://github.com/KULeuven-COSIC/TCHES20V3_CNN_SCA/raw/refs/heads/master/models/pretrained_models.z01',
    r'https://github.com/KULeuven-COSIC/TCHES20V3_CNN_SCA/raw/refs/heads/master/models/pretrained_models.zip'
]
_models_dir = os.path.join(RESOURCE_DIR, 'zaid_wouters_pretrained')
os.makedirs(_models_dir, exist_ok=True)
should_unzip = False
for url in urls:
    output_name = url.split(os.sep)[-1]
    if not os.path.exists(os.path.join(_models_dir, output_name)):
        download(url, os.path.join(_models_dir, output_name))
        should_unzip = True
if should_unzip:
    unzip(output_name, _models_dir, split=True)
WEIGHTS_BASE_DIR = os.path.join(_models_dir, 'pretrained_models', 'models')

class ZaidNet__ASCADv1f(GenericZaidNet):
    def __init__(self, pretrained_seed: Optional[int] = None):
        super().__init__(
            input_shape=(1, 700),
            output_classes=256,
            block_settings=[{'channels': 4, 'conv_kernel_size': 1, 'pool_size': 2}],
            dense_widths=[10, 10]
        )
        self.pretrained_seed = pretrained_seed
        if self.pretrained_seed is not None:
            self.load_pretrained_keras_params()
    
    def load_pretrained_keras_params(self):
        assert self.pretrained_seed is not None and 0 <= self.pretrained_seed < 10
        weights_filename = f'zaid_ascad_desync_0_feature_standardization_{self.pretrained_seed}.hdf5'
        weights_path = os.path.join(WEIGHTS_BASE_DIR, weights_filename)
        keras_params = unpack_keras_params(weights_path)
        keras_to_torch_mod(keras_params['block1_conv1'], self.conv_stage.block_1.conv)
        keras_to_torch_mod(keras_params['block1_norm1'], self.conv_stage.block_1.norm)
        keras_to_torch_mod(keras_params['fc1'], self.fc_stage.dense_1)
        keras_to_torch_mod(keras_params['fc2'], self.fc_stage.dense_2)
        keras_to_torch_mod(keras_params['predictions'], self.fc_stage.classifier)

class ZaidNet__DPAv4(GenericZaidNet):
    def __init__(self, pretrained_seed: Optional[int] = None):
        super().__init__(
            input_shape=(1, 4000),
            output_classes=256,
            block_settings=[{'channels': 2, 'conv_kernel_size': 1, 'pool_size': 2}],
            dense_widths=[2]
        )
        self.pretrained_seed = pretrained_seed
        if self.pretrained_seed is not None:
            self.load_pretrained_keras_params()
    
    def load_pretrained_keras_params(self):
        assert self.pretrained_seed is not None and 0 <= self.pretrained_seed < 10
        weights_filename = f'zaid_dpav4_feature_standardization_{self.pretrained_seed}.hdf5'
        weights_path = os.path.join(WEIGHTS_BASE_DIR, weights_filename)
        keras_params = unpack_keras_params(weights_path)
        keras_to_torch_mod(keras_params['block1_conv1'], self.conv_stage.block_1.conv)
        keras_to_torch_mod(keras_params['block1_norm1'], self.conv_stage.block_1.norm)
        keras_to_torch_mod(keras_params['fc1'], self.fc_stage.dense_1)
        keras_to_torch_mod(keras_params['predictions'], self.fc_stage.classifier)

class ZaidNet__AES_HD(GenericZaidNet):
    def __init__(self, pretrained_seed: Optional[int] = None):
        super().__init__(
            input_shape=(1, 1250),
            output_classes=256,
            block_settings=[{'channels': 2, 'conv_kernel_size': 1, 'pool_size': 2}],
            dense_widths=[2]
        )
        self.pretrained_seed = pretrained_seed
        if self.pretrained_seed is not None:
            self.load_pretrained_keras_params()
    
    def load_pretrained_keras_params(self):
        assert self.pretrained_seed is not None and 0 <= self.pretrained_seed < 10
        weights_filename = f'zaid_aes_hd_feature_standardization_{self.pretrained_seed}.hdf5'
        weights_path = os.path.join(WEIGHTS_BASE_DIR, weights_filename)
        keras_params = unpack_keras_params(weights_path)
        keras_to_torch_mod(keras_params['block1_conv1'], self.conv_stage.block_1.conv)
        keras_to_torch_mod(keras_params['block1_norm1'], self.conv_stage.block_1.norm)
        keras_to_torch_mod(keras_params['fc1'], self.fc_stage.dense_1)
        keras_to_torch_mod(keras_params['predictions'], self.fc_stage.classifier)

class WoutersNet__ASCADv1f(GenericWoutersNet):
    def __init__(self, pretrained_seed: Optional[int] = None):
        super().__init__(
            input_shape=(1, 700),
            output_classes=256,
            input_pool_size=2,
            dense_widths=[10, 10]
        )
        self.pretrained_seed = pretrained_seed
        if self.pretrained_seed is not None:
            self.load_pretrained_keras_params()
    
    def load_pretrained_keras_params(self):
        assert self.pretrained_seed is not None and 0 <= self.pretrained_seed < 10
        weights_filename = f'noConv1_ascad_desync_0_feature_standardization_{self.pretrained_seed}.hdf5'
        weights_path = os.path.join(WEIGHTS_BASE_DIR, weights_filename)
        keras_params = unpack_keras_params(weights_path)
        keras_to_torch_mod(keras_params['fc1'], self.fc_stage.dense_1)
        keras_to_torch_mod(keras_params['fc2'], self.fc_stage.dense_2)
        keras_to_torch_mod(keras_params['predictions'], self.fc_stage.classifier)

class WoutersNet__DPAv4(GenericWoutersNet):
    def __init__(self, pretrained_seed: Optional[int] = None):
        super().__init__(
            input_shape=(1, 4000),
            output_classes=256,
            input_pool_size=2,
            dense_widths=[2]
        )
        self.pretrained_seed = pretrained_seed
        if self.pretrained_seed is not None:
            self.load_pretrained_keras_params()
    
    def load_pretrained_keras_params(self):
        assert self.pretrained_seed is not None and 0 <= self.pretrained_seed < 10
        weights_filename = f'noConv1_dpav4_feature_standardization_{self.pretrained_seed}.hdf5'
        weights_path = os.path.join(WEIGHTS_BASE_DIR, weights_filename)
        keras_params = unpack_keras_params(weights_path)
        keras_to_torch_mod(keras_params['fc1'], self.fc_stage.dense_1)
        keras_to_torch_mod(keras_params['predictions'], self.fc_stage.classifier)

class WoutersNet__AES_HD(GenericWoutersNet):
    def __init__(self, pretrained_seed: Optional[int] = None):
        super().__init__(
            input_shape=(1, 1250),
            output_classes=256,
            input_pool_size=2,
            dense_widths=[2]
        )
        self.pretrained_seed = pretrained_seed
        if self.pretrained_seed is not None:
            self.load_pretrained_keras_params()
    
    def load_pretrained_keras_params(self):
        assert self.pretrained_seed is not None and 0 <= self.pretrained_seed < 10
        weights_filename = f'noConv1_aes_hd_feature_standardization_{self.pretrained_seed}.hdf5'
        weights_path = os.path.join(WEIGHTS_BASE_DIR, weights_filename)
        keras_params = unpack_keras_params(weights_path)
        keras_to_torch_mod(keras_params['fc1'], self.fc_stage.dense_1)
        keras_to_torch_mod(keras_params['predictions'], self.fc_stage.classifier)
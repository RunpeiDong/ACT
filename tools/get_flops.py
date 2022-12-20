import torch
import torch.nn as nn
from tools import builder
from utils import misc, dist_utils, parser
from utils.logger import *
from utils.config import *

try:
    from mmcv.cnn import get_model_complexity_info
except ImportError:
    raise ImportError('Please upgrade mmcv to >0.6.2')

def main():

    args = parser.get_args()
    input_shape = tuple([1024, 3])

    logger = get_logger('flops')
    print_log('Tester start ... ', logger=logger)

    # config
    config = get_config(args, logger=logger)

    model = builder.model_builder(config.model)

    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    # if hasattr(model, 'forward_dummy'):
    #     model.forward = model.forward_dummy
    # else:
    #     raise NotImplementedError(
    #         'FLOPs counter is currently not supported for {}'.format(
    #             model.__class__.__name__))

    flops, params = get_model_complexity_info(model, input_shape)
    split_line = '=' * 30
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\n{split_line}')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


if __name__ == '__main__':
    main()

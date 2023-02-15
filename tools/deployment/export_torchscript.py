# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import warnings

import numpy as np
import torch

from mmpose.apis import init_pose_model


def _convert_batchnorm(module):
    """Convert the syncBNs into normal BN3ds."""
    module_output = module
    if isinstance(module, torch.nn.SyncBatchNorm):
        module_output = torch.nn.BatchNorm3d(module.num_features, module.eps,
                                             module.momentum, module.affine,
                                             module.track_running_stats)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
            # keep requires_grad unchanged
            module_output.weight.requires_grad = module.weight.requires_grad
            module_output.bias.requires_grad = module.bias.requires_grad
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
    for name, child in module.named_children():
        module_output.add_module(name, _convert_batchnorm(child))
    del module
    return module_output


def export_torchscript(model,
                       input_shape,
                       show=False,
                       output_file='tmp.onnx',
                       verify=False):
    """Convert pytorch model to onnx model.

    Args:
        model (:obj:`nn.Module`): The pytorch model to be exported.
        input_shape (tuple[int]): The input tensor shape of the model.
        opset_version (int): Opset version of onnx used. Default: 11.
        show (bool): Determines whether to print the onnx model architecture.
            Default: False.
        output_file (str): Output onnx model name. Default: 'tmp.onnx'.
        verify (bool): Determines whether to verify the onnx model.
            Default: False.
    """
    model.eval()

    example = torch.randn(input_shape).cuda()

    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    traced_script_module = torch.jit.trace(model, example)

    if show:
        print(traced_script_module.graph)

    # Save the TorchScript model
    traced_script_module.save(output_file)

    print(f'Successfully exported TorchScript model: {output_file}')
    if verify:
        script_model = torch.jit.load(output_file)

        # get outputs
        pytorch_results = model(example)
        script_results = script_model(example)

        # check the numerical value
        assert np.allclose(
            pytorch_results.detach().cpu(), script_results.detach().cpu(), atol=1.e-5
        ), 'The outputs are different between Pytorch and TorchScript'
        print('The numerical values are same between Pytorch and TorchScript')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MMPose models to ONNX')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--show', action='store_true', help='show onnx graph')
    parser.add_argument('--output-file', type=str, default='tmp.onnx')
    parser.add_argument(
        '--verify',
        action='store_true',
        help='verify the onnx model output against pytorch output')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[1, 3, 256, 192],
        help='input size')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    assert args.opset_version == 11, 'MMPose only supports opset 11 now'

    # Following strings of text style are from colorama package
    bright_style, reset_style = '\x1b[1m', '\x1b[0m'
    red_text, blue_text = '\x1b[31m', '\x1b[34m'
    white_background = '\x1b[107m'

    msg = white_background + bright_style + red_text
    msg += 'DeprecationWarning: This tool will be deprecated in future. '
    msg += blue_text + 'Welcome to use the unified model deployment toolbox '
    msg += 'MMDeploy: https://github.com/open-mmlab/mmdeploy'
    msg += reset_style
    warnings.warn(msg)

    model = init_pose_model(args.config, args.checkpoint)
    model = _convert_batchnorm(model)

    # onnx.export does not support kwargs
    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'Please implement the forward method for exporting.')

    # convert model to onnx file
    export_torchscript(
        model,
        args.shape,
        opset_version=args.opset_version,
        show=args.show,
        output_file=args.output_file,
        verify=args.verify)

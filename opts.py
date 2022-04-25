import configargparse

from log import *


def parse_args():

    parser = configargparse.ArgParser()

    parser.add_argument('--height', type=int, default=128)
    parser.add_argument('--width', type=int, default=416)
    parser.add_argument('--seq-len', type=int, default=2)
    parser.add_argument('--num-scales', type=int, default=4)
    parser.add_argument('--bidirectional', action='store_true')
    parser.add_argument('--config-file', is_config_file=True)

    parser.add_argument('--dataset', default='kitti',
                        choices=['kitti', 'sintel', 'tartanair', 'waymo'])
    parser.add_argument(
        '--dataset-dir', default='/data/ra153646/datasets/KITTI/raw_data')
    parser.add_argument(
        '--train-file', default='/home/phd/ra153646/robustness/robustdepthflow/data/kitti/train.txt')
    parser.add_argument(
        '--val-file', default='/home/phd/ra153646/robustness/robustdepthflow/data/kitti/val.txt')

    parser.add_argument('--val-step-size', type=int, default=1)
    parser.add_argument('--data-parallel', action='store_true')

    # Optimization
    parser.add_argument('--optimizer', type=str, default='adam')

    parser.add_argument('-b', '--batch-size', type=int, default=12)
    parser.add_argument('-l', '--learning-rate', type=float, default=5e-5)
    parser.add_argument(
        '--scheduler',
        type=str,
        default='step',
        choices=[
            'step',
            'cosine',
            'coswr'])
    parser.add_argument('--scheduler-step-size', type=float, default=1e6)
    parser.add_argument('--clip-grad', type=float, default=0)

    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--weight-ssim', type=float, default=0)

    parser.add_argument('--depth-backbone', type=str, default='resnet')
    parser.add_argument('--depthnet-out', type=str, default='disp')

    parser.add_argument('--flow-backbone', type=str, default='resnet')
    parser.add_argument('--motion-burnin-iters', type=int, default=0)
    parser.add_argument('--init-iters', type=int, default=0)

    parser.add_argument('--weight-ds', type=float, default=1e-2)
    parser.add_argument('--ds-at-level', type=int,
                        default=-1)  # -1 = all levels
    parser.add_argument('--ds-alpha', type=int, default=1)

    parser.add_argument('--auto-mask', action='store_true')
    parser.add_argument('--rigid-mask', type=str, default='none')

    # The meaning of the threshold depends on the rigid mask method
    parser.add_argument('--rigid-mask-threshold', type=float, default=0.0)

    parser.add_argument('--motion-mode', type=str, default='multi_frame',
                        choices=['multi_frame', 'by_pair', 'by_image'])
    parser.add_argument('--fs-alpha', type=int, default=1)
    parser.add_argument('--weight-fs', type=float, default=1e-3)
    parser.add_argument('--flow-mask', type=str, default='none')

    parser.add_argument('--weight-ofc', type=float, default=0.0)

    parser.add_argument('--pose-layers', type=int, default=4)
    parser.add_argument('--pose-dp', type=float, default=0)
    parser.add_argument('--pose-bn', action='store_true')

    parser.add_argument('--weight-rigid', type=float, default=1.0)
    parser.add_argument('--weight-nonrigid', type=float, default=1.0)
    parser.add_argument('--merge-op', default='sum', choices=['sum', 'min'])
    parser.add_argument('--nonrigid-mode', default='opt',
                        choices=['opt', 'scene'])
    parser.add_argument('--stop-grads-rigid', action='store_true')

    parser.add_argument('--weight-dc', type=float, default=0)

    parser.add_argument('--weight-fc', type=float, default=0)
    parser.add_argument('--fc-detach', action='store_true')
    parser.add_argument('--fc-norm', type=str, default=None)
    parser.add_argument('--fc-diff', type=str, default='L1_unitsum')
    parser.add_argument('--fc-mode', type=str, default='enc',
                        choices=['enc', 'dec', 'all'])

    parser.add_argument('--weight-sc', type=float, default=0)

    parser.add_argument('--weight-rc', type=float, default=0)
    parser.add_argument('--weight-tc', type=float, default=0)
    parser.add_argument('--weight-msm', type=float, default=0)  # 0.25
    parser.add_argument('--weight-msp', type=float, default=0)  # 0.05

    parser.add_argument('--weight-ec', type=float, default=0)
    parser.add_argument('--ec-mode', type=str,
                        default='alg', choices=['alg', 'samp'])

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--ckp-freq', type=int, default=2)
    parser.add_argument('--log', type=str, default='sample-exp')
    parser.add_argument('--verbose', type=int, default=LOG_STANDARD)
    parser.add_argument('--load-model', type=str, default=None)

    parser.add_argument('--learn-intrinsics', action='store_true')

    # loss config
    parser.add_argument('--rep-cons', action='store_true')
    parser.add_argument('--softmin-beta', type=float,
                        default='inf')  # TODO: check this
    parser.add_argument('--loss-full-res', action='store_true')

    # Architecture config
    parser.add_argument('--norm', default='bn')
    parser.add_argument('--dec-dropout', type=float, default=0.0)

    parser.add_argument('--uncertainty', action='store_true')

    # l1, student, charbonnier, cauchy, general adaptive
    parser.add_argument('--loss', default='l1')

    parser.add_argument('--loss-params-type', default='none')
    parser.add_argument('--loss-params-lb', type=float,
                        default=1e-4)  # none, 'var', 'func'
    parser.add_argument('--loss-aux-weight', type=float, default=1.0)
    parser.add_argument('--loss-outliers-mode', type=str, default=None)
    parser.add_argument('--loss-outliers-qt', type=float, default=0.0)

    # Consistency regularization
    parser.add_argument('--downsample', type=str, default='bilinear')
    parser.add_argument('--weak-transform', type=str, default='wt_v0.2')
    parser.add_argument('--strong-transform', type=str, default='wt_v0.2')
    parser.add_argument('--weight-cr', type=float, default=0.0)

    parser.add_argument('--cr-num-calls', type=int, default=1)
    parser.add_argument('--cr-full-res', action='store_true')
    parser.add_argument('--cr-scale-depth', action='store_true')
    parser.add_argument('--cr-img-thresh', type=float, default=0.18)
    parser.add_argument('--cr-img-beta', type=float, default=0.0)
    parser.add_argument('--cr-pix-thresh', type=float, default=0.2)
    parser.add_argument('--cr-pix-beta', type=float, default=0.0)
    parser.add_argument('--cr-pix-recerr', action='store_true')
    parser.add_argument('--cr-error-fn', type=str, default='MSE')
    parser.add_argument('--cr-weight-egom', type=float,
                        default=0.0) 

    parser.add_argument('--cr-weight-gm', type=float,
                        default=0.0) 
    parser.add_argument('--cr-ramp-iters', type=int, default=0)

    parser.add_argument('--cr-mode', type=str, default='pl',
                        choices=['pl', 'ema', 'pi', 'swa'])
    parser.add_argument('--mt-ema', type=float, default=0.99)
    parser.add_argument('--cr-teacher-update', type=float,
                        default=0.0)  
    parser.add_argument('--cr-feats-noise', type=float, default=0)
    parser.add_argument('--cr-student-dp', type=float, default=0)

    # debug in/out using hooks
    parser.add_argument('--debug-model', action='store_true')
    # debug trainig process with a few iterations
    parser.add_argument('--debug-training', action='store_true')
    # debug trainig process with a few iterations
    parser.add_argument('--debug-valset', action='store_true')
    # log params and gradients
    parser.add_argument('--debug-metrics-train', action='store_true')
    # log params and gradients
    parser.add_argument('--debug-params', action='store_true')
    parser.add_argument('--debug-step', type=int, default=500)

    # Testing improvements
    parser.add_argument('--upscale-pred', action='store_true')
    parser.add_argument('--loss-noaug', action='store_true')

    # Evaluation
    parser.add_argument('--batchwise', action='store_true')
    parser.add_argument('--single-scalor', action='store_true')

    # TODO: add crop = ['eigen', 'dense']
    parser.add_argument('--crop-eigen', action='store_true')

    parser.add_argument('-c', '--checkpoint')
    parser.add_argument('-p', '--pred-file', default=None)

    parser.add_argument(
        '--test-file', default="data/kitti/test_files_eigen.txt")
    parser.add_argument('--gt-file', default=None)

    parser.add_argument('--min-depth', type=float,
                        default=1e-3, help="Threshold for minimum depth")
    parser.add_argument('--max-depth', type=float, default=80,
                        help="Threshold for maximum depth")

    # inference
    parser.add_argument('--input-dir', type=str)
    parser.add_argument('--output-dir', default='./out')
    parser.add_argument('--output-type', default='png', choices=['png', 'npz'])

    return parser.parse_args()

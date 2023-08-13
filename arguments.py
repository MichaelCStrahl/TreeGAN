import argparse


class Arguments:
    def __init__(self):
        self._parser = argparse.ArgumentParser(description='Arguments for TreeGAN.')

        # Dataset argument
        self._parser.add_argument('--dataset_path', type=str, default='/content/shapenetcore_partanno_segmentation_benchmark_v0', help='Dataset file path.')
        self._parser.add_argument('--class_choice', type=str, help='Select one class to generate. [plane, chair, ...] (default:all_class)')
        self._parser.add_argument('--batch_size', type=int, default=5, help='Integer value for batch size.')
        self._parser.add_argument('--point_num', type=int, default=2048, help='Integer value for number of points.')
        
        # Training argument
        self._parser.add_argument('--gpu', type=int, default=0, help='GPU number to use.')
        self._parser.add_argument('--epochs', type=int, default=1000, help='Integer value for epochs.')
        self._parser.add_argument('--lr', type=float, default=1e-4, help='Float value for learning rate.')
        self._parser.add_argument('--ckpt_path', type=str, default='/content/drive/MyDrive/ResultadosLothar/Checkpoints/', help='Checkpoint path.')
        self._parser.add_argument('--ckpt_save', type=str, default='tree_2class_ckpt_', help='Checkpoint name to save.')
        self._parser.add_argument('--ckpt_load', type=str, default='tree_2class_ckpt_630.pt', help='Checkpoint name to load.')
        self._parser.add_argument('--visdom_port', type=int, default=8097, help='Visdom port number. (default:8097)')
        self._parser.add_argument('--visdom_color', type=int, default=4, help='Number of colors for visdom pointcloud visualization. (default:4)')

        # Network argument
        self._parser.add_argument('--lambdaGP', type=int, default=10, help='Lambda for GP term.')
        self._parser.add_argument('--D_iter', type=int, default=5, help='Number of iterations for discriminator.')
        self._parser.add_argument('--support', type=int, default=10, help='Support value for TreeGCN loop term.')
        self._parser.add_argument('--DEGREE', type=int, default=[2,  2,  2,   2,   2,   64], nargs='+', help='Upsample degrees for generator.')
        self._parser.add_argument('--G_FEAT', type=int, default=[96, 64, 64,  64,  64,  64, 3], nargs='+', help='Features for generator.')
        self._parser.add_argument('--D_FEAT', type=int, default=[3,  64, 128, 256, 512, 1024], nargs='+', help='Features for discriminator.')

    def parser(self):
        return self._parser

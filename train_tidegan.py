from argparse import ArgumentParser
from training import train


def parse_arguments():
    parser = ArgumentParser(description='Train TideGAN')
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--name', type=str, default='tidegan', help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--model_size', type=int, default=64, help='Model size parameter used in WaveGAN')
    parser.add_argument('--ngpus', type=int, default=1, help='Number of GPUs to use for training')
    parser.add_argument('--latent_dim', type=int, default=100, help='Size of latent dimension used by generator')
    parser.add_argument('--lrelu_alpha', dest='alpha', type=float, default=0.2, help='Slope of negative part of LReLU used by discriminator')
    parser.add_argument('--phase_shuffle_batchwise', dest='batch_shuffle', action='store_true', help='If true, apply phase shuffle to entire batches rather than individual samples')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--phase_shuffle_shift_factor', dest='shift_factor', type=int, default=2, help='Maximum shift used by phase shuffle')
    parser.add_argument('--batches_per_epoch', type=int, default=10, help='Batches per training epoch')
    parser.add_argument('--post_proc_filt_len', type=int, default=512, help='Length of post processing filter used by generator. Set to 0 to disable.')
    parser.add_argument('--which_direction', type=str, default='AtoB', help='AtoB or BtoA')
    parser.add_argument('--display_freq', type=int, default=400, help='frequency of showing training results on screen')
    parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
    parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
    parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
    parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
    parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
    parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
    parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
    parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
    parser.add_argument('--gan-loss', default='wgan-wp', help='GAN loss type', choices=['gan', 'lsgan', 'wgan-wp'])
    parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
    parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
    parser.add_argument('--lambda_wp', type=float, default=10.0, help='weight for wgan-wp gradient penalty')
    parser.add_argument('--lambda_identity', type=float, default=0.5,
                        help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss.'
                        'For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
    parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
    parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau')
    parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
    parser.add_argument('audio_dir_A', help='Path to domain A audio samples')
    parser.add_argument('audio_dir_B', help='Path to domain B audio samples')
    parser.add_argument('checkpoints_dir', help='models and outputs are saved here')

    args = parser.parse_args()
    args.isTrain = True
    return args


if __name__ == '__main__':
    train(parse_arguments())

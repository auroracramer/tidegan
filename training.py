import os
import json
import time
import logging
from sample import get_all_audio_filepaths, create_data_split
from cyclegan import CycleGANModel
from utils import np_to_input_tensor, save_tidegan_samples

LOGGER = logging.getLogger('tidegan')
LOGGER.setLevel(logging.DEBUG)


def train(opt):
    audio_filepaths_A = get_all_audio_filepaths(opt.audio_dir_A)
    genA, valid_data_A, test_data_A = create_data_split(audio_filepaths_A, 0.1, 0.1, opt.batchSize, 64, 64)

    audio_filepaths_B = get_all_audio_filepaths(opt.audio_dir_B)
    genB, valid_data_B, test_data_B = create_data_split(audio_filepaths_B, 0.1, 0.1, opt.batchSize, 64, 64)


    model_dir = os.path.join(opt.checkpoints_dir, opt.name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    config_path = os.path.join(model_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(opt), f)

    model = CycleGANModel()
    model.initialize(opt)
    total_steps = 0
    use_cuda = opt.ngpus > 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for batch_idx in range(opt.batches_per_epoch):
            data_A = np_to_input_tensor(next(genA)['X'], use_cuda=use_cuda)
            data_B = np_to_input_tensor(next(genB)['X'], use_cuda=use_cuda)
            data = {'A': data_A, 'B': data_B, 'A_paths': [], 'B_paths': []}

            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(data)
            model.optimize_parameters()

            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                save_tidegan_samples(opt.checkpoints_dir, model.get_current_audibles(), total_steps)

            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize

            if total_steps % opt.save_latest_freq == 0:
                LOGGER.info('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save('latest')

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:
            LOGGER.info('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save('latest')
            model.save(epoch)

        LOGGER.info('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()

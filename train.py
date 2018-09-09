import time
from options.train_options import TrainOptions
from data import get_data_loaders
from models import create_model
from util.visualizer import Visualizer

if __name__ == '__main__':
    opt = TrainOptions().parse()
    visualizer = Visualizer(opt)
    logger = visualizer.logger

    data_loaders = get_data_loaders(opt, modes=['train', 'val'])
    dataset = data_loaders['train']
    dataset_size = len(dataset)
    fixed_real_imgs = next(iter(data_loaders['val']))

    model = create_model(opt)
    model.setup(opt)


    total_steps = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters(optimize_G=(i+1) % opt.train_D_repeat == 0)

            if total_steps % opt.log_freq == 0:
                losses = model.get_current_losses()
                t = (time.time() - iter_start_time) / opt.batch_size
                visualizer.log_current_losses(epoch, epoch_iter, losses, total_steps)

                model.set_input(fixed_real_imgs)
                if opt.model == 'star_gan':
                    model.forward(use_fixed_labels=True)
                else:
                    model.forward()
                visualizer.log_current_visuals(model.get_current_visuals(),
                                               epoch, total_steps)

            if total_steps % opt.save_latest_freq == 0:
                logger.info('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_networks('latest')

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:
            logger.info('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

            logger.info('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()

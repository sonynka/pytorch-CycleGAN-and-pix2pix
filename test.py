import os
from options.test_options import TestOptions
from data import get_data_loaders
from models import create_model
from util.visualizer import save_images
from util import html


if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display
    data_loaders = get_data_loaders(opt)
    dataset = data_loaders['test']
    dataset_size = len(dataset)
    model = create_model(opt)
    model.setup(opt)
    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    # test
    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()[0]
        if i % 5 == 0:
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(web_dir, visuals, str(i))

    webpage.save()

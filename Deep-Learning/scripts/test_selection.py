from tools import *
from visualization import *
from torchvision.utils import save_image
from tqdm import tqdm

import sys
sys.path.append('..')

torch.set_printoptions(profile="full")
torch.set_printoptions(linewidth=10000)
torch.set_printoptions(sci_mode=False)


if __name__ == '__main__':
    path_model = "C://Users//gaubi//Documents//CDocuments//Cours PC local//MVA_local//S1//Deep Learning//Projet//runs//alto_word//baseline//model.pth"
    data_path = os.path.join(PROJECT_PATH,"datasets/Alto/alto_word_humanistique")
    # dataset_args = {'sep':' ', 'space':'_'}
    # dataloader = load_data(data_path,split = 'train', **dataset_args)
    dataloader = load_data(data_path,split = 'train')

    #batch = get_val_images(data_path,20, padding_mode='edge')

    for i, batch in enumerate(dataloader):
        print(i)
    # batch = next(iter(load_data(data_path,split = 'test', dataset_size=20, batch_size=20)))

    # alphabet_occurences = get_alphabet(dataloader)
    # trained_model =  load_supervised_model_trained(path_model, set(alphabet_occurences.keys()))
    

    # output = inference_supervised_model(trained_model, batch)
    # segmentation = segmentation_trained_model(trained_model, batch)
    # segmentation_grid = make_segmentation_grid(batch['x'], output['reconstructed'], segmentation)

    # for i in range(len(segmentation)):
    #     seg_i = torch.cat((batch['x'][i],output['reconstructed'][i],segmentation[i]), dim=1)
    #     save_image(seg_i, os.path.join(PROJECT_PATH,'outputs/preds/{}.png'.format(i)))
    #     save_image(batch['x'][i], os.path.join(PROJECT_PATH,'outputs/preds/img_{}.png'.format(i)))
    #     save_image(output['reconstructed'][i], os.path.join(PROJECT_PATH,'outputs/preds/reco_{}.png'.format(i)))
    #     save_image(segmentation[i], os.path.join(PROJECT_PATH,'outputs/preds/seg_{}.png'.format(i)))

    # save_image(output['reconstructed'], os.path.join(PROJECT_PATH,'outputs/preds/reconstructed.png'))
    # save_image(segmentation, os.path.join(PROJECT_PATH,'outputs/preds/segmentation.png'))
    # save_image(segmentation_grid, os.path.join(PROJECT_PATH,'outputs/preds/segmentation_grid.png'))





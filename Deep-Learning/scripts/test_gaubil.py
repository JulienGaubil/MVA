# -*- coding: utf-8 -*-
import sys, hydra, os, os.path
sys.path.append('..')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from omegaconf import open_dict

from learnable_typewriter.utils.defaults import PROJECT_PATH, RUNS_PATH
from learnable_typewriter.trainer import Trainer

from tools import load_transformation
from learnable_typewriter.typewriter.utils import count_parameters




@hydra.main(config_path=PROJECT_PATH/"configs/", config_name="supervised-google.yaml")
def test_supervised_trainer_google(config):
    seed = 4321
    #Change les parametres nécessaires dans les configs
    with open_dict(config):
        #tag of experiment (tensorboard)
        config.run_dir = os.path.join(RUNS_PATH, 'DL', 'google', "supervised/MarioNette/blank_sprite/")  #"test" remplaxe le tag        

        #training
        # config.training.supervised = True
        config.training.batch_size = 32 #batch size pour l'entraînement ligne entière
        config.training.num_epochs = 1 #change le nombre d'epochs, si 1 évaluation uniquement
        config.training.device = 0
        # config.dataset.N_min = 0
        # config.dataset.W_max = 1650
        # config.training.flush_mem = True
        # config.training.flush_per = 30

        #sprites - background
        # config.model.background.init.constant = 0.7
        # config.model.background.init.freeze = True

        #loss - penalizations
        # config.model.loss.ctc_factor = 0.1
        # config.model.loss.gamma_penalization = 0  #enleve la pénalisation pour l'overlapping
        # config.model.loss.gaussian_weighting = False  #enleve la pondération gaussienne pour la loss de reconstruction, n'est pas utile en ligne entieres
        # config.model.loss.beta_sparse = 0  #penalizes the number of blanks used
        # config.model.loss.beta_reg = 0  #pushes the weights to be either 0 or 1

        #others
        # config.model.transformation.canvas_size = [config.model.encoder.H, config.dataset.crop_width]
        # config.training.log.train.images.every=1
        # config.training.log.val.reconstruction.every=2  #transformations logged every 2 iters of validation
        # config.reassignment.milestone.num_iters = 2000  #validation is performed every 2000 iters
    
    trainer = Trainer(config, seed=seed)
    trainer.run(seed=seed)    

# test_supervised_trainer_google()



@hydra.main(config_path=PROJECT_PATH/"configs/", config_name="Volume0002_words.yaml")
def test_supervised_trainer_google_words(config):
    seed = 4321
    #Change les parametres nécessaires dans les configs
    with open_dict(config):
        #tag of experiment (tensorboard)
        config.run_dir = os.path.join(RUNS_PATH, config["dataset"]["alias"], "supervised/MarioNette/blank_sprite/test")  #"test" remplaxe le tag        
        
        #training
        config.training.supervised = True
        config.training.batch_size = 16 #batch size pour l'entraînement ligne entière
        config.training.num_epochs = 50 #change le nombre d'epochs, si 1 évaluation uniquement
        config.training.device = 0
        config.dataset.N_min = 0
        config.dataset.W_max = float('inf')

        #sprites - background
        config.model.background.init.constant = 0.7
        config.model.background.init.freeze = True

        #loss - penalizations
        config.model.loss.ctc_factor = 0.1
        config.model.loss.gamma_penalization = 0  #enleve la pénalisation pour l'overlapping
        config.model.loss.gaussian_weighting = False  #enleve la pondération gaussienne pour la loss de reconstruction, n'est pas utile en ligne entieres
        config.model.loss.beta_sparse = 0  #penalizes the number of blanks used
        config.model.loss.beta_reg = 0  #pushes the weights to be either 0 or 1

        #others
        config.model.transformation.canvas_size = [config.model.encoder.H, config.dataset.crop_width]
        config.training.log.train.images.every=1
        config.training.log.val.reconstruction.every=2  #transformations logged every 2 iters of validation
        config.reassignment.milestone.num_iters = 500  #validation is performed every 2000 iters

    # trainer = TrainerSupervised(config, seed=seed)
    trainer = Trainer(config, seed=seed)
    trainer.run(seed=seed)
        


# test_supervised_trainer_google_words()



@hydra.main(config_path=PROJECT_PATH/"configs/", config_name="test_gaubil_alto.yaml")
def test_supervised_trainer_alto(config):
    seed = 4321
    #Change les parametres nécessaires dans les configs
    with open_dict(config):
        #dataset
        # config.dataset.path = "Alto/alto_word_humanistique"
        # config.dataset.alias = "alto_word_humanistique"

        #tag of experiment (tensorboard)
        config.run_dir = os.path.join(RUNS_PATH, 'DL', 'alto_word_humanistique', "morpho/test")  #"test" remplaxe le tag        
        
        #training
        config.training.supervised = True
        config.training.batch_size = 16 #batch size
        config.training.num_epochs = 1 #change le nombre d'epochs, si 1 évaluation uniquement
        config.training.device = 0
        # config.training['pretrained'] = 'runs/DL/alto_word/tsf_tps/model_baseline.pth'
        # config.dataset.N_min=0
        # config.dataset.W_max=float('inf')
        

        #sprites - background
        config.model.background.init.constant = 0.7
        # config.model.background.init.freeze = True

        #loss - penalizations
        # config.model.loss.ctc_factor = 0
        # config.model.loss.gamma_penalization = 0 #enleve la pénalisation pour l'overlapping
        # config.model.loss.gaussian_weighting = True  #enleve la pondération gaussienne pour la loss de reconstruction, n'est pas utile en ligne entieres
        # config.model.loss.beta_sparse = 0  #penalizes the number of blanks used
        # config.model.loss.beta_reg = 0  #pushes the weights to be either 0 or 1
        config.model.transformation.layer.ops = ['color', 'position','morpho']

        #others
        # config.model.transformation.canvas_size = [config.model.encoder.H, config.dataset.crop_width]
        # config.training.log.train.images.every=1
        # config.training.log.val.reconstruction.every=2  #transformations logged every 2 iters of validation
        # config.reassignment.milestone.num_iters = 1000  #logging images is performed every 2000 iters
    
    
    trainer= Trainer(config, seed=seed)
    # trainer.load_from_dir(model_path='runs/DL/alto_word_humanistique/baseline/model.pth')
    trainer.run(seed=seed)    


test_supervised_trainer_alto()



@hydra.main(config_path=PROJECT_PATH/"configs/", config_name="test_gaubil_copiale.yaml")
def test_supervised_trainer_copiale(config):
    seed = 4321
    #Change les parametres nécessaires dans les configs
    with open_dict(config):
        #tag of experiment (tensorboard)
        config.run_dir = os.path.join(RUNS_PATH, 'DL', 'alto_lines', "supervised/MarioNette/blank_sprite/")  #"test" remplaxe le tag        
        
        #training
        # config.training.supervised = True
        config.training.batch_size = 8 #batch size pour l'entraînement ligne entière
        config.training.num_epochs = 400 #change le nombre d'epochs, si 1 évaluation uniquement
        config.training.device = 1
        # config.dataset.N_min=0
        # config.dataset.W_max=float('inf')
        # config.training.flush_mem = True
        # config.training.flush_per = 15
        # config['trainer']={'mode': 'path-select'}

        #sprites - background
        # config.model.background.init.constant = 0.7
        # config.model.background.init.freeze = True

        #loss - penalizations
        # config.model.loss.ctc_factor = 0.1
        # config.model.loss.gamma_penalization = 0  #enleve la pénalisation pour l'overlapping
        # config.model.loss.gaussian_weighting = False  #enleve la pondération gaussienne pour la loss de reconstruction, n'est pas utile en ligne entieres
        # config.model.loss.beta_sparse = 0  #penalizes the number of blanks used
        # config.model.loss.beta_reg = 0  #pushes the weights to be either 0 or 1

        #others
        # config.model.transformation.canvas_size = [config.model.encoder.H, config.dataset.crop_width]
        # config.training.log.train.images.every=1
        # config.training.log.val.reconstruction.every=2  #transformations logged every 2 iters of validation
        # config.reassignment.milestone.num_iters = 89 #logging images is performed every 2000 iters

    trainer = Trainer(config, seed=seed)
    trainer.run(seed=seed)    


# test_supervised_trainer_copiale()




@hydra.main(config_path=PROJECT_PATH/"configs/", config_name="test_gaubil_alto_lines.yaml")
def test_supervised_trainer_alto_lines(config):
    seed = 4321
    #Change les parametres nécessaires dans les configs
    with open_dict(config):
        #tag of experiment (tensorboard)
        config.run_dir = os.path.join(RUNS_PATH, 'DL', 'alto_lines', "2_sprites")  #"test" remplaxe le tag        

        #training
        # config.training.supervised = True
        config.training.batch_size = 8 #batch size pour l'entraînement ligne entière
        config.training.num_epochs = 400  #change le nombre d'epochs, si 1 évaluation uniquement
        config.training.device = 0
        config.dataset.N_min=0
        config.dataset.W_max=1650
        config.training.flush_mem = True
        config.training.flush_per = 15

        #sprites - background
        # config.model.background.init.constant = 0.7
        # config.model.background.init.freeze = True

        #loss - penalizations
        # config.model.loss.ctc_factor = 0.1
        # config.model.loss.gamma_penalization = 0  #enleve la pénalisation pour l'overlapping
        # config.model.loss.gaussian_weighting = False  #enleve la pondération gaussienne pour la loss de reconstruction, n'est pas utile en ligne entieres
        # config.model.loss.beta_sparse = 0  #penalizes the number of blanks used
        # config.model.loss.beta_reg = 0  #pushes the weights to be either 0 or 1

        #others
        # config.model.transformation.canvas_size = [config.model.encoder.H, config.dataset.crop_width]
        # config.training.log.train.images.every=1
        # config.training.log.val.reconstruction.every=2  #transformations logged every 2 iters of validation
        # config.reassignment.milestone.num_iters = 500  #logging images is performed every 2000 iters

    # trainer = TrainerSupervised(config, seed=seed)
    trainer = Trainer(config, seed=seed)
    trainer.run(seed=seed)    


# test_supervised_trainer_alto_lines()




@hydra.main(config_path=PROJECT_PATH/"configs/", config_name="test_gaubil_fontenay.yaml")
def test_supervised_trainer_fontenay(config):
    seed = 4321
    #Change les parametres nécessaires dans les configs
    with open_dict(config):
        #tag of experiment (tensorboard)
        config.run_dir = os.path.join(RUNS_PATH, config["dataset"]["alias"], "supervised/MarioNette/blank_sprite/test/ctc_-2")  #"test" remplaxe le tag        
        
        #training
        config.training.supervised = True
        config.training.batch_size = 4 #batch size pour l'entraînement ligne entière
        config.training.num_epochs = 1600 #change le nombre d'epochs, si 1 évaluation uniquement
        config.training.device = 2
        config.dataset.N_min=0
        config.dataset.W_max=float("inf")
        config.training.flush_mem = True
        config.training.flush_per = 3
        # config['trainer']={'mode': 'path-select'}

        #sprites - background
        config.model.background.init.constant = 0.7
        config.model.background.init.freeze = True

        #loss - penalizations
        config.model.loss.ctc_factor = 0.01
        config.model.loss.gamma_penalization = 0  #enleve la pénalisation pour l'overlapping
        config.model.loss.gaussian_weighting = False  #enleve la pondération gaussienne pour la loss de reconstruction, n'est pas utile en ligne entieres
        config.model.loss.beta_sparse = 0  #penalizes the number of blanks used
        config.model.loss.beta_reg = 0  #pushes the weights to be either 0 or 1

        #others
        config.model.transformation.canvas_size = [config.model.encoder.H, config.dataset.crop_width]
        config.training.log.train.images.every=1
        config.training.log.val.reconstruction.every=2  #transformations logged every 2 iters of validation
        config.reassignment.milestone.num_iters = 200 #logging images is performed every 2000 iters

    ctc_fs = [-2,-1]
    for ctc_f in ctc_fs:
        with open_dict(config):
            config.run_dir = os.path.join(RUNS_PATH, config["dataset"]["alias"], "supervised/MarioNette/blank_sprite/test/ctc_"+str(ctc_f))  #"test" remplaxe le tag        
            config.model.loss.ctc_factor = 10**ctc_f

        trainer = Trainer(config, seed=seed)
        trainer.run(seed=seed)

    # trainer = Trainer(config, seed=seed)
    # from learnable_typewriter.trainer import TrainerSupervised
    # trainer = TrainerSupervised(config, seed=seed)

    # trainer.run(seed=seed)    


# test_supervised_trainer_fontenay()
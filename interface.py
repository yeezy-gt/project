from model import ResNet as TheModel
from train import main as the_trainer
from predict import main as the_predictor
from dataset import IrisDataset as TheDataset
from dataset import get_train_loader as the_dataloader
from config import DATASET_CONFIG
from config import TRAIN_CONFIG

the_batch_size = DATASET_CONFIG['batch_size']
total_epochs = TRAIN_CONFIG['num_epochs']

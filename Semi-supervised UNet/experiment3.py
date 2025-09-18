import os
import warnings

#warnings.filterwarnings("ignore")
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"
#os.getcwd()

from main3 import *
from utils3 import *
from model3 import *

def experiments(case):

    image_path = r'data/all_img'
    masks_path = r'data/all_masks'
    labeled_filename = r'csv/labeled_data.csv'
    unlabeled_filename = r'csv/unlabeled_data.csv'
    vol1_path = r'data/23w_equal'
    vol2_path = r'data/23w_train_equal'
    test_filename = r'csv/unlabeled_test_data.csv'
    #vol1_path_test = r'data/21w'
    #vol2_path_test = r'data/22w'
    vol1_path_test = r'data/24w'
    vol2_path_test = r'data/25w'

    key_values = case_dictionary[case]
    training_mode = key_values[0]
    alpha = key_values[1]
    pretraining = key_values[2]
    out_model = key_values[3]

    figpath = f'/home/antonio/workspace/ucl-thesis/loss/{case}.png'

    train_df, train_set, train_loader, valid_set, valid_loader, test_set, train_unlabeled_df, train_unlabeled_set, train_unlabeled_loader, valid_unlabeled_set, valid_unlabeled_loader, test_unlabeled_set = prepare_data(image_path, masks_path, labeled_filename, unlabeled_filename, vol1_path, vol2_path, test_filename, vol1_path_test, vol2_path_test) #vol3_path_test, vol4_path_test

    if training_mode == 'train':

        if pretraining == 'finetune':
            input_model = 'initial_weights.pt'
            print(f'Loaded {input_model} model')
            output_model = out_model
        else:
            input_model = None
            print('No model loaded')
            output_model = out_model
        
        saved_model = train(train_loader, valid_loader, train_unlabeled_df, valid_unlabeled_loader, alpha, input_model, output_model, figpath)
    else:
        saved_model = out_model
        print(f'Loading existing model: {saved_model}\n')

    train_set = SegmentationDataset(train_df, get_valid_augs())

    train_iou, valid_iou, test_iou, train_unlabeled_iou, valid_unlabeled_iou, test_unlabeled_iou = evaluate(train_set, valid_set, test_set, train_unlabeled_set, valid_unlabeled_set, test_unlabeled_set, saved_model)
    
    print(f'IOUs = train labeled: {train_iou}, valid labeled: {valid_iou}, test labeled: {test_iou}, train unlabeled: {train_unlabeled_iou}, valid unlabeled: {valid_unlabeled_iou}, test unlabeled: {test_unlabeled_iou}')

    return saved_model

case_dictionary = {
    #'initial_weights-iou': ['iou', 0.0, 'iou', 'initial_weights.pt'],
    'case-3_alpha-0.5_epochs-150_lr-0.008_batch-size-8_gamma-0.5_step-size-50': ['train', 0.5, 'finetune', 'case-3_alpha-0.5_epochs-150_lr-0.008_batch-size-8_gamma-0.5_step-size-50.pt'],
    
    #'case-3_alpha-0.5_epochs-150_lr-0.003_batch-size-8_no-scheduler': ['train', 0.5, 'finetune', 'case-3_alpha-0.5_epochs-150_lr-0.003_batch-size-8_no-scheduler.pt'],
    #'case-3_alpha-0.5_epochs-150_lr-0.003_batch-size-8_gamma-0.5_step-size-75': ['train', 0.5, 'finetune', 'case-3_alpha-0.5_epochs-150_lr-0.003_batch-size-8_gamma-0.5_step-size-75.pt'],
    #'case-3_alpha-0.5_epochs-150_lr-0.003_batch-size-8_gamma-0.9_step-size-25': ['train', 0.5, 'finetune', 'case-3_alpha-0.5_epochs-150_lr-0.003_batch-size-8_gamma-0.9_step-size-25.pt'],
    #'case-3_alpha-0.5_epochs-150_lr-0.003_batch-size-8_gamma-0.7_step-size-50': ['train', 0.5, 'finetune', 'case-3_alpha-0.5_epochs-150_lr-0.003_batch-size-8_gamma-0.7_step-size-50.pt'],
    #'case-3_alpha-0.5_epochs-150_lr-0.008_batch-size-8_gamma-0.9_step-size-15': ['train', 0.5, 'finetune', 'case-3_alpha-0.5_epochs-150_lr-0.008_batch-size-8_gamma-0.9_step-size-15.pt'],
    #'case-3_alpha-0.5_epochs-150_lr-0.003_batch-size-8_max-lr-0.01': ['train', 0.5, 'finetune', 'case-3_alpha-0.5_epochs-150_lr-0.003_batch-size-8_max-lr-0.01.pt'],
    #'case-3_alpha-0.5_epochs-150_lr-0.001_batch-size-8_max-lr-0.008': ['train', 0.5, 'finetune', 'case-3_alpha-0.5_epochs-150_lr-0.001_batch-size-8_max-lr-0.008.pt'],
    #'case-3_alpha-0.5_epochs-300_lr-0.008_batch-size-8_gamma-0.5_step-size-75': ['train', 0.5, 'finetune', 'case-3_alpha-0.5_epochs-300_lr-0.008_batch-size-8_gamma-0.5_step-size-75.pt'],
    #'case-3_alpha-0.5_epochs-200_lr-0.006_batch-size-8_gamma-0.7_step-size-40': ['train', 0.5, 'finetune', 'case-3_alpha-0.5_epochs-200_lr-0.006_batch-size-8_gamma-0.7_step-size-40.pt'],
    #'case-3_alpha-0.5_epochs-150_lr-0.001_batch-size-8_no-scheduler': ['train', 0.5, 'finetune', 'case-3_alpha-0.5_epochs-150_lr-0.001_batch-size-8_no-scheduler.pt'],
    #'case-3_alpha-0.5_epochs-200_lr-0.003_batch-size-8_gamma-0.85_step-size-20': ['train', 0.5, 'finetune', 'case-3_alpha-0.5_epochs-200_lr-0.006_batch-size-8_gamma-0.85_step-size-20.pt'],
    #'case-3_alpha-0.5_epochs-200_lr-0.001_batch-size-8_no-scheduler': ['train', 0.5, 'finetune', 'case-3_alpha-0.5_epochs-200_lr-0.001_batch-size-8_no-scheduler.pt'],
    #'case-3_alpha-0.5_epochs-300_lr-0.006_batch-size-8_power-1': ['train', 0.5, 'finetune', 'case-3_alpha-0.5_epochs-300_lr-0.006_batch-size-8_power-1.pt'],
    #'case-3_alpha-0.5_epochs-200_lr-0.008_batch-size-8_gamma-0.9': ['train', 0.5, 'finetune', 'case-3_alpha-0.5_epochs-200_lr-0.008_batch-size-8_gamma-0.9.pt'],
    #'case-3_alpha-0.5_epochs-200_lr-0.003_batch-size-8_t-max-200_eta-min-1e-3': ['train', 0.5, 'finetune', 'case-3_alpha-0.5_epochs-200_lr-0.003_batch-size-8_t-max-200_eta-min-1e-3.pt'],
    #'case-3_alpha-0.5_epochs-200_lr-0.008_batch-size-8_gamma-0.5_milestones-40-80-120': ['train', 0.5, 'finetune', 'case-3_alpha-0.5_epochs-200_lr-0.008_batch-size-8_gamma-0.5_milestones-40-80-120.pt'],
}

for case in case_dictionary:

    experiments(case)

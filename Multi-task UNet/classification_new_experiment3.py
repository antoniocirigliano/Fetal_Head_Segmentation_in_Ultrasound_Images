import os
import warnings

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.getcwd()

from classification_new_main3 import *
from classification_new_utils3 import *
from classification_new_model3 import *

def experiments(case):

    image_path = r'data/all_img'
    masks_path = r'data/all_masks'
    labeled_filename = r'csv/labeled_data.csv'
    abdomen_femur_path = r'data/img_abdomen_femur'
    abdomen_femur_masks_path = r'data/masks_abdomen_femur'
    abdome_femur_filename = r'csv/abdome_femur.csv'
    images_black_border_path = r'data/all_img_black_border'
    masks_black_border_path = r'data/all_masks_black_border'
    labeled_black_border_filename = r'csv/labeled_data_black_border.csv'
    unlabeled_filename = r'csv/unlabeled_data.csv'
    vol1_path = r'data/23w_equal'
    vol2_path = r'data/23w_train_equal'
    test_filename = r'csv/unlabeled_test_data.csv'
    vol1_path_test = r'data/24w'
    vol2_path_test = r'data/25w'

    key_values = case_dictionary[case]
    training_mode = key_values[0]
    alpha = key_values[1]
    pretraining = key_values[2]
    out_model = key_values[3]
    beta = key_values[4]

    figpath = f'/home/antonio/workspace/ucl-thesis/loss/{case}.png'

    train_df, train_set, train_loader, valid_set, valid_loader, test_set, train_unlabeled_df, train_unlabeled_set, train_unlabeled_loader, valid_unlabeled_set, valid_unlabeled_loader, test_unlabeled_set, test_head_set, test_abdomen_femur_set, test_black_border_set = prepare_data(image_path, masks_path, labeled_filename, abdomen_femur_path, abdomen_femur_masks_path, abdome_femur_filename, images_black_border_path, masks_black_border_path, labeled_black_border_filename, unlabeled_filename, vol1_path, vol2_path, test_filename, vol1_path_test, vol2_path_test)

    if training_mode == 'train':

        if pretraining == 'finetune':
            input_model = 'case-3_alpha-0.5_epochs-150_lr-0.008_batch-size-8_gamma-0.5_step-size-50.pt'
            print(f'Loaded {input_model} model')
            output_model = out_model
        else:
            input_model = None
            print('No model loaded')
            output_model = out_model
        
        saved_model = train(train_loader, valid_loader, train_unlabeled_df, valid_unlabeled_loader, alpha, beta, input_model, output_model, figpath)
    else:
        saved_model = out_model
        print(f'Loading existing model: {saved_model}\n')

    train_set = SegmentationDataset(train_df, get_valid_augs())

    train_iou, valid_iou, test_iou, train_unlabeled_iou, valid_unlabeled_iou, test_unlabeled_iou, test_head_iou, test_black_border_iou, test_abdomen_femur_iou = evaluate(train_set, valid_set, test_set, train_unlabeled_set, valid_unlabeled_set, test_unlabeled_set, test_head_set, test_abdomen_femur_set, test_black_border_set, saved_model)

    return saved_model

case_dictionary = {
    'classification-new-case-3_abdomen-femur-0.25_alpha-0.5_beta_1.0_epochs-150_lr-0.00003_batch-size-8_no-scheduler': ['train', 0.5, 'finetune', 'classification-new-case-3_abdomen-femur-0.25_alpha-0.5_beta_1.0_epochs-150_lr-0.00003_batch-size-8_no-scheduler.pt', 1.0],
}

for case in case_dictionary:
    
    experiments(case)

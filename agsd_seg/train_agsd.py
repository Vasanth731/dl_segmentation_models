# classic imports
import cv2
import argparse
import torch.optim as optim
from skimage import morphology
from tqdm import tqdm
from tqdm import trange
from datetime import datetime 

# imports from own libraries
from utils import *
from dataset import *
from model import MyUNet
from feature_extraction import FeatureExtraction

# wandb login 
import wandb
wandb.login()
user = "vasanth-ambrose"
project = "segmentation"
display_name = "loss-visual"
wandb.init(entity=user, project=project, name=display_name)

# hyperparameters
# naming = "check_forming" # just to make directory 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 0.0005
SIM_FEATURE_CNN = "vgg16"
SIM_FEATURE_LAYERS = ["relu5_3"]
EPOCH_NUM = 100
START_EPOCH = 0
pos_ratio = 1.5
neg_ratio = 1
fg_ratio = 0.5
bg_ratio = 2
SIM_FG_MARGINS = [0.05] 
SIM_BG_MARGINS = [0.45] 
FEATURE_MAP_SIZE =  256,320
BATCH_SIZE = 4
log_step_size = 16


ORIGINAL_IMG_SIZE = [1080,1920]
PREPROCESS_CROP_SIZE = [1024,1280]
PREPROCESS_CROP_LOCATION = [28,320]
MODEL_INPUT_SIZE = [512,640]

IMG_DIR = "./data_endovis17/image"
# GROUND_TRUTH_DIR = "./data_endovis17/ground_truth"
POS_ANCHOR_DIR = "./data_endovis17/anchors/pos"
NEG_ANCHOR_DIR = "./data_endovis17/anchors/neg"
TRAIN_DIR_LIST = ["instrument_dataset_1_train"]
# pos_prob_dir = "./endovis17-SS-full/pos_prob-0/instrument_dataset_1_train"
# neg_prob_dir = "./endovis17-SS-full/neg_prob-0/instrument_dataset_1_train"
# pos_mask_dir = "./endovis17-SS-full/pos_mask-0/instrument_dataset_1_train"
# neg_mask_dir = "./endovis17-SS-full/neg_mask-0/instrument_dataset_1_train"


# few needed functions 
def train_log(loss,step_size):    
    print(f' loss {loss} step {step_size}')
    wandb.log({"Loss": loss},step=step_size)

def save_checkpoint(state,epoch,model_name):
    date=datetime.date(datetime.now())
    time=datetime.time(datetime.now())
    date_time=str(date)+str("__")+str(time)
    date_time=date_time[0:20]

    # '/home/htic/Downloads/agsd_git_repo/segmentation_checkpoint'
    filename = f'/home/endodl/codes_vasanth/agsd_git_repo/segmentation_checkpoint/SEG_{model_name}.pth'

    print("=> Saving checkpoint")
    torch.save(state, filename)


def train_one_epoch(train_train_loader, model,currentepoch):
        

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    feature_extractor = FeatureExtraction(
        feature_extraction_cnn=SIM_FEATURE_CNN, 
        normalization=True, last_layer=','.join(SIM_FEATURE_LAYERS))

    feature_extractor.eval()

    print(f'EPOCH : {currentepoch}')

    model.train()

        # anchor loss
    for batch_idx, data in enumerate(tqdm(train_train_loader)):

        img = data['model_img'].to(device)
        pos_anchor = data['model_pos_anchor'].to(device).permute(0, 1, 3, 4, 2)
        neg_anchor = data['model_neg_anchor'].to(device).permute(0, 1, 3, 4, 2)
        
        optimizer.zero_grad()
        
        img_1, img_2 = img[:,0,:,:,:], img[:,1,:,:,:]
        pos_anchor_1, pos_anchor_2 = pos_anchor[:,0,:,:,:], pos_anchor[:,1,:,:,:] # channel 1 , channel 2
        neg_anchor_1, neg_anchor_2 = neg_anchor[:,0,:,:,:], neg_anchor[:,1,:,:,:]

        logits_1 = model(img_1) # pred for frame t
        logits_2 = model(img_2) # pred for frame t+1

        anchor_loss_1, anchor_loss_details_1 = get_anchor_loss(logits_1, -logits_1, pos_anchor_1, neg_anchor_1, pos_ratio, neg_ratio) # for frame t

        anchor_loss_2, anchor_loss_details_2 = get_anchor_loss(logits_2, -logits_2, pos_anchor_2, neg_anchor_2, pos_ratio, neg_ratio) # for frame t+1

        anchor_loss = anchor_loss_1 + anchor_loss_2

        anchor_loss_details = {k: anchor_loss_details_1[k] + anchor_loss_details_2[k]
                for k in anchor_loss_details_1.keys()}
        

        # diffusion loss
        with torch.no_grad():
            feature_maps_1 = feature_extractor(img_1) # for frame t
            feature_maps_2 = feature_extractor(img_2) # for frame t+1

        diffusion_loss = {}

        diffusion_loss_details = {}

        for i, key in enumerate(SIM_FEATURE_LAYERS):

            feature_maps_1[i] = F.interpolate(feature_maps_1[i], size=FEATURE_MAP_SIZE, mode='bicubic', align_corners=True)

            feature_maps_2[i] = F.interpolate(feature_maps_2[i], size=FEATURE_MAP_SIZE, mode='bicubic', align_corners=True)

            _diff_loss, _diff_details = get_diffusion_loss(
                feature_maps_1[i], feature_maps_2[i], logits_1, logits_2,
                fg_margin=SIM_FG_MARGINS[i], bg_margin=SIM_BG_MARGINS[i],
                fg_ratio=fg_ratio, bg_ratio=bg_ratio,
                naming='sim_{}'.format(key))

            diffusion_loss[key] = _diff_loss
            diffusion_loss_details[key] = _diff_details

        train_loss = anchor_loss

        for key in SIM_FEATURE_LAYERS:
            train_loss += diffusion_loss[key]

        train_loss.backward()

        optimizer.step()
        
        if batch_idx%log_step_size==0:
            print(f'EPOCH : {currentepoch}')
            print(f'loss  : {train_loss}')

            # Plot(loss,(batch_idx*log_step_size)) 
            if currentepoch==0:
                train_log(train_loss.item(),(batch_idx*log_step_size))
            else:
                train_log(train_loss.item(),((batch_idx*log_step_size)+(len(train_train_loader)*log_step_size*currentepoch)))
        
    print(f'loss  : {train_loss}')
    return train_loss, model
            
#             if batch_idx%log_step_size==0:
#                 print(f'EPOCH : {currentepoch}')
#                 print(f'loss  : {total_loss}')

#                 # Plot(loss,(batch_idx*log_step_size)) 
#                 if currentepoch==0:
#                     train_log(loss.item(),(batch_idx*log_step_size))
#                 else:
#                     train_log(loss.item(),((batch_idx*log_step_size)+(len(train_loader)*log_step_size*currentepoch)))
                
#             loop.set_postfix(loss=loss.item())

def val_one_epoch(val_loader, model,currentepoch):
    
#     model = MyUNet(transposed_conv = False, align_corners = False)
#     model.to(device)

    model = model
    model.eval()
    
    
    feature_extractor = FeatureExtraction(
        feature_extraction_cnn=SIM_FEATURE_CNN, 
        normalization=True, last_layer=','.join(SIM_FEATURE_LAYERS))

    feature_extractor.eval()


    print(f'EPOCH : {currentepoch}')


    losses = []
    with torch.no_grad():
    
        # anchor loss
        for _, data in enumerate(tqdm(val_loader)):

            img = data['model_img'].to(device)
            pos_anchor = data['model_pos_anchor'].to(device).permute(0, 1, 3, 4, 2)
            neg_anchor = data['model_neg_anchor'].to(device).permute(0, 1, 3, 4, 2)



            img_1, img_2 = img[:,0,:,:,:], img[:,1,:,:,:]
            pos_anchor_1, pos_anchor_2 = pos_anchor[:,0,:,:,:], pos_anchor[:,1,:,:,:] # channel 1 , channel 2
            neg_anchor_1, neg_anchor_2 = neg_anchor[:,0,:,:,:], neg_anchor[:,1,:,:,:]

            logits_1 = model(img_1) # pred for frame t
            logits_2 = model(img_2) # pred for frame t+1

            anchor_loss_1, anchor_loss_details_1 = get_anchor_loss(logits_1, -logits_1, pos_anchor_1, neg_anchor_1, pos_ratio, neg_ratio) # for frame t

            anchor_loss_2, anchor_loss_details_2 = get_anchor_loss(logits_2, -logits_2, pos_anchor_2, neg_anchor_2, pos_ratio, neg_ratio) # for frame t+1

            anchor_loss = anchor_loss_1 + anchor_loss_2

            anchor_loss_details = {k: anchor_loss_details_1[k] + anchor_loss_details_2[k]
                for k in anchor_loss_details_1.keys()}



            # diffusion loss
            with torch.no_grad():
                feature_maps_1 = feature_extractor(img_1) # for frame t
                feature_maps_2 = feature_extractor(img_2) # for frame t+1

            diffusion_loss = {}
            diffusion_loss_details = {}


            for i, key in enumerate(SIM_FEATURE_LAYERS):

                feature_maps_1[i] = F.interpolate(feature_maps_1[i], size=FEATURE_MAP_SIZE, mode='bicubic', align_corners=True)

                feature_maps_2[i] = F.interpolate(feature_maps_2[i], size=FEATURE_MAP_SIZE, mode='bicubic', align_corners=True)

                _diff_loss, _diff_details = get_diffusion_loss(
                    feature_maps_1[i], feature_maps_2[i], logits_1, logits_2,
                    fg_margin=SIM_FG_MARGINS[i], bg_margin=SIM_BG_MARGINS[i],
                    fg_ratio=fg_ratio, bg_ratio=bg_ratio,
                    naming='sim_{}'.format(key))

                diffusion_loss[key] = _diff_loss
                diffusion_loss_details[key] = _diff_details

            val_loss = anchor_loss

            for key in SIM_FEATURE_LAYERS:
                val_loss += diffusion_loss[key]

            losses.append(val_loss.item())


        print(f'validation loss  : {val_loss}')
        mean_loss = np.mean(losses)
        return mean_loss


def main():
    
    model = MyUNet(transposed_conv = False, align_corners = False)
    model.to(device)
    

    train_train_datadict = get_datadict(   # training data in the Single Stage setting
        img_dir=IMG_DIR,
        pos_anchor_dir=POS_ANCHOR_DIR,
        neg_anchor_dir=NEG_ANCHOR_DIR,
        ground_truth_dir=None,             # Unsupervised training
        sub_dir_list=TRAIN_DIR_LIST
    )

    train_train_dataset = SegmentationDataset(
        train_train_datadict, 
        original_img_size = ORIGINAL_IMG_SIZE,
        preprocess_crop_size = PREPROCESS_CROP_SIZE,
        preprocess_crop_location = PREPROCESS_CROP_LOCATION,
        model_input_size = MODEL_INPUT_SIZE,
        data_aug = True, 
        clip_size = 2)

    train_train_loader = torch.utils.data.DataLoader(
        train_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    

    val_datadict = get_datadict(   # validation data in the Single Stage setting
        img_dir=IMG_DIR,
        pos_anchor_dir=POS_ANCHOR_DIR,
        neg_anchor_dir=NEG_ANCHOR_DIR,
        ground_truth_dir=None,             # Unsupervised training
        sub_dir_list=TRAIN_DIR_LIST
    )

    val_dataset = SegmentationDataset(
        val_datadict, 
        original_img_size = ORIGINAL_IMG_SIZE,
        preprocess_crop_size = PREPROCESS_CROP_SIZE,
        preprocess_crop_location = PREPROCESS_CROP_LOCATION,
        model_input_size = MODEL_INPUT_SIZE,
        data_aug = True, 
        clip_size = 2)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    
    best_val_loss = 1e9
    for currentepoch in range(START_EPOCH,EPOCH_NUM):
        wandb.watch(model,log="all")

        train_loss, model = train_one_epoch(train_train_loader, model,currentepoch)
        mean_loss = val_one_epoch(val_loader, model,currentepoch)
    
        segentation_checkpoint = {
                "state_dict": model.state_dict(),
            }
        model_name="segmentation_checkpoint"
        save_checkpoint(segentation_checkpoint,currentepoch,model_name)

        if mean_loss < best_val_loss:
            best_val_loss = mean_loss

            segentation_checkpoint = {
                "state_dict": model.state_dict(),
            }
            model_name="best_segmentation_checkpoint"

            save_checkpoint(segentation_checkpoint,currentepoch,model_name)


    return mean_loss

    

if __name__ == '__main__':
    main()
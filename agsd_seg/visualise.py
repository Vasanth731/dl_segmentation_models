# classic imports
import torch
import cv2
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from utils import *
from model import MyUNet
from skimage import morphology

# hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# path = "/home/htic/Videos/segmentation_checkpoint/SEG_best_segmentation_checkpoint.pth"
path = "SEG_best_segmentation_checkpoint.pth"
PREPROCESS_CROP_SIZE = [1024,1280]
transposed_conv = False
align_corners = False
image_path = '/home/htic/Pictures/dataset_img/bipolar dissector/clip_016657.png'
# image_path = '/home/htic/Pictures/archive/tiny-imagenet-200/val/images/val_0.JPEG'

# function defenitions for prediction image
def transform(image_path):
    preprocess = transforms.Compose([
    transforms.Resize((512, 640)),  # Resize to match the expected input size
    transforms.ToTensor(),            # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

    input_image = Image.open(image_path).convert('RGB')
    input_image = preprocess(input_image).unsqueeze(0)  # Add batch dimension
    input_image = input_image.unsqueeze(1)
    input_image = input_image.repeat(1,2,1,1,1)
    img = input_image
    img_1, img_2 = img[:,0,:,:,:], img[:,1,:,:,:]
    img_1 = img_1.to(device)
    return img_1

def pred(output):
    probs = torch.sigmoid(output).squeeze().cpu().detach().numpy()
    probs = cv2.resize(probs, (PREPROCESS_CROP_SIZE[1], PREPROCESS_CROP_SIZE[0]))
    probs = (probs * 255).astype(np.uint8)
    pred = get_mask(probs, threshold=255*0.6)
    pred = morphology.remove_small_objects(pred, 10000)
    return pred

# function defenitions for anchor image
def anchor_transform(image_path):
    preprocess = transforms.Compose([
    transforms.Resize((512, 640)),  # Resize to match the expected input size
    transforms.ToTensor(),            # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
    anchor_image = Image.open(image_path).convert('RGB')
    anchor_image = preprocess(anchor_image).unsqueeze(0)  # Add batch dimension
    anchor_image = anchor_image.unsqueeze(1)
    anchor_image = anchor_image.repeat(1,2,1,1,1)
    anchor_image = anchor_image.permute(0, 1, 3, 4, 2)
    anchor_image = anchor_image[:, :, :, :, 0:1] 
    anc = anchor_image.to(device)
    return anc

def anchor(anc):
    pos_anchor = (anc.squeeze().cpu().detach().numpy() * 255).astype(np.uint8)
    neg_anchor = (anc.squeeze().cpu().detach().numpy() * 255).astype(np.uint8)
    pos_anchor = get_mask(pos_anchor, threshold=255*0.6)
    neg_anchor = get_mask(255 - neg_anchor, threshold=255*0.6)
    pos_anchor = morphology.remove_small_objects(pos_anchor, 10000)
    neg_anchor = morphology.remove_small_objects(neg_anchor, 10000)
    pos_anchor = pos_anchor[0]
    neg_anchor = neg_anchor[0]
    return pos_anchor, neg_anchor




if __name__ == "__main__":

    # declare model
    model = MyUNet(transposed_conv, align_corners)
    model = model.to(device)
    model_pretrained_statedict = torch.load(path)
    model.load_state_dict(model_pretrained_statedict["state_dict"])

    # pred image part
    transform_inst = transform(image_path)
    output = model(transform_inst)
    pred_inst = pred(output)

    # anchor image part
    anchor_transform_inst = anchor_transform(image_path)
    anchor_inst_pos, anchor_inst_neg  = anchor(anchor_transform_inst)

    # normal image
    img_1 = transform_inst.squeeze().permute(1, 2, 0).cpu().numpy()

    # plot
    plt.subplot(141)
    plt.imshow(pred_inst, cmap='gray')
    plt.title("Predicted Image")
    plt.axis("off")

    plt.subplot(142)
    plt.imshow(anchor_inst_pos, cmap='gray')
    plt.title("Positive Anchor Mask")
    plt.axis("off")

    plt.subplot(143)
    plt.imshow(anchor_inst_neg, cmap='gray')
    plt.title("Negative Anchor Mask")
    plt.axis("off")

    plt.subplot(144)
    plt.imshow(img_1, cmap='gray')
    plt.title("Original Image")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
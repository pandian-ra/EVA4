import os
import PIL
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision.utils import make_grid, save_image

class GradCamHelper():
  def getGradCamImg(im):
    normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    torch_img = torch.from_numpy(np.asarray(im)).permute(2, 0, 1).unsqueeze(0).float().div(255).cuda()
    torch_img = F.upsample(torch_img, size=(32, 32), mode='bilinear', align_corners=False)
    normed_torch_img = normalizer(torch_img)
    mask, _ = resnet_gradcam(normed_torch_img)
    heatmap, result = visualize_cam(mask, torch_img)
    return result

  def wrong_plot(true,ima,pred,encoder,inv_normalize,n_figures ):
    print('Classes in order Actual and Predicted')
    n_row = int(n_figures/5)
    fig,axes = plt.subplots(figsize=(15, 15), nrows = n_row, ncols=5)
    print(axes)
    for ax in axes.flatten():
        a = random.randint(0,len(true)-1)
        image,correct,wrong = ima[a],true[a],pred[a]
        f = 'A:'+classes[correct[0]] + ',' +'P:'+classes[wrong[0]]
        ax.imshow(transforms.ToPILImage()(getGradCamImg(torchvision.utils.make_grid(image.cpu(), nrow=5).permute(1, 2, 0))), cmap='brg', interpolation='none')
        ax.set_title(f)
        ax.axis('off')
    plt.show()
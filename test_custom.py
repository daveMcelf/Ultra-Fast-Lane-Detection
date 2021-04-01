import torch, os, cv2
from model.model import parsingNet
from utils.common import merge_config
from utils.dist_utils import dist_print
import torch
import scipy.special, tqdm
import numpy as np
import bezier
import torchvision.transforms as transforms
from data.dataset import LaneTestDataset
from data.constant import culane_row_anchor, tusimple_row_anchor

def plot_line(points):
  points = np.array(points,dtype=int).T
  curve = bezier.Curve(points, degree=points.shape[1]-1)
  s_vals = np.linspace(0.0, 1.0, 100)   # 100 is the num of pionts
  data = curve.evaluate_multi(s_vals)
  data = tuple(data.T)
  return data

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    args, cfg = merge_config()
    dist_print('start testing...')
    assert cfg.backbone in ['18','34','50','101','152','50next','101next','50wide','101wide']
    if cfg.dataset == 'CULane':
        cls_num_per_lane = 18
    elif cfg.dataset == 'Tusimple':
        cls_num_per_lane = 56
    else:
        raise NotImplementedError

    net = parsingNet(pretrained = False, backbone=cfg.backbone,cls_dim = (cfg.griding_num+1,cls_num_per_lane,4),
                    use_aux=False).cuda() # we dont need auxiliary segmentation in testing

    state_dict = torch.load(cfg.test_model, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()

    img_transforms = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    cap = cv2.VideoCapture(cfg.video_path)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    vout = cv2.VideoWriter(str(123)+'.avi', fourcc , 30.0, (int(cap.get(3)),int(cap.get(4))))

    print("w = {},h = {}".format(cap.get(3),cap.get(4)))

    from PIL import Image
    # print('加载cuda是否成功：',torch.cuda.is_available())
    while 1:
        rval,frame = cap.read()
        if rval == False:
            break
        # cv2.imwrite("ssss.jpg",frame)
        # img_ = Image.open("ssss.jpg")
        img  = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        img_ = Image.fromarray(img)
        imgs = img_transforms(img_)
        imgs = imgs.unsqueeze(0)
        imgs = imgs.cuda()
        with torch.no_grad():
            out = net(imgs)

        col_sample = np.linspace(0, 800 - 1, cfg.griding_num)
        col_sample_w = col_sample[1] - col_sample[0]
        out_j = out[0].data.cpu().numpy()
        out_j = out_j[:, ::-1, :]
        prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
        idx = np.arange(cfg.griding_num) + 1
        idx = idx.reshape(-1, 1, 1)
        loc = np.sum(prob * idx, axis=0)
        out_j = np.argmax(out_j, axis=0)
        loc[out_j == cfg.griding_num] = 0
        out_j = loc
        # import pdb; pdb.set_trace()
        # vis = cv2.imread(os.path.join(cfg.data_root,names[0]))
        for i in range(out_j.shape[1]):
            line = []
            if np.sum(out_j[:, i] != 0) > 2:
                for k in range(out_j.shape[0]):
                    if out_j[k, i] > 0:
                        ppp = (int(out_j[k, i] * col_sample_w * cap.get(3) / 800) - 1, int(cap.get(4) * (culane_row_anchor[cls_num_per_lane-1-k]/288)) - 1)
                        # cv2.circle(frame,ppp,2,(0,255,0),-1)
                        line.append(ppp)
                vis_dots = plot_line(line)
                for dot in vis_dots:
                    cv2.circle(frame,(int(dot[0]), int(dot[1])),5,(0,255,0),-1)
        vout.write(frame)
    vout.release()
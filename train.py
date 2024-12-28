import argparse
import math
import os
import ssl
import time
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.models import vgg16
from torchvision.utils import save_image
from tqdm import tqdm
from Model import BMFH
from Model_util import padding_image
from make import getTxt
from perceptual import LossNetwork
from pytorch_msssim import msssim
from test_dataset import dehaze_test_dataset
from train_dataset import dehaze_train_dataset
from utils_test import to_psnr, to_ssim_skimage

ssl._create_default_https_context = ssl._create_unverified_context

# --- Parse hyper-parameters train --- #
parser = argparse.ArgumentParser(description='Siamese Dehaze Network')
parser.add_argument('-learning_rate', help='Set the learning rate', default=1e-4, type=float)
parser.add_argument('-train_batch_size', help='Set the training batch size', default=4, type=int)
parser.add_argument('-train_epoch', help='Set the training epoch', default=10000, type=int)
parser.add_argument("--type", default=-1, type=int, help="choose a type 123456")

# parser.add_argument('--train_dir', type=str, default='')
parser.add_argument('--train_dir', type=str, default='')
parser.add_argument('--train_name', type=str, default='hazy,clean')
parser.add_argument('--test_dir', type=str, default='')
parser.add_argument('--test_name', type=str, default='hazy,clean')

parser.add_argument('--model_save_dir', type=str, default='./output_result')
parser.add_argument('--log_dir', type=str, default=None)
parser.add_argument('--gpus', default='3', type=str)
# --- Parse hyper-parameters test --- #
parser.add_argument('--predict_result', type=str, default='./output_result/picture/')
parser.add_argument('-test_batch_size', help='Set the testing batch size', default=1, type=int)
parser.add_argument('--vgg_model', default='', type=str, help='load trained model or not')
parser.add_argument('--use_bn', action='store_true', help='if bs>8 please use bn')
parser.add_argument('--restart', action='store_true', help='')
parser.add_argument('--num', type=str, default='9999999', help='')
parser.add_argument('--sep', type=int, default='5', help='')
parser.add_argument('--save_psnr', action='store_true', help='')
parser.add_argument('--seps', action='store_true', help='')

print('+++++++++++++++++++++++++++++++ Train set ++++++++++++++++++++++++++++++++++++++++')

args = parser.parse_args()
# --- train --- #
learning_rate = args.learning_rate
train_batch_size = args.train_batch_size
train_epoch = args.train_epoch
start_epoch = 0
sep = args.sep

if args.type == 1:
    args.train_dir = '/data1/ghy/lsl/Datasets/thin_660/test/'
    args.train_name = 'hazy,clean'
    args.test_dir = "/data1/ghy/lsl/Datasets/thin_660/test/"
    args.test_name = 'hazy,clean'
    args.out_pic = './output_pic/thin'
    args.pre_name = 'pre_model_thin.pkl'
    tag = 'thin'
elif args.type == 2:
    args.train_dir = '/T2020027/datasets/Haze1k/train/'
    args.train_name = 'hazy,clean'
    args.test_dir = "/T2020027/datasets/Haze1k/moderate/test/"
    args.test_name = 'hazy,clean'
    args.out_pic = './output_pic/moderation'
    args.pre_name = 'pre_model_moderation.pkl'
    tag = 'moderation'
elif args.type == 3:
    args.train_dir = '/data1/ghy/lsl/Datasets/thick_660/test/'
    args.train_name = 'hazy,clean'
    args.test_dir = "/data1/ghy/lsl/Datasets/thick_660/test/"
    args.test_name = 'hazy,clean'
    args.out_pic = './output_pic/thick'
    args.pre_name = 'pre_model_thick.pkl'
    tag = 'thick'
elif args.type == 4:
    args.train_dir = '/data1/ghy/lsl/Datasets/dataset_rice1/train/'
    args.train_name = 'hazy,clean'
    args.test_dir = "/data1/ghy/lsl/Datasets/dataset_rice1/test/"
    args.test_name = 'hazy,clean'
    args.out_pic = './output_pic/RICE1'
    args.pre_name = 'pre_model_RICE1.pkl'
    tag = 'RICE1'
elif args.type == 5:
    args.train_dir = '/data1/ghy/lsl/Datasets/dataset_rice2/train/'
    args.train_name = 'hazy,clean'
    args.test_dir = "/data1/ghy/lsl/Datasets/dataset_rice2/test/"
    args.test_name = 'hazy,clean'
    args.out_pic = './output_pic/RICE2'
    args.pre_name = 'pre_model_RICE2.pkl'
    tag = 'RICE2'
elif args.type == 6:
    args.train_dir = '/T2020027/zqf/dataset/RSID_1/RSID_1/train/'
    args.train_name = 'hazy,clean'
    args.test_dir = "/T2020027/zqf/dataset/RSID_1/RSID_1/test/"
    args.test_name = 'hazy,clean'
    args.out_pic = './output_pic/RICE2'
    args.pre_name = 'pre_model_RSID.pkl'
    tag = 'RSID'

elif args.type == 7:
    args.train_dir = '/data1/lsl/zqf/dataset/haze1k/train/'
    args.train_name = 'hazy,clean'
    args.test_dir = "/data1/lsl/zqf/dataset/haze1k/test/"
    args.test_name = 'hazy,clean'
    args.out_pic = './output_pic/thin'
    args.pre_name = 'pre_model_thin.pkl'
    tag = 'thin'


# ********************our_data*************

elif args.type == 11:
    args.train_dir = '/data1/ghy/lsl/Datasets/dataset_moderation/train/'
    args.train_name = 'hazy,clean'
    args.test_dir = "/data1/ghy/lsl/Datasets/dataset_moderation/test/"
    args.test_name = 'hazy,clean'
    args.out_pic = './output_pic/moderation'
    args.pre_name = 'pre_model_moderation.pkl'
    tag = 'moderation_pt'
elif args.type == 12:
    args.train_dir = '/data1/ghy/lsl/Datasets/RICE1_ALL/'
    args.train_name = 'hazy,clean'
    args.test_dir = "/data1/ghy/lsl/Datasets/RICE1_ALL/"
    args.test_name = 'hazy,clean'
    args.out_pic = './output_pic/RICE1'
    args.pre_name = 'pre_model_RICE1.pkl'
    tag = 'RICE1'
elif args.type == 13:
    args.train_dir = '/data1/ghy/lsl/Datasets/RICE2_ALL/'
    args.train_name = 'hazy,clean'
    args.test_dir = "/data1/ghy/lsl/Datasets/RICE2_ALL/"
    args.test_name = 'hazy,clean'
    args.out_pic = './output_pic/RICE2'
    args.pre_name = 'pre_model_RICE2.pkl'
    tag = 'RICE2'
elif args.type == 14:
    args.train_dir = '/data1/ghy/lsl/Datasets/RSID_all/'
    args.train_name = 'hazy,clean'
    args.test_dir = "/data1/ghy/lsl/Datasets/RSID_all/"
    args.test_name = 'hazy,clean'
    args.out_pic = './output_pic/RSID'
    args.pre_name = 'pre_model_RSID.pkl'
    tag = 'RSID'
elif args.type == 15:
    args.train_dir = '/data1/ghy/lsl/Datasets/huapo/'
    args.train_name = 'hazy,clean'
    args.test_dir = "/data1/ghy/lsl/Datasets/huapo/"
    args.test_name = 'hazy,clean'
    args.out_pic = './output_pic/huapo'
    args.pre_name = ''
    tag = 'huapo'

print('We are training datasets: ', tag)

getTxt(args.train_dir, args.train_name, args.test_dir, args.test_name)

predict_result = args.predict_result
test_batch_size = args.test_batch_size

# --- output picture and check point --- #
if not os.path.exists(args.model_save_dir):
    os.makedirs(args.model_save_dir)
# output_dir = os.path.join(args.model_save_dir, 'output_result')

# --- Gpu device --- #
device_ids = [int(i) for i in list(filter(str.isdigit, args.gpus))]

print('use gpus ->', args.gpus)
device = torch.device("cuda:{}".format(device_ids[0]) if torch.cuda.is_available() else "cpu")

# --- Define the network --- #
if args.use_bn:
    print('we are using BatchNorm')
else:
    print('we are using InstanceNorm')

D3D = BMFH().to(device)
print('D3D parameters:', sum(param.numel() for param in D3D.parameters()))

# --- Build optimizer --- #
G_optimizer = torch.optim.Adam(D3D.parameters(), lr=0.0001)
scheduler_G = torch.optim.lr_scheduler.MultiStepLR(G_optimizer, milestones=[5000, 7000, 8000], gamma=0.5)

# --- Load training data --- #
dataset = dehaze_train_dataset(args.train_dir, args.train_name, tag)
print('trainDataset len: ', len(dataset))
train_loader = DataLoader(dataset=dataset, batch_size=train_batch_size, shuffle=True, pin_memory=True, drop_last=True,
                          num_workers=4)
# --- Load testing data --- #

test_dataset = dehaze_test_dataset(args.test_dir, args.test_name, tag)
print('testDataset len: ', len(test_dataset))
test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=0,
                         pin_memory=True)

# val_dataset = dehaze_val_dataset(val_dataset)
# val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=0)

# --- Multi-GPU --- #
D3D = D3D.to(device)
D3D = torch.nn.DataParallel(D3D, device_ids=device_ids)

writer = SummaryWriter(os.path.join(args.model_save_dir, 'tensorboard'))

# --- Define the perceptual loss network --- #
vgg_model = vgg16(pretrained=True)
# vgg_model.load_state_dict(torch.load(os.path.join(args.vgg_model , 'vgg16.pth')))
vgg_model = vgg_model.features[:16].to(device)
for param in vgg_model.parameters():
    param.requires_grad = False

loss_network = LossNetwork(vgg_model)
loss_network.eval()

msssim_loss = msssim

# --- Load the Pre_Model weight --- #
# pre_model.load_state_dict(
#    torch.load(os.path.join('Pre_Model/', args.pre_name),
#               map_location="cuda:{}".format(device_ids[0]
#                                             )))
# pre_dict = pre_model.state_dict()
# model_dict = D3D.state_dict()
# pre_model_dict = {k: v for k, v in pre_dict.items() if k in model_dict}
# model_dict.update(pre_model_dict)
# D3D.load_state_dict(model_dict)
# print('--- Pre_Train weight loaded! ---')

# --- Load the network weight --- #
if args.restart:
    pkl_list = [i for i in os.listdir(args.model_save_dir) if '.pkl' in i]
    num = sorted([int(i.split('.')[0].split('_')[1]) for i in pkl_list])[-1]
    name = [i for i in pkl_list if 'epoch_' + str(num) + '_' in i][0]
    D3D.load_state_dict(
        torch.load(os.path.join(args.model_save_dir, name),
                   map_location="cuda:{}".format(device_ids[0])))
    print('--- {} epoch weight loaded ---'.format(num))
    start_epoch = int(num) + 1
elif args.num != '9999999':
    pkl_list = [i for i in os.listdir(args.model_save_dir) if '.pkl' in i]
    name = [i for i in pkl_list if 'epoch_' + str(args.num) + '_' in i][0]
    D3D.load_state_dict(
        torch.load(os.path.join(args.model_save_dir, name),
                   map_location="cuda:{}".format(device_ids[0])))
    print('--- {} epoch weight loaded ---'.format(args.num))
    start_epoch = int(args.num) + 1
else:
    print('--- no weight loaded ---')


def cosine_similarity_loss(output1, output2, eps=1e-6):
    
    output1_pooled = F.adaptive_avg_pool2d(output1, (1, 1)).squeeze(-1).squeeze(-1)
    output2_pooled = F.adaptive_avg_pool2d(output2, (1, 1)).squeeze(-1).squeeze(-1)

    cos_sim = F.cosine_similarity(output1_pooled, output2_pooled, dim=1, eps=eps)

    loss = 1 - cos_sim
    return loss.mean()


iteration = 0
best_epoch_psnr = 0
best_epoch_ssim = 0
pl = []
sl = []
best_psnr = 0
best_psnr_ssim = 0
best_ssim = 0
best_ssim_psnr = 0
print()
start_time = time.time()

for epoch in range(start_epoch, train_epoch):
    print('++++++++++++++++++++++++ {} Datasets +++++++ {} epoch ++++++++++++++++++++++++'.format(tag, epoch))
    scheduler_G.step()
    D3D.train()
    with tqdm(total=len(train_loader)) as t:
        for (hazy, clean) in train_loader:
            # print(batch_idx)
            iteration += 1
            hazy = hazy.to(device)
            clean = clean.to(device)
            CNN_feature,tf_feature,img = D3D(hazy)
            # no more forward
            D3D.zero_grad()

            smooth_loss_l1 = F.smooth_l1_loss(img, clean)
            perceptual_loss = loss_network(img, clean)
            msssim_loss_ = 1 - msssim_loss(img, clean, normalize=True)
            feature_align = 0.0
            for i in range(len(CNN_feature)):
                feature_align += cosine_similarity_loss(CNN_feature[i], tf_feature[i])



            total_loss = smooth_loss_l1 + 0.01 * perceptual_loss + 0.5 * msssim_loss_ + feature_align

            total_loss.backward()
            G_optimizer.step()

            writer.add_scalars('training', {'training total loss': total_loss.item()
                                            }, iteration)
            writer.add_scalars('training_img', {'img loss_l1': smooth_loss_l1.item(),
                                                'perceptual': perceptual_loss.item(),
                                                'msssim': msssim_loss_.item()

                                                }, iteration)

            t.set_description(
                "===> Epoch[{}] :  total_loss: {:.2f}   ".format(
                    epoch, total_loss.item(),
                    time.time() - start_time))
            t.update(1)

    if args.seps:
        torch.save(D3D.state_dict(),
                   os.path.join(args.model_save_dir,
                                'epoch_' + str(epoch) + '_' + '.pkl'))
        continue

    if tag in []:
        if epoch >= 30:
            sep = 1
    elif tag in ['thin', 'thick', 'moderation', 'RICE1', 'RICE2', 'RSID', 'NID', 'moderation_pt', 'RICE1_ALL',
                 'RICE2_ALL']:
        if epoch >= 100:
            sep = 1
    else:
        if epoch >= 500:
            sep = 1

    if epoch % sep == 0:

        with torch.no_grad():
            i = 0
            psnr_list = []
            ssim_list = []
            D3D.eval()
            for (hazy, clean, _) in tqdm(test_loader):
                hazy = hazy.to(device)
                clean = clean.to(device)

                h, w = hazy.shape[2], hazy.shape[3]
                max_h = int(math.ceil(h / 4)) * 4
                max_w = int(math.ceil(w / 4)) * 4
                hazy, ori_left, ori_right, ori_top, ori_down = padding_image(hazy, max_h, max_w)

                CNN_feature, tf_feature,frame_out = D3D(hazy)
                if i % 200 == 0:
                    save_image(frame_out, os.path.join(args.out_pic, str(epoch) + '_' + str(i) + '_' + '.png'))
                i = i + 1

                frame_out = frame_out.data[:, :, ori_top:ori_down, ori_left:ori_right]

                psnr_list.extend(to_psnr(frame_out, clean))
                ssim_list.extend(to_ssim_skimage(frame_out, clean))

            avr_psnr = sum(psnr_list) / len(psnr_list)
            avr_ssim = sum(ssim_list) / len(ssim_list)
            pl.append(avr_psnr)
            sl.append(avr_ssim)
            if avr_psnr >= max(pl):
                best_epoch_psnr = epoch
                best_psnr = avr_psnr
                best_psnr_ssim = avr_ssim
            if avr_ssim >= max(sl):
                best_epoch_ssim = epoch
                best_ssim = avr_ssim
                best_ssim_psnr = avr_psnr

            print(epoch, 'dehazed', avr_psnr, avr_ssim)
            if best_epoch_psnr == best_epoch_ssim:
                print('best epoch is {}, psnr: {}, ssim: {}'.format(best_epoch_psnr, best_psnr, best_ssim))
            else:
                print('best psnr epoch is {}: PSNR: {}, ssim: {}'.format(best_epoch_psnr, best_psnr, best_psnr_ssim))
                print('best ssim epoch is {}: psnr: {}, SSIM: {}'.format(best_epoch_ssim, best_ssim_psnr, best_ssim))
            print()
            frame_debug = torch.cat((frame_out, clean), dim=0)
            writer.add_images('my_image_batch', frame_debug, epoch)
            writer.add_scalars('testing', {'testing psnr': avr_psnr,
                                           'testing ssim': avr_ssim
                                           }, epoch)
            if best_epoch_psnr == epoch or best_epoch_ssim == epoch:
                torch.save(D3D.state_dict(),
                           os.path.join(args.model_save_dir,
                                        'epoch_' + str(epoch) + '_' + str(round(avr_psnr, 2)) + '_' + str(
                                            round(avr_ssim, 3)) + '_' + str(tag) + '.pkl'))
os.remove(os.path.join(args.train_dir, 'train.txt'))
os.remove(os.path.join(args.test_dir, 'test.txt'))
writer.close()

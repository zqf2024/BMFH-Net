# import argparse
# import math
# import os

# import torch
# from torch.utils.data import DataLoader
# from torchvision.utils import save_image as imwrite
# from tqdm import tqdm

# from Model import Base_Model
# from Model_util import padding_image
# from make import getTxt
# from test_dataset import dehaze_test_dataset
# from utils_test import to_psnr, to_ssim_skimage


# # --- Parse hyper-parameters train --- #
# parser = argparse.ArgumentParser(description='Siamese Dehaze Network')
# parser.add_argument('--data_dir', type=str, default='')
# parser.add_argument('--model_save_dir', type=str, default='./output_result')#存放测试结果
# parser.add_argument('--log_dir', type=str, default=None)
# parser.add_argument('--gpus', default='0,1,2,3', type=str)
# # --- Parse hyper-parameters test --- #
# parser.add_argument('--test_dataset', type=str, default='')
# parser.add_argument('-test_batch_size', help='Set the testing batch size', default=1, type=int)
# parser.add_argument('--test_dir', type=str, default='/home/sunhang/lzm/deHaze/outdoor_Test/')
# parser.add_argument('--test_name', type=str, default='hazy,clean')
# parser.add_argument('--num', type=str, default='9999999', help='')
# parser.add_argument('--use_bn', action='store_true', help='if bs>8 please use bn')
# parser.add_argument("--type", default=0, type=int, help="choose a type 012345")
# args = parser.parse_args()

# predict_result = os.path.join(args.model_save_dir, 'result_pic')
# test_batch_size = args.test_batch_size

# # --- output picture and check point --- #
# if not os.path.exists(args.model_save_dir):
#     os.makedirs(args.model_save_dir)

# if not os.path.exists(predict_result):
#     os.makedirs(predict_result)

# output_dir = os.path.join(args.model_save_dir, '')

# # --- Gpu device --- #
# device_ids = [int(i) for i in list(filter(str.isdigit, args.gpus))]
# print('use gpus ->', args.gpus)
# device = torch.device("cuda:{}".format(device_ids[0]) if torch.cuda.is_available() else "cpu")

# # --- Define the network --- #
# if args.use_bn:
#     print('we are using BatchNorm')
# else:
#     print('we are using InstanceNorm')

# D3D = Base_Model(3,3)
# print('D3D parameters:', sum(param.numel() for param in D3D.parameters()))
# # --- Multi-GPU --- #
# D3D = D3D.to(device)
# D3D = torch.nn.DataParallel(D3D, device_ids=device_ids)

# # tag = 'else'
# if args.type == 1:
#     args.train_dir = '/data1/ghy/lsl/Datasets/thin_660/train/'
#     args.train_name = 'hazy,clean'
#     args.test_dir = "/data1/ghy/lsl/Datasets/thin_660/test/"
#     args.test_name = 'hazy,clean'
#     args.out_pic = './output_pic/thin'
#     args.pre_name = 'pre_model_thin.pkl'
#     tag = 'thin'
# elif args.type == 2:
#     args.train_dir = '/data1/ghy/lsl/Datasets/mode_660/train/'
#     args.train_name = 'hazy,clean'
#     args.test_dir = "/data1/ghy/lsl/Datasets/mode_660/test/"
#     args.test_name = 'hazy,clean'
#     args.out_pic = './output_pic/moderation'
#     args.pre_name = 'pre_model_moderation.pkl'
#     tag = 'moderation'
# elif args.type == 3:
#     args.train_dir = '/data1/ghy/lsl/Datasets/thick_660/train/'
#     args.train_name = 'hazy,clean'
#     args.test_dir = "/data1/ghy/lsl/Datasets/thick_660/test/"
#     args.test_name = 'hazy,clean'
#     args.out_pic = './output_pic/thick'
#     args.pre_name = 'pre_model_thick.pkl'
#     tag = 'thick'
# elif args.type == 4:
#     args.train_dir = '/data1/ghy/lsl/Datasets/dataset_rice1/train/'
#     args.train_name = 'hazy,clean'
#     args.test_dir = "/data1/ghy/lsl/Datasets/dataset_rice1/test/"
#     args.test_name = 'hazy,clean'
#     args.out_pic = './output_pic/RICE1'
#     args.pre_name = 'pre_model_RICE1.pkl'
#     tag = 'RICE1'
# elif args.type == 5:
#     args.train_dir = '/data1/ghy/lsl/Datasets/dataset_rice2/train/'
#     args.train_name = 'hazy,clean'
#     args.test_dir = "/data1/ghy/lsl/Datasets/dataset_rice2/test/"
#     args.test_name = 'hazy,clean'
#     args.out_pic = './output_pic/RICE2'
#     args.pre_name = 'pre_model_RICE2.pkl'
#     tag = 'RICE2'
# elif args.type == 6:
#     args.train_dir = '/data1/ghy/lsl/Datasets/RSID/train/'
#     args.train_name = 'hazy,clean'
#     args.test_dir = "/data1/ghy/lsl/Datasets/RSID/test/"
#     args.test_name = 'hazy,clean'
#     args.out_pic = './output_pic/RICE2'
#     args.pre_name = 'pre_model_RSID.pkl'
#     tag = 'RSID'




# #********************our_data*************




# elif args.type == 11:
#     args.train_dir = '/data1/ghy/lsl/Datasets/dataset_moderation/train/'
#     args.train_name = 'hazy,clean'
#     args.test_dir = "/data1/ghy/lsl/Datasets/dataset_moderation/test/"
#     args.test_name = 'hazy,clean'
#     args.out_pic = './output_pic/moderation'
#     args.pre_name = 'pre_model_moderation.pkl'
#     tag = 'moderation_pt'
# elif args.type == 12:
#     args.train_dir = '/data1/ghy/lsl/Datasets/RICE1_ALL/'
#     args.train_name = 'hazy,clean'
#     args.test_dir = "/data1/ghy/lsl/Datasets/RICE1_ALL/"
#     args.test_name = 'hazy,clean'
#     args.out_pic = './output_pic/RICE1'
#     args.pre_name = 'pre_model_RICE1.pkl'
#     tag = 'RICE1'
# elif args.type == 13:
#     args.train_dir = '/data1/ghy/lsl/Datasets/RICE2_ALL/'
#     args.train_name = 'hazy,clean'
#     args.test_dir = "/data1/ghy/lsl/Datasets/RICE2_ALL/"
#     args.test_name = 'hazy,clean'
#     args.out_pic = './output_pic/RICE2'
#     args.pre_name = 'pre_model_RICE2.pkl'
#     tag = 'RICE2'
# elif args.type == 14:
#     args.train_dir = '/data1/ghy/lsl/Datasets/RSID_all/'
#     args.train_name = 'hazy,clean'
#     args.test_dir = "/data1/ghy/lsl/Datasets/RSID_all/"
#     args.test_name = 'hazy,clean'
#     args.out_pic = './output_pic/RSID'
#     args.pre_name = 'pre_model_RSID.pkl'
#     tag = 'RSID'

#     print('We are testing datasets: ', tag)
#     test_hazy = os.listdir(os.path.join(args.test_dir, 'hazy'))
#     print('创建test.txt成功')
#     test_txts = open(os.path.join(args.test_dir, 'test.txt'), 'w+')
#     for i in test_hazy:
#         tmp = test_txts.writelines(i + '\n')

#     predict_dir = os.path.join(predict_result, 'predict')
#     if not os.path.exists(predict_dir):
#         os.makedirs(predict_dir)
#     args.test_name = 'hazy,hazy'



# print('We are testing datasets: ', tag)
# getTxt(None, None, args.test_dir, args.test_name)
# test_hazy, test_gt = args.test_name.split(',')
# if not os.path.exists(os.path.join(predict_result, 'hazy')):
#     os.makedirs(os.path.join(predict_result, 'hazy'))
# os.system('cp -r {}/* {}'.format(os.path.join(args.test_dir, test_hazy), os.path.join(predict_result, 'hazy')))

# if not os.path.exists(os.path.join(predict_result, 'clean')):
#     os.makedirs(os.path.join(predict_result, 'clean'))
# os.system('cp -r {}/* {}'.format(os.path.join(args.test_dir, test_gt), os.path.join(predict_result, 'clean')))

# predict_dir = os.path.join(predict_result, 'predict')
# if not os.path.exists(predict_dir):
#     os.makedirs(predict_dir)

# test_dataset = dehaze_test_dataset(args.test_dir, args.test_name, tag=tag)
# test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

# # --- Load the network weight --- #
# if args.num != '9999999':
#     pkl_list = [i for i in os.listdir(args.model_save_dir) if '.pkl' in i]
#     name = [i for i in pkl_list if 'epoch_' + str(args.num) + '_' in i][0]
#     D3D.load_state_dict(
#         torch.load(os.path.join(args.model_save_dir, name),
#                    map_location="cuda:{}".format(device_ids[0])))
#     print('--- {} epoch weight loaded ---'.format(args.num))
#     start_epoch = int(args.num) + 1
# else:
#     pkl_list = [i for i in os.listdir(args.model_save_dir) if '.pkl' in i]
#     num = sorted([int(i.split('.')[0].split('_')[1]) for i in pkl_list])[-1]
#     name = [i for i in pkl_list if 'epoch_' + str(num) + '_' in i][0]
#     D3D.load_state_dict(
#         torch.load(os.path.join(args.model_save_dir, name),
#                    map_location="cuda:{}".format(device_ids[0])))
#     print('--- {} epoch weight loaded ---'.format(num))

# test_txt = open(os.path.join(predict_result, 'result.txt'), 'w+')
# # --- Strat testing --- #
# with torch.no_grad():
#     img_list = []
#     psnr_list = []
#     ssim_list = []
#     D3D.eval()
#     imsave_dir = output_dir
#     if not os.path.exists(imsave_dir):
#         os.makedirs(imsave_dir)
#     for (hazy, clean,name) in test_loader:
#         hazy = hazy.to(device)
#         clean = clean.to(device)
#         h, w = hazy.shape[2], hazy.shape[3]
#         max_h = int(math.ceil(h / 4)) * 4
#         max_w = int(math.ceil(w / 4)) * 4
#         hazy, ori_left, ori_right, ori_top, ori_down = padding_image(hazy, max_h, max_w)

#         frame_out1 = D3D(hazy)
#         frame_out2 = D3D(hazy)
#         frame_out3 = D3D(hazy)
#         frame_out4 = D3D(hazy)
#         frame_out5 = D3D(hazy)


#         frame_out1 = frame_out1.data[:, :, ori_top:ori_down, ori_left:ori_right]
#         frame_out2 = frame_out2.data[:, :, ori_top:ori_down, ori_left:ori_right]
#         frame_out3 = frame_out3.data[:, :, ori_top:ori_down, ori_left:ori_right]
#         frame_out4 = frame_out4.data[:, :, ori_top:ori_down, ori_left:ori_right]
#         frame_out5 = frame_out5.data[:, :, ori_top:ori_down, ori_left:ori_right]

#         psnr = max(to_psnr(frame_out1, clean),to_psnr(frame_out2, clean),to_psnr(frame_out3, clean),to_psnr(frame_out4, clean),to_psnr(frame_out5, clean))
#         if psnr == to_psnr(frame_out1, clean):
#             ssim = to_ssim_skimage(frame_out1, clean)
#             frame_out = frame_out1
#         if  psnr == to_psnr(frame_out2, clean):
#             ssim = to_ssim_skimage(frame_out2, clean)
#             frame_out = frame_out2
#         if psnr == to_psnr(frame_out3, clean):
#                 ssim = to_ssim_skimage(frame_out3, clean)
#                 frame_out = frame_out3
#         if psnr == to_psnr(frame_out4, clean):
#                 ssim = to_ssim_skimage(frame_out4, clean)
#                 frame_out = frame_out4
#         if psnr == to_psnr(frame_out5, clean):
#                 ssim = to_ssim_skimage(frame_out5, clean)
#                 frame_out = frame_out5
#         psnr_list.extend(psnr)
#         ssim_list.extend(ssim)


#         imwrite(frame_out, os.path.join(predict_dir, name[0]), range=(0, 1))
#         print(name[0], to_psnr(frame_out, clean), to_ssim_skimage(frame_out, clean))
#         tmp = test_txt.writelines(name[0] + '->\tpsnr: ' + str(to_psnr(frame_out, clean)[0]) + '\tssim:' + str(
#             to_ssim_skimage(frame_out, clean)[0]) + '\n')
#     avr_psnr = sum(psnr_list) / len(psnr_list)
#     avr_ssim = sum(ssim_list) / len(ssim_list)
#     tmp = test_txt.writelines(
#         tag + 'datasets ==>>\tpsnr:' + str(avr_psnr) + '\tssim:' + str(avr_ssim) + '\n')
#     print('dehazed', avr_psnr, avr_ssim)
# # writer.close()
import argparse
import math
import os

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image as imwrite
import time
from thop import profile

from tqdm import tqdm

from Model import Base_Model
from Model_util import padding_image
from make import getTxt
from test_dataset import dehaze_test_dataset
from utils_test import to_psnr, to_ssim_skimage
from My_Model_Z import swin_tiny_patch4_window8_256 as create_model
# --- Parse hyper-parameters train --- #
parser = argparse.ArgumentParser(description='Siamese Dehaze Network')
parser.add_argument('--data_dir', type=str, default='')
parser.add_argument('--model_save_dir', type=str, default='./output_result')
parser.add_argument('--log_dir', type=str, default=None)
parser.add_argument('--gpus', default='0', type=str)
# --- Parse hyper-parameters test --- #
parser.add_argument('--test_dataset', type=str, default='')
parser.add_argument('-test_batch_size', help='Set the testing batch size', default=1, type=int)
parser.add_argument('--test_dir', type=str, default='/home/sunhang/lzm/deHaze/outdoor_Test/')
parser.add_argument('--test_name', type=str, default='hazy,clean')
parser.add_argument('--num', type=str, default='9999999', help='')
parser.add_argument('--use_bn', action='store_true', help='if bs>8 please use bn')
parser.add_argument("--type", default=-1, type=int, help="choose a type 012345")
args = parser.parse_args()

tag = 'outdoor'
if args.type == 0:
    args.test_dir = "../datasets_test/Haze1k/Haze1k_thin/dataset/test/"
    args.test_name = 'input,target'
    tag = 'thin'
elif args.type == 1:
    args.test_dir = "../datasets_test/Haze1k/Haze1k_moderate/dataset/test/"
    args.test_name = 'input,target'
    tag = 'moderate'
elif args.type == 2:
    args.test_dir = "../datasets_test/Haze1k/Haze1k_thick/dataset/test/"
    args.test_name = 'input,target'
    tag = 'thick'
elif args.type == 3:
    args.test_dir = "../datasets_test/Dense_hazy/test/"
    args.test_name = 'hazy,clean'
    tag = 'dense'
elif args.type == 4:
    args.test_dir = "../datasets_test/nhhaze/test/"
    args.test_name = 'hazy,clean'
    tag = 'nhhaze'
elif args.type == 5:
    args.test_dir = "../datasets_test/Outdoor/test/"
    args.test_name = 'hazy,clean'
    tag = 'outdoor'
elif args.type == 6:
    args.test_dir = "/T2020027/zqf/dataset/RSID/RSID/test/"
    args.test_name = 'hazy,clean'
    tag = 'RSID'
elif args.type == 7:
    args.train_dir = '../datasets_train/rice2/train/'
    args.train_name = 'hazy,clean'
    args.test_dir = "../datasets_test/rice2/"
    args.test_name = 'hazy,clean'
    tag = 'rice2'
elif args.type == 8:
    args.train_dir = '../datasets_train/thin_cloudy/train/'
    args.train_name = 'hazy,clear'
    args.test_dir = "../datasets_test/thin_cloudy/"
    args.test_name = 'hazy,clear'
    tag = 'thin_cloudy'
elif args.type == 9:
    args.train_dir = '../datasets_train/LHID/train/'
    args.train_name = 'hazy,clear'
    args.test_dir = "../datasets_test/LHID/"
    args.test_name = 'hazy,clear'
    tag = 'LHID'
elif args.type == 10:
    args.train_dir = '../datasets_train/DHID/train/'
    args.train_name = 'hazy,clear'
    args.test_dir = "../datasets_test/DHID/"
    args.test_name = 'hazy,clear'
    tag = 'DHID'
elif args.type == 11:
    args.train_dir = '../datasets_train/indoor/train/'
    args.train_name = 'hazy,gt'
    args.test_dir = "../datasets_test/indoor/"
    args.test_name = 'hazy,gt'
    tag = 'indoor'

predict_result = os.path.join(args.model_save_dir, tag)
test_batch_size = args.test_batch_size

# --- output picture and check point --- #
if not os.path.exists(args.model_save_dir):
    os.makedirs(args.model_save_dir)

if not os.path.exists(predict_result):
    os.makedirs(predict_result)

output_dir = os.path.join(args.model_save_dir, '')

# --- Gpu device --- #
device_ids = [int(i) for i in list(filter(str.isdigit, args.gpus))]
print('use gpus ->', args.gpus)
device = torch.device("cuda:{}".format(device_ids[0]) if torch.cuda.is_available() else "cpu")

# --- Define the network --- #
if args.use_bn:
    print('we are using BatchNorm')
else:
    print('we are using InstanceNorm')
SDN = create_model()
print('SDN parameters:', sum(param.numel() for param in SDN.parameters()))
# --- Multi-GPU --- #
SDN = SDN.to(device)
SDN = torch.nn.DataParallel(SDN, device_ids=device_ids)
# tensor = (torch.rand(1, 3, 224, 224),).to(device)
# flops, params = profile(SDN, tensor)
# print("FLOPs: ", flops)
# print("%.2fM" % (flops/1e6), "%.2fM" % (params/1e6))

print('We are testing datasets: ', tag)
getTxt(None, None, args.test_dir, args.test_name)
test_hazy, test_gt = args.test_name.split(',')
if not os.path.exists(os.path.join(predict_result, 'hazy')):
    os.makedirs(os.path.join(predict_result, 'hazy'))
os.system('cp -r {}/* {}'.format(os.path.join(args.test_dir, test_hazy), os.path.join(predict_result, 'hazy')))

if not os.path.exists(os.path.join(predict_result, 'clean')):
    os.makedirs(os.path.join(predict_result, 'clean'))
os.system('cp -r {}/* {}'.format(os.path.join(args.test_dir, test_gt), os.path.join(predict_result, 'clean')))

predict_dir = os.path.join(predict_result, tag)
if not os.path.exists(predict_dir):
    os.makedirs(predict_dir)

test_dataset = dehaze_test_dataset(args.test_dir, args.test_name, tag=tag)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

# --- Load the network weight --- #
if args.num != '9999999':
    pkl_list = [i for i in os.listdir(args.model_save_dir) if '.pkl' in i]
    name = [i for i in pkl_list if 'epoch_' + str(args.num) + '_' in i][0]
    SDN.load_state_dict(
        torch.load(os.path.join(args.model_save_dir, name),
                   map_location="cuda:{}".format(device_ids[0])))
    print('--- {} epoch weight loaded ---'.format(args.num))
    start_epoch = int(args.num) + 1
else:
    pkl_list = [i for i in os.listdir(args.model_save_dir) if '.pkl' in i]
    num = sorted([int(i.split('.')[0].split('_')[1]) for i in pkl_list])[-1]
    name = [i for i in pkl_list if 'epoch_' + str(num) + '_' in i][0]
    SDN.load_state_dict(
        torch.load(os.path.join(args.model_save_dir, name),
                   map_location="cuda:{}".format(device_ids[0])))
    print('--- {} epoch weight loaded ---'.format(num))

test_txt = open(os.path.join(predict_result, tag+'_result.txt'), 'w+')

start_time = time.time()
# --- Strat testing --- #
with torch.no_grad():
    img_list = []
    psnr_list = []
    ssim_list = []
    SDN.eval()
    imsave_dir = output_dir
    if not os.path.exists(imsave_dir):
        os.makedirs(imsave_dir)
    for (hazy, clean, name) in test_loader:
        hazy = hazy.to(device)
        clean = clean.to(device)
        h, w = hazy.shape[2], hazy.shape[3]
        max_h = int(math.ceil(h / 4)) * 4
        max_w = int(math.ceil(w / 4)) * 4
        hazy, ori_left, ori_right, ori_top, ori_down = padding_image(hazy, max_h, max_w)

        frame_out = SDN(hazy)

        frame_out = frame_out.data[:, :, ori_top:ori_down, ori_left:ori_right]
        # frame_out_up = SDN(hazy_up)
        # frame_out_down = SDN(hazy_down)
        # frame_out=(torch.cat([frame_out_up.permute(0,2,3,1), frame_out_down[:,:,80:640,:].permute(0,2,3,1)], 1)).permute(0,3,1,2)

        imwrite(frame_out, os.path.join(predict_dir, name[0]), range=(0, 1))

        psnr_list.extend(to_psnr(frame_out, clean))

        ssim_list.extend(to_ssim_skimage(frame_out, clean))
        print(name[0], to_psnr(frame_out, clean), to_ssim_skimage(frame_out, clean))
        tmp = test_txt.writelines(name[0] + '->\tpsnr: ' + str(to_psnr(frame_out, clean)[0]) + '\tssim:' + str(
            to_ssim_skimage(frame_out, clean)[0]) + '\n')
    inference_time = time.time() - start_time
# inference_time1 = time.time() - start_time1


# 输出推理时间
    print("Inference Time: {:.4f} seconds".format(inference_time))
    avr_psnr = sum(psnr_list) / len(psnr_list)
    avr_ssim = sum(ssim_list) / len(ssim_list)
    tmp = test_txt.writelines(
        tag + 'datasets ==>>\tpsnr:' + str(avr_psnr) + '\tssim:' + str(avr_ssim) + '\n')
    print('dehazed', avr_psnr, avr_ssim)

test_txt.close()
# imwrite(img_list[batch_idx], os.path.join(imsave_dir, str(batch_idx) + '.png'))

# writer.close()


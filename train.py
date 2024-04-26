from __future__ import print_function
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.transforms as transforms
from tqdm import tqdm

from data_loader import SYSUData, RegDBData, LLCMData, TestData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb, eval_llcm
from model import embed_net
from utils import *
from loss import OriTripletLoss, CPMLoss
# from tensorboardX import SummaryWriter
from random_erasing import RandomErasing

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='regdb', help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str, help='network baseline:resnet18 or resnet50')
parser.add_argument('--resume', '-r', default='', type=str, help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--model_path', default='save_model/', type=str, help='model save path')
parser.add_argument('--save_epoch', default=20, type=int, metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='log/', type=str, help='log save path')
parser.add_argument('--vis_log_path', default='log/vis_log/', type=str, help='log save path')
parser.add_argument('--workers', default=0, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=144, type=int, metavar='imgw', help='img width')
parser.add_argument('--img_h', default=384, type=int, metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=4, type=int, metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=4, type=int, metavar='tb', help='testing batch size')
parser.add_argument('--margin', default=0.3, type=float, metavar='margin', help='triplet loss margin')
parser.add_argument('--erasing_p', default=0.5, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--num_pos', default=4, type=int, help='num of pos per identity in each modality')
parser.add_argument('--trial', default=2, type=int, metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int, metavar='t', help='random seed')
parser.add_argument('--gpu', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')
parser.add_argument('--lambda_1', default=0.8, type=float, help='lambda_1')
parser.add_argument('--lambda_2', default=0.01, type=float, help='lambda_2')

parser.add_argument("--temperature", type=float, default=0.02,
                    help="initial temperature value, if 0, don't use temperature")

parser.add_argument("--use_sdm", type=bool, default=False,
                    help="if use sdm loss")

parser.add_argument("--sdm_weight", type=float, default=0.1,
                    help="sdm loss weight")

############################# cross module args ####################
parser.add_argument("--cross_moudle", type=bool, default=False,
                    help="sdm loss weight")
parser.add_argument('--vit_dim', default='768', type=int, help='vit_dim')
parser.add_argument('--heads', default=12, type=int, help='heads')
parser.add_argument('--qkv_bias', default=True, type=bool, help='qkv_bias')
parser.add_argument('--cross_depth', default=4, type=int, help='cross_depth')
parser.add_argument('--patch_size', default=16, type=int, help='patch_size')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

set_seed(args.seed)

dataset = args.dataset
if dataset == 'sysu':
    data_path = './Datasets/SYSU-MM01/'
    log_path = args.log_path + 'sysu_log/'
    test_mode = [1, 2]  # thermal to visible
    pool_dim = 2048
elif dataset == 'regdb':
    data_path = './Datasets/RegDB/'
    log_path = args.log_path + 'regdb_log/'
    test_mode = [2, 1]  # visible to thermal
    pool_dim = 1024
elif dataset == 'llcm':
    data_path = './Datasets/LLCM/'
    log_path = args.log_path + 'llcm_log/'
    test_mode = [1, 2]  # [1, 2]: IR to VIS; [2, 1]: VIS to IR;
    pool_dim = 2048

checkpoint_path = args.model_path

if not os.path.isdir(log_path):
    os.makedirs(log_path)
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)
if not os.path.isdir(args.vis_log_path):
    os.makedirs(args.vis_log_path)

suffix = dataset
suffix = suffix + '_deen_p{}_n{}_lr_{}_seed_{}'.format(args.num_pos, args.batch_size, args.lr, args.seed)

if not args.optim == 'sgd':
    suffix = suffix + '_' + args.optim

if dataset == 'regdb':
    suffix = suffix + '_trial_{}'.format(args.trial)

sys.stdout = Logger(log_path + suffix + '_os.txt')

vis_log_dir = args.vis_log_path + suffix + '/'

if not os.path.isdir(vis_log_dir):
    os.makedirs(vis_log_dir)
# writer = SummaryWriter(vis_log_dir)
print("==========\nArgs:{}\n==========".format(args))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_sysu = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Pad(10),
    transforms.RandomGrayscale(p=0.5),
    transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
    RandomErasing(probability=args.erasing_p, sl=0.2, sh=0.8, r1=0.3, mean=[0.485, 0.456, 0.406]),
])
transform_regdb = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Pad(10),
    transforms.RandomGrayscale(p=0.5),
    transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
    RandomErasing(probability=args.erasing_p, sl=0.02, sh=0.4, r1=0.3, mean=[0.485, 0.456, 0.406]),
])
transform_llcm = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Pad(10),
    transforms.RandomGrayscale(p=0.5),
    transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
    RandomErasing(probability=args.erasing_p, sl=0.02, sh=0.4, r1=0.3, mean=[0.485, 0.456, 0.406]),
])
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize,
])

end = time.time()
if dataset == 'sysu':
    # training set
    trainset = SYSUData(data_path, transform=transform_sysu)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0)

elif dataset == 'regdb':
    # training set
    trainset = RegDBData(data_path, args.trial, transform=transform_regdb)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label = process_test_regdb(data_path, trial=args.trial, modal='visible')
    gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modal='thermal')

elif dataset == 'llcm':
    # training set
    trainset = LLCMData(data_path, args.trial, transform=transform_llcm)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label, query_cam = process_query_llcm(data_path, mode=test_mode[1])
    gall_img, gall_label, gall_cam = process_gallery_llcm(data_path, mode=test_mode[0], trial=0)

gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))

# testing data loader
gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

n_class = len(np.unique(trainset.train_color_label))
nquery = len(query_label)
ngall = len(gall_label)

print('Dataset {} statistics:'.format(dataset))
print('  ------------------------------')
print('  subset   | # ids | # images')
print('  ------------------------------')
print('  visible  | {:5d} | {:8d}'.format(n_class, len(trainset.train_color_label)))
print('  thermal  | {:5d} | {:8d}'.format(n_class, len(trainset.train_thermal_label)))
print('  ------------------------------')
print('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))
print('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gall_label)), ngall))
print('  ------------------------------')
print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

print('==> Building model..')
net = embed_net(args, n_class, dataset, arch=args.arch)
net.to(device)
# 打印模型参数量
print_model_parameters_in_million(net)
cudnn.benchmark = True

if len(args.resume) > 0:
    model_path = checkpoint_path + args.resume
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(model_path)
        start_epoch = 0  # checkpoint['epoch']
        net.load_state_dict(checkpoint['net'])
        print('==> loaded checkpoint {} (epoch {})'
              .format(args.resume, checkpoint['epoch']))
    else:
        print('==> no checkpoint found at {}'.format(args.resume))

# define loss function
criterion_id = nn.CrossEntropyLoss()

loader_batch = args.batch_size * args.num_pos
criterion_tri = OriTripletLoss(batch_size=loader_batch, margin=args.margin)
criterion_cpm = CPMLoss(margin=0.2)

criterion_id.to(device)
criterion_tri.to(device)
criterion_cpm.to(device)

if args.optim == 'sgd':
    ignored_params = list(map(id, net.bottleneck.parameters())) \
                     + list(map(id, net.classifier.parameters()))

    base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

    optimizer = optim.SGD([
        {'params': base_params, 'lr': 0.1 * args.lr},
        {'params': net.bottleneck.parameters(), 'lr': args.lr},
        {'params': net.classifier.parameters(), 'lr': args.lr}],
        weight_decay=5e-4, momentum=0.9, nesterov=True)


# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 10:
        lr = args.lr * (epoch + 1) / 10
    elif epoch >= 10 and epoch < 20:
        lr = args.lr
    elif epoch >= 20 and epoch < 80:
        lr = args.lr * 0.1
    elif epoch >= 80:
        lr = args.lr * 0.01
    elif epoch >= 120:
        lr = args.lr * 0.001

    optimizer.param_groups[0]['lr'] = 0.1 * lr
    for i in range(len(optimizer.param_groups) - 1):
        optimizer.param_groups[i + 1]['lr'] = lr

    return lr


def compute_sdm(feature, pid, logit_scale, image_id=None, factor=0.3, epsilon=1e-8):
    """
    Similarity Distribution Matching
    """
    split_size = feature.size(0) // 2

    # 使用切片操作分割张量
    vis_feat = feature[:split_size, :]  # 前半部分，形状为（32，2048）
    ir_feat = feature[split_size:, :]  # 后半部分，形状为（32，2048）
    logit_scale = logit_scale.to("cuda")
    batch_size = vis_feat.shape[0]
    pid = pid.reshape((batch_size, 1)).to("cuda")  # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()
    # 将对角线置为0
    # v2v_i2i_label = labels.fill_diagonal_(0)

    if image_id != None:
        # print("Mix PID and ImageID to create soft label.")
        image_id = image_id.reshape((-1, 1))
        image_id_dist = image_id - image_id.t()
        image_id_mask = (image_id_dist == 0).float()
        labels = (labels - image_id_mask) * factor + image_id_mask
        # labels = (labels + image_id_mask) / 2

    vis_norm = vis_feat / vis_feat.norm(dim=1, keepdim=True)
    ir_norm = ir_feat / ir_feat.norm(dim=1, keepdim=True)

    i2v_cosine_theta = ir_norm @ vis_norm.t()
    v2i_cosine_theta = i2v_cosine_theta.t()
    # i2i_cosine_theta = ir_norm @ ir_norm.t()
    # v2v_cosine_theta = vis_norm @ vis_norm.t()

    ir_proj_vis = logit_scale * i2v_cosine_theta
    vis_proj_ir = logit_scale * v2i_cosine_theta
    # ir_proj_ir = logit_scale * i2i_cosine_theta
    # vis_proj_vis = logit_scale * v2v_cosine_theta

    # normalize the true matching distribution
    labels_distribute = labels / labels.sum(dim=1)
    # v2v_i2i_label_distribute = v2v_i2i_label / v2v_i2i_label.sum(dim=1)

    v2i_pred = F.softmax(vis_proj_ir, dim=1)
    v2i_loss = v2i_pred * (F.log_softmax(vis_proj_ir, dim=1) - torch.log(labels_distribute + epsilon))

    i2v_pred = F.softmax(ir_proj_vis, dim=1)
    i2v_loss = i2v_pred * (F.log_softmax(ir_proj_vis, dim=1) - torch.log(labels_distribute + epsilon))

    # i2i_pred = F.softmax(ir_proj_ir, dim=1)
    # i2i_loss = i2i_pred * (F.log_softmax(ir_proj_vis, dim=1) - torch.log(v2v_i2i_label_distribute + epsilon))
    #
    # v2v_pred = F.softmax(vis_proj_vis, dim=1)
    # v2v_loss = v2v_pred * (F.log_softmax(vis_proj_vis, dim=1) - torch.log(v2v_i2i_label_distribute + epsilon))

    # loss = torch.mean(torch.sum(v2i_loss, dim=1)) + torch.mean(torch.sum(i2v_loss, dim=1)) + torch.mean(torch.sum(v2v_loss, dim=1)) + torch.mean(torch.sum(i2i_loss, dim=1))
    loss = torch.mean(torch.sum(v2i_loss, dim=1)) + torch.mean(torch.sum(i2v_loss, dim=1))

    return loss / 2


def patchify(imgs):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """
    p = args.patch_size  # 16
    assert imgs.shape[2] % p == 0

    h = imgs.shape[2] // p
    w = imgs.shape[3] // p

    x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
    return x


def unpatchify(x, patch_size, img_height, img_width):
    """
    x: Tensor of shape (N, L, patch_size**2 * 3) representing the patches.
    patch_size: The size of each patch (e.g., 16).
    img_height: The height of the original image (before patching).
    img_width: The width of the original image (before patching).

    Returns:
    imgs: Tensor of shape (N, 3, img_height, img_width) representing the original images.
    """
    N, L, _ = x.shape
    H, W = img_height // patch_size, img_width // patch_size  # calculate number of patches along height and width

    # Reshape to prepare for rearrangement
    x = x.reshape(N, H, W, 3, patch_size, patch_size)

    # Rearrange patches to the image grid
    x = torch.einsum('nhwpqc->nchpwq', x)

    # Combine patches into full images
    imgs = x.reshape(N, 3, img_height, img_width)
    return imgs


def train(epoch):
    current_lr = adjust_learning_rate(optimizer, epoch)
    train_loss = AverageMeter()
    id_loss = AverageMeter()
    tri_loss = AverageMeter()
    cpm_loss = AverageMeter()
    ort_loss = AverageMeter()
    sdm_loss = AverageMeter()
    cro_loss = AverageMeter()

    data_time = AverageMeter()
    batch_time = AverageMeter()
    correct = 0
    total = 0

    # switch to train mode
    net.train()
    end = time.time()

    for batch_idx, (input1, input2, label1, label2) in enumerate(trainloader):

        labs = torch.cat((label1, label2, label1, label2), 0)
        labels = torch.cat((label1, label2, label1, label2, label1, label2), 0)

        input1 = Variable(input1.cuda())
        input2 = Variable(input2.cuda())

        labs = Variable(labs.cuda())
        labels = Variable(labels.cuda())
        data_time.update(time.time() - end)

        feat1, out1, loss_ort, cross_x = net(input1, input2)

        loss_id = criterion_id(out1, labels.type(torch.long))

        loss_tri = criterion_tri(feat1, labels.type(torch.long))

        ft1, ft2, ft3 = torch.chunk(feat1, 3, 0)
        loss_cpm = (criterion_cpm(torch.cat((ft1, ft2), 0), labs) + criterion_cpm(torch.cat((ft1, ft3), 0),
                                                                                  labs)) * args.lambda_1
        loss_ort = loss_ort * args.lambda_2
        loss = loss_id + loss_tri + loss_cpm + loss_ort

        if args.use_sdm:
            logit_scale = torch.ones([]) * (1 / args.temperature)

            ft1, ft2, ft3 = torch.chunk(feat1, 3, 0)

            loss_sdm1 = compute_sdm(ft1, label1, logit_scale)
            loss_sdm2 = compute_sdm(ft2, label1, logit_scale)
            loss_sdm3 = compute_sdm(ft3, label1, logit_scale)

            loss_sdm = (loss_sdm1 + loss_sdm2 + loss_sdm3) / 3 * args.sdm_weight
            loss += loss_sdm

        ####################### cross module ###################
        if args.cross_moudle:
            split_size = label1.size(0)
            cross1, cross2, cross3 = torch.chunk(cross_x, 3, 0)
            # 使用切片操作分割张量
            vis_feat = cross1[:split_size, :]
            ir_feat = cross1[split_size:, :]

            cross_ir_feat = net.cross_former(ir_feat, vis_feat, vis_feat)  # q,k,v
            target = patchify(input2)  # ( 1,196,768 ) 只是一个维度变换
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5  # norm_pix_loss

            loss_cro = (cross_ir_feat - target) ** 2  # ( 1, 196, 768 )
            loss_cro = loss_cro.mean(dim=-1)  # [N, L], mean loss per patch   ( 1,196 )

            mask = torch.ones(loss_cro.shape).to('cuda')
            loss_cro = (loss_cro * mask).sum() / mask.sum()  # mean loss on removed patches

            loss += loss_cro

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update P
        train_loss.update(loss.item(), 2 * input1.size(0))
        id_loss.update(loss_id.item(), 2 * input1.size(0))
        tri_loss.update(loss_tri.item(), 2 * input1.size(0))
        cpm_loss.update(loss_cpm.item(), 2 * input1.size(0))
        ort_loss.update(loss_ort.item(), 2 * input1.size(0))
        if args.use_sdm:
            sdm_loss.update(loss_sdm.item(), 2 * input1.size(0))
        if args.cross_moudle:
            cro_loss.update(loss_cro.item(), 2 * input1.size(0))

        total += labels.size(0)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % 50 == 0:
            print('Epoch: [{}][{}/{}] '
                  'Loss:{train_loss.val:.3f} '
                  'iLoss:{id_loss.val:.3f} '
                  'TLoss:{tri_loss.val:.3f} '
                  'CLoss:{cpm_loss.val:.3f} '
                  'OLoss:{ort_loss.val:.3f} '
            .format(
                epoch, batch_idx, len(trainloader),
                train_loss=train_loss, id_loss=id_loss, tri_loss=tri_loss, cpm_loss=cpm_loss, ort_loss=ort_loss,
            ))

    # writer.add_scalar('total_loss', train_loss.avg, epoch)
    # writer.add_scalar('id_loss', id_loss.avg, epoch)
    # writer.add_scalar('tri_loss', tri_loss.avg, epoch)
    # writer.add_scalar('cpm_loss', cpm_loss.avg, epoch)
    # writer.add_scalar('ort_loss', ort_loss.avg, epoch)
    # writer.add_scalar('sdm_loss', sdm_loss.avg, epoch)
    # writer.add_scalar('lr', current_lr, epoch)


def test(epoch):
    # switch to evaluation mode
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat1 = np.zeros((ngall, pool_dim))
    gall_feat2 = np.zeros((ngall, pool_dim))
    gall_feat3 = np.zeros((ngall, pool_dim))
    gall_feat4 = np.zeros((ngall, pool_dim))
    gall_feat5 = np.zeros((ngall, pool_dim))
    gall_feat6 = np.zeros((ngall, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat, feat_att = net(input, input, test_mode[0])
            gall_feat1[ptr:ptr + batch_num, :] = feat[:batch_num].detach().cpu().numpy()
            gall_feat2[ptr:ptr + batch_num, :] = feat_att[:batch_num].detach().cpu().numpy()
            gall_feat3[ptr:ptr + batch_num, :] = feat[batch_num:batch_num * 2].detach().cpu().numpy()
            gall_feat4[ptr:ptr + batch_num, :] = feat_att[batch_num:batch_num * 2].detach().cpu().numpy()
            gall_feat5[ptr:ptr + batch_num, :] = feat[batch_num * 2:].detach().cpu().numpy()
            gall_feat6[ptr:ptr + batch_num, :] = feat_att[batch_num * 2:].detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    # switch to evaluation
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat1 = np.zeros((nquery, pool_dim))
    query_feat2 = np.zeros((nquery, pool_dim))
    query_feat3 = np.zeros((nquery, pool_dim))
    query_feat4 = np.zeros((nquery, pool_dim))
    query_feat5 = np.zeros((nquery, pool_dim))
    query_feat6 = np.zeros((nquery, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat, feat_att = net(input, input, test_mode[1])
            query_feat1[ptr:ptr + batch_num, :] = feat[:batch_num].detach().cpu().numpy()
            query_feat2[ptr:ptr + batch_num, :] = feat_att[:batch_num].detach().cpu().numpy()
            query_feat3[ptr:ptr + batch_num, :] = feat[batch_num:batch_num * 2].detach().cpu().numpy()
            query_feat4[ptr:ptr + batch_num, :] = feat_att[batch_num:batch_num * 2].detach().cpu().numpy()
            query_feat5[ptr:ptr + batch_num, :] = feat[batch_num * 2:].detach().cpu().numpy()
            query_feat6[ptr:ptr + batch_num, :] = feat_att[batch_num * 2:].detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    start = time.time()
    # compute the similarity
    distmat1 = np.matmul(query_feat1, np.transpose(gall_feat1))
    distmat2 = np.matmul(query_feat2, np.transpose(gall_feat2))
    distmat3 = np.matmul(query_feat3, np.transpose(gall_feat3))
    distmat4 = np.matmul(query_feat4, np.transpose(gall_feat4))
    distmat5 = np.matmul(query_feat5, np.transpose(gall_feat5))
    distmat6 = np.matmul(query_feat6, np.transpose(gall_feat6))
    distmat7 = distmat1 + distmat2 + distmat3 + distmat4 + distmat5 + distmat6

    # evaluation
    if dataset == 'regdb':
        cmc1, mAP1, mINP1 = eval_regdb(-distmat1, query_label, gall_label)
        cmc2, mAP2, mINP2 = eval_regdb(-distmat2, query_label, gall_label)
        cmc7, mAP7, mINP7 = eval_regdb(-distmat7, query_label, gall_label)
    elif dataset == 'sysu':
        cmc1, mAP1, mINP1 = eval_sysu(-distmat1, query_label, gall_label, query_cam, gall_cam)
        cmc2, mAP2, mINP2 = eval_sysu(-distmat2, query_label, gall_label, query_cam, gall_cam)
        cmc7, mAP7, mINP7 = eval_sysu(-distmat7, query_label, gall_label, query_cam, gall_cam)
    elif dataset == 'llcm':
        cmc1, mAP1, mINP1 = eval_llcm(-distmat1, query_label, gall_label, query_cam, gall_cam)
        cmc2, mAP2, mINP2 = eval_llcm(-distmat2, query_label, gall_label, query_cam, gall_cam)
        cmc7, mAP7, mINP7 = eval_llcm(-distmat7, query_label, gall_label, query_cam, gall_cam)
    print('Evaluation Time:\t {:.3f}'.format(time.time() - start))

    return cmc1, mAP1, mINP1, cmc2, mAP2, mINP2, cmc7, mAP7, mINP7


# training
print('==> Start Training...')
for epoch in range(start_epoch, 151 - start_epoch):

    print('==> Preparing Data Loader...')
    # identity sampler
    sampler = IdentitySampler(trainset.train_color_label, \
                              trainset.train_thermal_label, color_pos, thermal_pos, args.num_pos, args.batch_size,
                              epoch)

    trainset.cIndex = sampler.index1  # color index
    trainset.tIndex = sampler.index2  # thermal index
    print(epoch)
    print(trainset.cIndex)
    print(trainset.tIndex)

    loader_batch = args.batch_size * args.num_pos

    trainloader = data.DataLoader(trainset, batch_size=loader_batch, \
                                  sampler=sampler, num_workers=args.workers, drop_last=True)

    trainloader = tqdm(trainloader, desc=f'Epoch {epoch}')

    # training
    train(epoch)

    if epoch > 0 and epoch % 2 == 0:
        print('Test Epoch: {}'.format(epoch))

        # testing
        cmc1, mAP1, mINP1, cmc2, mAP2, mINP2, cmc7, mAP7, mINP7 = test(epoch)
        # save model
        if cmc7[0] > best_acc:  # not the real best for sysu-mm01
            best_acc = cmc7[0]
            best_epoch = epoch
            state = {
                'net': net.state_dict(),
                'cmc': cmc7,
                'mAP': mAP7,
                'mINP': mINP7,
                'epoch': epoch,
            }
            torch.save(state, checkpoint_path + suffix + '_best.t')

        print(
            'POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc1[0], cmc1[4], cmc1[9], cmc1[19], mAP1, mINP1))
        print(
            'POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc2[0], cmc2[4], cmc2[9], cmc2[19], mAP2, mINP2))
        print(
            'POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc7[0], cmc7[4], cmc7[9], cmc7[19], mAP7, mINP7))
        print('Best Epoch [{}]'.format(best_epoch))

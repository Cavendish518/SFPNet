# Demo code
import os
import time
import random
import numpy as np
import argparse
import torch
import torch.backends.cudnn as cudnn

import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
import spconv.pytorch as spconv

from tensorboardX import SummaryWriter
from util import config
from util.common_util import AverageMeter, intersectionAndUnionGPU, find_free_port
from Demo.demo_util import collation_fn_voxelmean_tta
from util.logger import get_logger
from util.Mid import Mid
from model.backbone import Semantic as Model

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/demo/Mid_demo.yaml', help='config file')
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    return cfg

def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)

def main_process():
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)

def main():
    args = get_parser()
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False

    if args.multiprocessing_distributed:
        port = find_free_port()
        args.dist_url = f"tcp://127.0.0.1:{port}"
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    global args, best_iou
    args, best_iou = argss, 0
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)


    model = Model(input_c=args.input_c,
        m=args.m,
        classes=args.classes,
        block_reps=args.block_reps,
        layers=args.layers,
        focal_r=args.focal_r,
        focal_th=args.focal_th,
        focal_h=args.focal_h,
        drop_path_rate=args.drop_path_rate,
        grad_checkpoint_layers=args.grad_checkpoint_layers,
        unet_layers=args.unet_layers,
    )

    if main_process():
        global logger, writer
        logger = get_logger(args.save_path)
        writer = SummaryWriter(args.save_path)
        logger.info(args)

    if args.distributed:
        torch.cuda.set_device(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        if args.sync_bn:
            if main_process():
                logger.info("use SyncBN")
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    else:
        model = torch.nn.DataParallel(model.cuda())

    if main_process():
        logger.info("=> creating model ...")
        logger.info("Classes: {}".format(args.classes))
        logger.info(model)
        if args.get("max_grad_norm", None):
            logger.info("args.max_grad_norm = {}".format(args.max_grad_norm))

    
    if args.weight:
        if os.path.isfile(args.weight):
            if main_process():
                logger.info("=> loading weight '{}'".format(args.weight))

            state_dict = torch.load(args.weight)
            model.load_state_dict(state_dict, strict=True)

            if main_process():
                logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            logger.info("=> no weight found at '{}'".format(args.weight))

    args.use_tta = getattr(args, "use_tta", False)
    if args.data_name == 'Mid360':
        val_data = Mid(data_path=args.data_root,
            voxel_size=args.voxel_size,
            split='val',
            rotate_aug=args.use_tta,
            flip_aug=args.use_tta,
            scale_aug=args.use_tta,
            transform_aug=args.use_tta,
            xyz_norm=args.xyz_norm,
            pc_range=args.get("pc_range", None),
            use_tta=args.use_tta,
            vote_num=args.vote_num
        )
    else:
        raise ValueError("The dataset {} is not supported.".format(args.data_name))

    if main_process():
        logger.info("demo_data samples: '{}'".format(len(val_data)))

    if args.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_data, shuffle=False)
    else:
        val_sampler = None
    if getattr(args, "use_tta", False):
        val_loader = torch.utils.data.DataLoader(val_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            sampler=val_sampler,
            collate_fn=collation_fn_voxelmean_tta
        )

    validate_tta(val_loader, model)
    exit()

def validate_tta(val_loader, model):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Quick Demo >>>>>>>>>>>>>>>>')

    batch_time = AverageMeter()
    data_time = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    torch.cuda.empty_cache()

    model.eval()
    end = time.time()
    for i, batch_data_list in enumerate(val_loader):

        data_time.update(time.time() - end)
    
        with torch.no_grad():
            output = 0.0
            for batch_data in batch_data_list:

                (coord, xyz, feat, target, offset, inds_reconstruct) = batch_data

                inds_reconstruct = inds_reconstruct.cuda(non_blocking=True)

                offset_ = offset.clone()
                offset_[1:] = offset_[1:] - offset_[:-1]
                batch = torch.cat([torch.tensor([ii]*o) for ii,o in enumerate(offset_)], 0).long()

                coord = torch.cat([batch.unsqueeze(-1), coord], -1)
                spatial_shape = np.clip((coord.max(0)[0][1:] + 1).numpy(), 128, None)
            
                coord, xyz, feat, target, offset = coord.cuda(non_blocking=True), xyz.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True), offset.cuda(non_blocking=True)
                batch = batch.cuda(non_blocking=True)

                sinput = spconv.SparseConvTensor(feat, coord.int(), spatial_shape, args.batch_size)

                assert batch.shape[0] == feat.shape[0]
                output_i = model(sinput, xyz, batch)
                output_i = F.softmax(output_i[inds_reconstruct, :], -1)
                
                output = output + output_i
            output = output / len(batch_data_list)

        output = output.max(1)[1]

        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Accuracy {accuracy:.4f}.'.format(i + 1, len(val_loader),
                                                          data_time=data_time,
                                                          batch_time=batch_time,
                                                          accuracy=accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)

    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if main_process():
        logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
        for i in range(args.classes):
            logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
        logger.info('<<<<<<<<<<<<<<<<< Demo ends <<<<<<<<<<<<<<<<<')
    
    return mIoU, mAcc, allAcc

if __name__ == '__main__':
    import gc
    gc.collect()
    main()

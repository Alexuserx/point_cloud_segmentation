import numpy as np
import torch.optim as optim
import sklearn.metrics as metrics
import torch.utils.data

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, LinearLR

from dgcnn.model import DGCNN_semseg
from dgcnn.utils import cal_loss, calculate_sem_IoU

def train(
        train_loader: DataLoader, 
        test_loader: DataLoader, 
        args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DGCNN_semseg(args).to(device)
    if args.from_checkpoint:
        model.load_state_dict(torch.load("./models/dgcnn_model_best.t7"))

    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-3)
    elif args.scheduler == 'step':
        scheduler = StepLR(opt, args.step_size, 0.5)
    elif args.scheduler == 'linear':
        scheduler = LinearLR(opt, start_factor=0.5)

    criterion = cal_loss

    weights = np.power(np.amax(args.labelweights) / args.labelweights, 1 / 3.0)
    weights = torch.Tensor(weights).to(device)

    classes = ['background', 'foreground', 'noise']
    class2label = {cls: i for i, cls in enumerate(classes)}
    seg_classes = class2label
    seg_label_to_cat = {}
    for i, cat in enumerate(seg_classes.keys()):
        seg_label_to_cat[i] = cat

    best_test_iou = 0
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_true_cls = []
        train_pred_cls = []
        train_true_seg = []
        train_pred_seg = []
        train_label_seg = []
        for data, seg in tqdm(train_loader, total=len(train_loader), smoothing=0.9):
            data = torch.Tensor(data.data.numpy())
            data, seg = data.float().to(device), seg.long().to(device)
            data, seg = data.to(device), seg.to(device)
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            seg_pred = model(data)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            loss = criterion(seg_pred.view(-1, args.num_classes), seg.view(-1,1).squeeze(), weight=weights)
            loss.backward()
            opt.step()
            pred = seg_pred.max(dim=2)[1]               # (batch_size, num_points)
            count += batch_size
            train_loss += loss.item() * batch_size
            seg_np = seg.cpu().numpy()                  # (batch_size, num_points)
            pred_np = pred.detach().cpu().numpy()       # (batch_size, num_points)
            train_true_cls.append(seg_np.reshape(-1))       # (batch_size * num_points)
            train_pred_cls.append(pred_np.reshape(-1))      # (batch_size * num_points)
            train_true_seg.append(seg_np)
            train_pred_seg.append(pred_np)
        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5
        train_true_cls = np.concatenate(train_true_cls)
        train_pred_cls = np.concatenate(train_pred_cls)
        train_acc = metrics.accuracy_score(train_true_cls, train_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(train_true_cls, train_pred_cls)
        train_true_seg = np.concatenate(train_true_seg, axis=0)
        train_pred_seg = np.concatenate(train_pred_seg, axis=0)
        train_ious = calculate_sem_IoU(train_pred_seg, train_true_seg)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f, train iou: %.6f' % (epoch, 
                                                                                                  train_loss*1.0/count,
                                                                                                  train_acc,
                                                                                                  avg_per_class_acc,
                                                                                                  np.mean(train_ious))
        print(outstr)

        ####################
        # Test
        ####################
        with torch.no_grad():
            test_loss = 0.0
            count = 0.0
            model.eval()
            test_true_cls = []
            test_pred_cls = []
            test_true_seg = []
            test_pred_seg = []
            labelweights = np.zeros(args.num_classes)
            for data, seg in tqdm(test_loader, total=len(test_loader), smoothing=0.9):
                data = torch.Tensor(data.data.numpy())
                data, seg = data.float().to(device), seg.long().to(device)
                data, seg = data.to(device), seg.to(device)
                data = data.permute(0, 2, 1)
                batch_size = data.size()[0]
                seg_pred = model(data)
                seg_pred = seg_pred.permute(0, 2, 1).contiguous()
                loss = criterion(seg_pred.view(-1, args.num_classes), seg.view(-1,1).squeeze(), weight=weights)
                pred = seg_pred.max(dim=2)[1]
                count += batch_size
                test_loss += loss.item() * batch_size
                seg_np = seg.cpu().numpy()
                pred_np = pred.detach().cpu().numpy()
                test_true_cls.append(seg_np.reshape(-1))
                test_pred_cls.append(pred_np.reshape(-1))
                test_true_seg.append(seg_np)
                test_pred_seg.append(pred_np)
                # --------------------------------
                tmp, _ = np.histogram(seg_np, range(args.num_classes + 1))
                labelweights += tmp
                # --------------------------------
            test_true_cls = np.concatenate(test_true_cls)
            test_pred_cls = np.concatenate(test_pred_cls)
            test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
            avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
            test_true_seg = np.concatenate(test_true_seg, axis=0)
            test_pred_seg = np.concatenate(test_pred_seg, axis=0)
            test_ious = calculate_sem_IoU(test_pred_seg, test_true_seg)
            labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
            outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % (epoch,
                                                                                                test_loss*1.0/count,
                                                                                                test_acc,
                                                                                                avg_per_class_acc,
                                                                                                np.mean(test_ious))
            print(outstr)
            if np.mean(test_ious) >= best_test_iou:
                best_test_iou = np.mean(test_ious)
                torch.save(model.state_dict(), './models/dgcnn_model_best.t7')

                # --------------------------------  
                iou_per_class_str = '------- IoU --------\n'
                for l in range(args.num_classes):
                    iou_per_class_str += 'class %s weight: %.3f, IoU: %.3f \n' % (
                        seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])), labelweights[l], test_ious[l])
                print(iou_per_class_str)
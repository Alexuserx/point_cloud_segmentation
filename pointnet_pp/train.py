import torch
import numpy as np
import torch.optim as optim 

from tqdm import tqdm
from pointnet_pp.model import get_model, get_loss


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)

def bn_momentum_adjust(m, momentum):
    if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
        m.momentum = momentum


def train(trainDataLoader, testDataLoader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classifier = get_model(args.IN_CHANNELS, args.NUM_CLASSES).to(device)
    criterion = get_loss().to(device)
    classifier.apply(inplace_relu)

    classifier = classifier.apply(weights_init)

    optimizer = torch.optim.Adam(
                classifier.parameters(),
                lr=args.LEARNING_RATE,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=1e-4
                )

    start_epoch = 0
    global_epoch = 0
    best_iou = 0
    log_string = print
    checkpoints_dir = "./models/"

    if args.from_checkpoint:
        model_path = str(checkpoints_dir) + '/pointnet_pp_model_best.pth'
        classifier.load_state_dict(torch.load(model_path)["model_state_dict"])
        optimizer.load_state_dict(torch.load(model_path)["optimizer_state_dict"])

    classes = ['background', 'foreground', 'noise']
    class2label = {cls: i for i, cls in enumerate(classes)}
    seg_classes = class2label
    seg_label_to_cat = {}
    for i, cat in enumerate(seg_classes.keys()):
        seg_label_to_cat[i] = cat
    
    weights = np.power(np.amax(args.labelweights) / args.labelweights, 1 / 3.0)
    weights = torch.Tensor(weights).to(device)

    for epoch in range(start_epoch, args.N_EPOCHS):
        '''Train on chopped scenes'''
        # log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.N_EPOCHS))
        lr = max(args.LEARNING_RATE * (args.LR_DECAY ** (epoch // args.STEP_SIZE)), args.LEARNING_RATE_CLIP)
        # log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = args.MOMENTUM_ORIGINAL * (args.MOMENTUM_DECCAY ** (epoch // args.MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        # log_string('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        num_batches = len(trainDataLoader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        classifier = classifier.train()

        for i, (points, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            points = points.data.numpy()
            points = torch.Tensor(points)
            points, target = points.float().to(device), target.long().to(device)
            points = points.transpose(2, 1)

            seg_pred, trans_feat = classifier(points)
            seg_pred = seg_pred.contiguous().view(-1, args.NUM_CLASSES)

            batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
            target = target.view(-1, 1)[:, 0]
            loss = criterion(seg_pred, target, trans_feat, weights)
            loss.backward()
            optimizer.step()

            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
            correct = np.sum(pred_choice == batch_label)
            total_correct += correct
            total_seen += (args.BATCH_SIZE * args.NUM_POINT)
            loss_sum += loss
        # log_string('Training mean loss: %f' % (loss_sum / num_batches))
        # log_string('Training accuracy: %f' % (total_correct / float(total_seen)))
            
        outstr = 'Train %d, loss: %.6f, train acc: %.6f' % (epoch, 
                                                            loss_sum / num_batches,
                                                            total_correct / float(total_seen))
        print(outstr)

        # if epoch % 5 == 0:
        #     # log_string('Save model...')
        #     savepath = str(checkpoints_dir) + '/pointnet_pp_model.pth'
        #     # log_string('Saving at %s' % savepath)
        #     state = {
        #         'epoch': epoch,
        #         'model_state_dict': classifier.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #     }
        #     torch.save(state, savepath)
        #     # log_string('Saving model....')

        '''Evaluate on chopped scenes'''
        with torch.no_grad():
            num_batches = len(testDataLoader)
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            labelweights = np.zeros(args.NUM_CLASSES)
            total_seen_class = [0 for _ in range(args.NUM_CLASSES)]
            total_correct_class = [0 for _ in range(args.NUM_CLASSES)]
            total_iou_deno_class = [0 for _ in range(args.NUM_CLASSES)]
            classifier = classifier.eval()

            # log_string('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))
            for i, (points, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                points = points.data.numpy()
                points = torch.Tensor(points)
                points, target = points.float().to(device), target.long().to(device)
                points = points.transpose(2, 1)

                seg_pred, trans_feat = classifier(points)
                pred_val = seg_pred.contiguous().cpu().data.numpy()
                seg_pred = seg_pred.contiguous().view(-1, args.NUM_CLASSES)

                batch_label = target.cpu().data.numpy()
                target = target.view(-1, 1)[:, 0]
                loss = criterion(seg_pred, target, trans_feat, weights)
                loss_sum += loss
                pred_val = np.argmax(pred_val, 2)
                correct = np.sum((pred_val == batch_label))
                total_correct += correct
                total_seen += (args.BATCH_SIZE * args.NUM_POINT)
                tmp, _ = np.histogram(batch_label, range(args.NUM_CLASSES + 1))
                labelweights += tmp

                for l in range(args.NUM_CLASSES):
                    total_seen_class[l] += np.sum((batch_label == l))
                    total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                    total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))

            labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
            mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float32) + 1e-6))
            # log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
            # log_string('eval point avg class IoU: %f' % (mIoU))
            # log_string('eval point accuracy: %f' % (total_correct / float(total_seen)))
            # log_string('eval point avg class acc: %f' % (
            #     np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float32) + 1e-6))))

            iou_per_class_str = '------- IoU --------\n'
            for l in range(args.NUM_CLASSES):
                iou_per_class_str += 'class %s weight: %.3f, IoU: %.3f \n' % (
                    seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])), labelweights[l],
                    total_correct_class[l] / float(total_iou_deno_class[l]))

            # log_string(iou_per_class_str)
            # log_string('Eval mean loss: %f' % (loss_sum / num_batches))
            # log_string('Eval accuracy: %f' % (total_correct / float(total_seen)))
                
            avg_acc = np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float32) + 1e-6))
            outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % (epoch,
                                                                                                  loss_sum / float(num_batches),
                                                                                                  total_correct / float(total_seen),
                                                                                                  avg_acc,
                                                                                                  mIoU)
            print(outstr)

            if mIoU >= best_iou:
                best_iou = mIoU
                # log_string('Save model...')
                savepath = str(checkpoints_dir) + '/pointnet_pp_model_best.pth'
                # log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'class_avg_iou': mIoU,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                print(iou_per_class_str)
                # log_string('Saving model....')
            # log_string('Best mIoU: %f' % best_iou)
        global_epoch += 1
import os
import sys
import time
import random
import string
import argparse
import datetime
import matplotlib.pyplot as plt
import re

import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
import numpy as np

from utils import CTCLabelConverter, CTCLabelConverterForBaiduWarpctc, AttnLabelConverter, Averager
from dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from model import Model
from test import validation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import matplotlib.pyplot as plt
import os
import numpy as np

# def save_augmented_grid(images,labels,output_dir,step):
#     """
#     Cerca coppie di immagini con stessa label partendo dalle prime 5 immagini.
#     Se non trova la coppia, quella riga non viene plottata.
#     """
#     os.makedirs(output_dir, exist_ok=True)
#     images = images.cpu()

#     found_pairs = []

#     used_indices = set()

#     # Step 1: prendi prime 5 immagini
#     for idx_ref in range(2):
#         label_ref = labels[idx_ref]
#         used_indices.add(idx_ref)

#         # Cerca una seconda immagine con stessa label nel batch
#         idx_match = None
#         for idx in range(len(labels)):
#             if idx != idx_ref and labels[idx] == label_ref and idx not in used_indices:
#                 idx_match = idx
#                 used_indices.add(idx)
#                 break

#         if idx_match is not None:
#             found_pairs.append((idx_ref, idx_match))
#         else:
#             print(f" Nessun match trovato per label '{label_ref}', salta la coppia.")

#     if len(found_pairs) == 0:
#         print(" Nessuna coppia trovata, griglia non salvata.")
#         return

#     # Step 2: plot solo le coppie trovate
#     fig, axs = plt.subplots(len(found_pairs), 2, figsize=(8, 3*len(found_pairs)))

#     if len(found_pairs) == 1:
#         axs = np.expand_dims(axs, axis=0)  # per evitare errore con 1 sola riga

#     for i, (idx_ref, idx_match) in enumerate(found_pairs):
#         img_orig = images[idx_ref].squeeze().numpy()
#         img_orig = (img_orig * 255).astype(np.uint8)

#         img_match = images[idx_match].squeeze().numpy()
#         img_match = (img_match * 255).astype(np.uint8)

#         label = labels[idx_ref]

#         axs[i, 0].imshow(img_orig, cmap='gray')
#         axs[i, 0].axis('off')
#         axs[i, 0].set_title(f"Orig: {label}", fontsize=10)

#         axs[i, 1].imshow(img_match, cmap='gray')
#         axs[i, 1].axis('off')
#         axs[i, 1].set_title(f"Match: {label}", fontsize=10)

#     plt.tight_layout()
#     output_path = os.path.join(output_dir, f'step{step}_matched_pairs.png')
#     plt.savefig(output_path)
#     plt.close()
#     print(f' Grid salvata con {len(found_pairs)} coppie: {output_path}')


def train(opt):
    """ dataset preparation """
    if not opt.data_filtering_off:
        print('Filtering the images containing characters which are not in opt.character')
        print('Filtering the images whose label is longer than opt.batch_max_length')
        # see https://github.com/clovaai/deep-text-recognition-benchmark/blob/6593928855fb7abb999a99f428b3e4477d4ae356/dataset.py#L130

    opt.select_data = opt.select_data.split('-')
    opt.batch_ratio = opt.batch_ratio.split('-')
    train_dataset = Batch_Balanced_Dataset(opt)

    log = open(f'./saved_models/{opt.exp_name}/log_dataset.txt', 'a',encoding='utf-8')
    AlignCollate_valid = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD,is_training=True)
    valid_dataset, valid_dataset_log = hierarchical_dataset(root=opt.valid_data, opt=opt)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=opt.batch_size,
        shuffle=True,  # 'True' to check training progress with validation function.
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_valid, pin_memory=True)
    log.write(valid_dataset_log)
    print('-' * 80)
    log.write('-' * 80 + '\n')
    log.close()
    
    """ model configuration """
    if 'CTC' in opt.Prediction:
        if opt.baiduCTC:
            converter = CTCLabelConverterForBaiduWarpctc(opt.character)
        else:
            converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)

    # weight initialization
    for name, param in model.named_parameters():
        if 'localization_fc2' in name:
            print(f'Skip {name} as it is already initialized')
            continue
        try:
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.kaiming_normal_(param)
        except Exception as e:  # for batchnorm.
            if 'weight' in name:
                param.data.fill_(1)
            continue

    # data parallel for multi-GPU
    model = torch.nn.DataParallel(model).to(device)
    model.train()
    if opt.saved_model != '':
        print(f'loading pretrained model from {opt.saved_model}')
        if opt.FT:
            model.load_state_dict(torch.load(opt.saved_model), strict=False)
        else:
            print(f'loading pretrained model from {opt.saved_model}')
            state_dict = torch.load(opt.saved_model, map_location=device)
            model_state_dict = model.state_dict()

            for name, param in state_dict.items():
                if name in model_state_dict:
                    if model_state_dict[name].shape == param.shape:
                        model_state_dict[name].copy_(param)
                    else:
                        print(f'Skip {name} due to shape mismatch')
            model.load_state_dict(model_state_dict)

            # model.load_state_dict(torch.load(opt.saved_model))
    print("Model:")
    print(model)

    """ setup loss """
    if 'CTC' in opt.Prediction:
        if opt.baiduCTC:
            # need to install warpctc. see our guideline.
            from warpctc_pytorch import CTCLoss 
            criterion = CTCLoss()
        else:
            criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0
    # loss averager
    loss_avg = Averager()
    train_losses = []
    valid_losses = []
    epochs = []
    iters_per_epoch = 100000/opt.batch_size

    # filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print('Trainable params num : ', sum(params_num))
    # [print(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, model.named_parameters())]

    # setup optimizer
    if opt.adam:
        optimizer = optim.Adam(filtered_parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
    else:
        optimizer = optim.Adadelta(filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)
    print("Optimizer:")
    print(optimizer)

    """ final options """
    # print(opt) nel log:
    with open(f'./saved_models/{opt.exp_name}/opt.txt', 'a', encoding='utf-8') as opt_file:
        opt_log = '------------ Options -------------\n'
        args = vars(opt)
        for k, v in args.items():
            opt_log += f'{str(k)}: {str(v)}\n'
        opt_log += '---------------------------------------\n'
        print(opt_log)
        opt_file.write(opt_log)

    """ start training """
    start_iter = 0
    if opt.saved_model != '':
        try:
            start_iter = int(opt.saved_model.split('_')[-1].split('.')[0])
            print(f'continue to train, start_iter: {start_iter}')
        except:
            pass

    start_time = time.time()
    best_accuracy = -1
    best_norm_ED = -1
    iteration = start_iter

    while(True):
        # salvataggio di data e ora dell'allenamento:
        if iteration == start_iter:
            with open(f'./saved_models/{opt.exp_name}/log_train.txt', 'a', encoding='utf-8') as log:
                log.write(f'Training started at: {opt.start_time}\n')
                log.write(f'Expected total epochs: {opt.num_epochs:.2f}\n')

        # train part
        image_tensors, labels = train_dataset.get_batch()
        image = image_tensors.to(device)
        text, length = converter.encode(labels, batch_max_length=opt.batch_max_length)
        batch_size = image.size(0)

        if 'CTC' in opt.Prediction:
            preds = model(image, text)
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            if opt.baiduCTC:
                preds = preds.permute(1, 0, 2)  # to use CTCLoss format
                cost = criterion(preds, text, preds_size, length) / batch_size
            else:
                preds = preds.log_softmax(2).permute(1, 0, 2)
                cost = criterion(preds, text, preds_size, length)

        else:
            preds = model(image, text[:, :-1])  # align with Attention.forward
            target = text[:, 1:]  # without [GO] Symbol
            cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))

        model.zero_grad()
        cost.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)
        optimizer.step()

        loss_avg.add(cost)

        # validation part
        if (iteration + 1) % opt.valInterval == 0 or iteration == 0: # To see training progress, we also conduct validation when 'iteration == 0' 
            elapsed_time = time.time() - start_time
            # for log
            with open(f'./saved_models/{opt.exp_name}/log_train.txt', 'a', encoding='utf-8') as log:
                model.eval()
                with torch.no_grad():
                    valid_loss, current_accuracy, current_norm_ED, preds, confidence_score, labels, infer_time, length_of_data = validation(
                        model, criterion, valid_loader, converter, opt)
                model.train()

                # training loss and validation loss
                loss_log = f'[{iteration+1}/{opt.num_iter}] Train loss: {loss_avg.val():0.5f}, Valid loss: {valid_loss:0.5f}, Elapsed_time: {elapsed_time:0.5f}'
                train_losses.append(loss_avg.val())
                valid_losses.append(valid_loss)
                epochs.append((iteration + 1) / iters_per_epoch)

                loss_avg.reset()

                current_model_log = f'{"Current_accuracy":17s}: {current_accuracy:0.3f}, {"Current_norm_ED":17s}: {current_norm_ED:0.2f}'

                # keep best accuracy model (on valid dataset)
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    torch.save(model.state_dict(), f'./saved_models/{opt.exp_name}/best_accuracy.pth')
                if current_norm_ED > best_norm_ED:
                    best_norm_ED = current_norm_ED
                    torch.save(model.state_dict(), f'./saved_models/{opt.exp_name}/best_norm_ED.pth')
                best_model_log = f'{"Best_accuracy":17s}: {best_accuracy:0.3f}, {"Best_norm_ED":17s}: {best_norm_ED:0.2f}'

                loss_model_log = f'{loss_log}\n{current_model_log}\n{best_model_log}'
                print(loss_model_log)
                log.write(loss_model_log + '\n')

                # show some predicted results
                dashed_line = '-' * 80
                head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
                predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
                for gt, pred, confidence in zip(labels[:5], preds[:5], confidence_score[:5]):
                    if 'Attn' in opt.Prediction:
                        gt = gt[:gt.find('[s]')]
                        pred = pred[:pred.find('[s]')]

                    predicted_result_log += f'{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n'
                predicted_result_log += f'{dashed_line}'
                print(predicted_result_log)
                log.write(predicted_result_log + '\n')
        output_dir = os.path.join('./saved_models/', opt.exp_name)
       #save_augmented_grid(image_tensors[:2],labels[:2],output_dir,iteration+1)

        # save model per 1e+5 iter.
        if (iteration + 1) % 1e+5 == 0:
            torch.save(
                model.state_dict(), f'./saved_models/{opt.exp_name}/iter_{iteration+1}.pth'
    )
        # Salva il numero di iterazione cumulativo
            with open(f'./saved_models/{opt.exp_name}/last_iter.txt', 'w') as f:
                f.write(str(iteration + 1))
        
        if (iteration + 1) == opt.num_iter:
            # Salva anche qui!
            with open(f'./saved_models/{opt.exp_name}/last_iter.txt', 'w') as f:
                f.write(str(iteration + 1))

            # plot delle loss:
            # Parametri dataset
            dataset_size = 100000
            batch_size = opt.batch_size
            iters_per_epoch = dataset_size / batch_size

            # Percorsi
            log_file = f'./saved_models/{opt.exp_name}/log_train.txt'
            iter_file = f'./saved_models/{opt.exp_name}/last_iter.txt'

            # Leggi iterazioni precedenti se esistono
            if os.path.exists(iter_file):
                with open(iter_file, 'r') as f:
                    previous_iter = int(f.read())
                    print(f' Iterazioni precedenti trovate: {previous_iter}')
            else:
                previous_iter = 0
                print(' Nessun last_iter.txt trovato: si parte da 0.')

            # Estrai dati dal log
            iterations = []
            train_loss = []
            valid_loss = []

            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    match = re.search(r'\[(\d+)/', line)
                    if match:
                        iter_num = int(match.group(1))
                        train_match = re.search(r'Train loss: ([0-9.]+)', line)
                        valid_match = re.search(r'Valid loss: ([0-9.]+)', line)
                        if train_match and valid_match:
                            iterations.append(iter_num)
                            train_loss.append(float(train_match.group(1)))
                            valid_loss.append(float(valid_match.group(1)))

            # Iterazioni cumulative
            iterations_cumulative = [previous_iter + it for it in iterations]

            # Epoche cumulative
            epochs = [it / iters_per_epoch for it in iterations_cumulative]

            sys.exit()

        iteration += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', help='Where to store logs and models')
    parser.add_argument('--train_data', required=True, help='path to training dataset')
    parser.add_argument('--valid_data', required=True, help='path to validation dataset')
    parser.add_argument('--manualSeed', type=int, default=1111, help='for random seed setting')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=48, help='input batch size')
    parser.add_argument('--num_iter', type=int, default=300000, help='number of iterations to train for')
    parser.add_argument('--valInterval', type=int, default=2000, help='Interval between each validation')
    parser.add_argument('--saved_model', default='', help="path to model to continue training")
    parser.add_argument('--FT', action='store_true', help='whether to do fine-tuning')
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is Adadelta)')
    parser.add_argument('--lr', type=float, default=1, help='learning rate, default=1.0 for Adadelta')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--rho', type=float, default=0.95, help='decay rate rho for Adadelta. default=0.95')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps for Adadelta. default=1e-8')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping value. default=5')
    parser.add_argument('--baiduCTC', action='store_true', help='for data_filtering_off mode')
    """ Data processing """
    parser.add_argument('--select_data', type=str, default='/', 
                     help='select training data (default is MJ-ST, which means MJ and ST used as training data)') 
    parser.add_argument('--batch_ratio', type=str, default='1', 
                     help='assign ratio for each selected data in the batch') 
    parser.add_argument('--total_data_usage_ratio', type=str, default='1.0',
                        help='total data usage ratio, this ratio is multiplied to total number of data.')
    parser.add_argument('--batch_max_length', type=int, default=7, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str,
                        default='0123456789ABCDEFGHIJKLMNPQRSTUVWXYZ皖沪津渝冀晋蒙辽吉黑苏浙京闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新警学O', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    parser.add_argument('--data_filtering_off', action='store_true', help='for data_filtering_off mode')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=True,
                        help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1,
                        help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    opt = parser.parse_args()
    opt.is_training = True
    # Aggiungi data e ora di avvio
    start_time = datetime.datetime.now()
    opt.start_time = start_time.strftime("%Y-%m-%d %H:%M:%S")

    # Calcola numero di epoche stimate
    images_per_epoch = 100000
    batches_per_epoch = images_per_epoch / opt.batch_size
    opt.num_epochs = opt.num_iter / batches_per_epoch

    if not opt.exp_name:
        opt.exp_name = f'{opt.Transformation}-{opt.FeatureExtraction}-{opt.SequenceModeling}-{opt.Prediction}'
        opt.exp_name += f'-Seed{opt.manualSeed}'
        # print(opt.exp_name)

    os.makedirs(f'./saved_models/{opt.exp_name}', exist_ok=True)

    """ vocab / character number configuration """
    if opt.sensitive:
        # opt.character += 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    """ Seed and GPU setting """
    # print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed(opt.manualSeed)

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()
    # print('device count', opt.num_gpu)
    if opt.num_gpu > 1:
        print('------ Use multi-GPU setting ------')
        print('if you stuck too long time with multi-GPU setting, try to set --workers 0')
        # check multi-GPU issue https://github.com/clovaai/deep-text-recognition-benchmark/issues/1
        opt.workers = opt.workers * opt.num_gpu
        opt.batch_size = opt.batch_size * opt.num_gpu

        """ previous version
        print('To equlize batch stats to 1-GPU setting, the batch_size is multiplied with num_gpu and multiplied batch_size is ', opt.batch_size)
        opt.batch_size = opt.batch_size * opt.num_gpu
        print('To equalize the number of epochs to 1-GPU setting, num_iter is divided with num_gpu by default.')
        If you dont care about it, just commnet out these line.)
        opt.num_iter = int(opt.num_iter / opt.num_gpu)
        """

    train(opt)
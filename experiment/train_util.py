#coding=utf8
from __future__ import division
import torch
import os,time,datetime
from torch.autograd import Variable
import logging
import numpy as np
from math import ceil
from sklearn.metrics import mean_squared_error

def dt():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def trainlog(logfilepath, head='%(message)s'):
    logger = logging.getLogger('mylogger')
    logging.basicConfig(filename=logfilepath, level=logging.INFO, format=head)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(head)
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def train(model,
          epoch_num,
          start_epoch,
          optimizer,
          criterion,
          exp_lr_scheduler,
          data_set,
          data_loader,
          save_dir,
          print_inter=200,
          val_inter=3500
          ):

    step = -1
    for epoch in range(start_epoch,epoch_num):
        # train phase
        exp_lr_scheduler.step(epoch)
        model.train(True)  # Set model to training mode

        for batch_cnt, data in enumerate(data_loader['train']):

            step+=1
            model.train(True)
            # print data
            inputs, labels, length = data

            inputs = Variable(inputs.cuda())
            labels = Variable(torch.from_numpy(np.array(labels)).float().cuda())

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = model(inputs, None)
            loss = criterion(outputs, labels)

            # _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()



            # batch loss
            if step % print_inter == 0:
                labels = labels.squeeze().cpu().numpy()
                outputs = outputs.squeeze().data.cpu().numpy()
                ture_label = np.array([]); ture_outputs = np.array([])
                for i in range(labels.shape[0]):
                    ture_label = np.append(ture_label, labels[i][:length[i]])
                    ture_outputs = np.append(ture_outputs, outputs[i][:length[i]])
                mse = mean_squared_error(ture_label, ture_outputs)
                logging.info('%s [%d-%d] | batch-loss: %.3f | mse@1: %.3f'
                             % (dt(), epoch, batch_cnt, loss.data.cpu().numpy(), mse))


            if step % val_inter == 0:
                logging.info('current lr:%s' % exp_lr_scheduler.get_lr())
                # val phase
                model.train(False)  # Set model to evaluate mode

                val_loss = 0
                val_size = ceil(len(data_set['val']) / data_loader['val'].batch_size)

                t0 = time.time()
                ture_label=np.array([])
                ture_outputs = np.array([])

                for batch_cnt_val, data_val in enumerate(data_loader['val']):
                    # print data
                    inputs,  labels, length = data_val

                    inputs = Variable(inputs.cuda())
                    labels = Variable(torch.from_numpy(np.array(labels)).float().cuda())

                    # forward
                    outputs = model(inputs, None)

                    loss = criterion(outputs, labels)

                    # statistics
                    labels = labels.squeeze().cpu().numpy()
                    outputs = outputs.squeeze().data.cpu().numpy()
                    ture_label = np.array([]);
                    ture_outputs = np.array([])
                    for i in range(labels.shape[0]):
                        ture_label = np.append(ture_label, labels[i][:length[i]])
                        ture_outputs = np.append(ture_outputs, outputs[i][:length[i]])
                    val_loss += loss.data.cpu().numpy()

                val_loss = val_loss / val_size
                val_mse = mean_squared_error(ture_label, ture_outputs)

                t1 = time.time()
                since = t1-t0
                logging.info('--'*30)
                logging.info('current lr:%s' % exp_lr_scheduler.get_lr())

                logging.info('%s epoch[%d]-val-loss: %.4f ||val-mse@1: %.4f ||time: %d'
                             % (dt(), epoch, val_loss, val_mse, since))

                # save model
                save_path = os.path.join(save_dir,
                        'weights-%d-%d-[%.4f].pth'%(epoch,batch_cnt,val_mse))
                torch.save(model.state_dict(), save_path)
                logging.info('saved model to %s' % (save_path))
                logging.info('--' * 30)



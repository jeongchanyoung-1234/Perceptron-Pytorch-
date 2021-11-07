import time

import torch
import torch.optim as optim


optimizer_map = {'adam': optim.Adam,
                 'sgd': optim.SGD,
                 'rmsprop': optim.RMSprop}


class IrisTrainer:
    def __init__(self,
                 config,
                 model,
                 optimizer,
                 criterion,
                 train_dataloader,
                 valid_dataloader):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        self.best_loss = float('inf')
        self.best_acc = 0.
        self.best_epoch = 0.

    def train(self):
        total_train_loss = 0.
        total_train_acc = 0.
        train_cnt = 0

        total_valid_loss = 0.
        total_valid_acc = 0.
        valid_cnt = 0

        start = time.time()
        for epoch in range(self.config.epochs):
            # train_loop
            for i, (batch_x, batch_y) in enumerate(self.train_dataloader) :
                batch_y_hat = self.model(batch_x.float())
                loss = self.criterion(batch_y_hat, batch_y.long())
                acc = (batch_y_hat.argmax(1) == batch_y).sum().item() / len(batch_y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_train_loss += loss
                total_train_acc += acc
                train_cnt += 1

            train_loss = total_train_loss / train_cnt
            train_acc = total_train_acc / train_cnt

            # valid_loop
            for i, (batch_x, batch_y) in enumerate(self.valid_dataloader) :
                with torch.no_grad() :
                    batch_y_hat = self.model(batch_x.float())
                    loss = self.criterion(batch_y_hat, batch_y.long())
                    acc = (batch_y_hat.argmax(1) == batch_y).sum().item() / len(batch_y)

                total_valid_loss += loss
                total_valid_acc += acc
                valid_cnt += 1

            valid_loss = total_valid_loss / valid_cnt
            valid_acc = total_valid_acc / valid_cnt

            if valid_acc > self.best_acc:
                self.best_epoch = epoch + 1
                self.best_loss = valid_loss
                self.best_acc = valid_acc


            if (epoch + 1) % self.config.verbose == 0:
                print('|EPOCH ({}/{})| train_loss={:.4f} train_acc={:3.2f} valid_loss={:.4f} valid_acc={:3.2f}  ({:2.2f}sec)'.format(
                    epoch + 1,
                    self.config.epochs,
                    train_loss,
                    train_acc * 100,
                    valid_loss,
                    valid_acc * 100,
                    time.time() - start
                ))

            total_train_loss, total_train_acc, train_cnt, total_valid_loss, total_valid_acc, valid_cnt = 0, 0, 0, 0, 0, 0

        print()
        print('|Training completed| Best loss={:.4f}, Best Accuracy={:2.2f}% ({}epoch)'.format(
            self.best_loss, self.best_acc * 100, self.best_epoch
        ))






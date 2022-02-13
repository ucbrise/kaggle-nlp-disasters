import flor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torchtext.legacy.data import Field, TabularDataset, BucketIterator
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import flor
from multiprocessing import set_start_method
from utils import CLR_Scheduler
try:
    set_start_method('spawn')
except RuntimeError:
    pass
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
flor.namespace_stack.test_force(device, 'device')
device
label_field = Field(sequential=False, use_vocab=False, batch_first=True,
    dtype=torch.float)
flor.namespace_stack.test_force(label_field, 'label_field')
text_field = Field(tokenize='spacy', lower=True, include_lengths=True,
    batch_first=True)
flor.namespace_stack.test_force(text_field, 'text_field')
fields = [('words', text_field), ('target', label_field)]
flor.namespace_stack.test_force(fields, 'fields')
fields_test = [('words', text_field)]
flor.namespace_stack.test_force(fields_test, 'fields_test')
train, valid = TabularDataset.splits(path='data', train='train_rnn.csv',
    validation='valid_rnn.csv', format='CSV', fields=fields, skip_header=True)
flor.namespace_stack.test_force(train, 'train')
flor.namespace_stack.test_force(valid, 'valid')
test = TabularDataset(path='data/test_rnn.csv', format='CSV', fields=
    fields_test, skip_header=True)
flor.namespace_stack.test_force(test, 'test')
train_iter = BucketIterator(train, batch_size=200, sort_key=lambda x: len(x
    .words), device=device, sort=True, sort_within_batch=True)
flor.namespace_stack.test_force(train_iter, 'train_iter')
valid_iter = BucketIterator(valid, batch_size=200, sort_key=lambda x: len(x
    .words), device=device, sort=True, sort_within_batch=True)
flor.namespace_stack.test_force(valid_iter, 'valid_iter')
test_iter = BucketIterator(test, batch_size=200, sort_key=lambda x: len(x.
    words), device=device, sort=True, sort_within_batch=True)
flor.namespace_stack.test_force(test_iter, 'test_iter')
text_field.build_vocab(train, min_freq=5)


class LSTM(nn.Module):

    def __init__(self, dimension=128):
        try:
            flor.namespace_stack.new()
            super(LSTM, self).__init__()
            self.embedding = nn.Embedding(len(text_field.vocab), dimension)
            flor.namespace_stack.test_force(self.embedding, 'self.embedding')
            self.lstm = nn.LSTM(input_size=dimension, hidden_size=dimension,
                num_layers=1, batch_first=True, bidirectional=True)
            flor.namespace_stack.test_force(self.lstm, 'self.lstm')
            self.drop = nn.Dropout(p=0.85)
            flor.namespace_stack.test_force(self.drop, 'self.drop')
            self.dimension = dimension
            flor.namespace_stack.test_force(self.dimension, 'self.dimension')
            self.fc = nn.Linear(2 * dimension, 1)
            flor.namespace_stack.test_force(self.fc, 'self.fc')
            self.relu = nn.ReLU()
            flor.namespace_stack.test_force(self.relu, 'self.relu')
        finally:
            flor.namespace_stack.pop()

    def forward(self, text, text_len):
        try:
            flor.namespace_stack.new()
            text_emb = self.relu(self.embedding(text))
            flor.namespace_stack.test_force(text_emb, 'text_emb')
            packed_input = pack_padded_sequence(text_emb, text_len,
                batch_first=True, enforce_sorted=False)
            flor.namespace_stack.test_force(packed_input, 'packed_input')
            packed_output, _ = self.lstm(packed_input)
            flor.namespace_stack.test_force(packed_output, 'packed_output')
            flor.namespace_stack.test_force(_, '_')
            output, _ = pad_packed_sequence(packed_output, batch_first=True)
            flor.namespace_stack.test_force(output, 'output')
            flor.namespace_stack.test_force(_, '_')
            out_forward = output[(range(len(output))), (text_len - 1), :
                self.dimension]
            flor.namespace_stack.test_force(out_forward, 'out_forward')
            out_reverse = output[:, (0), self.dimension:]
            flor.namespace_stack.test_force(out_reverse, 'out_reverse')
            out_reduced = torch.cat((out_forward, out_reverse), 1)
            flor.namespace_stack.test_force(out_reduced, 'out_reduced')
            text_fea = out_reduced
            flor.namespace_stack.test_force(text_fea, 'text_fea')
            text_fea = self.fc(self.drop(text_fea))
            flor.namespace_stack.test_force(text_fea, 'text_fea')
            text_fea = torch.squeeze(text_fea, 1)
            flor.namespace_stack.test_force(text_fea, 'text_fea')
            text_out = torch.sigmoid(text_fea)
            flor.namespace_stack.test_force(text_out, 'text_out')
            return text_out
        finally:
            flor.namespace_stack.pop()


def train(model, optimizer, criterion=nn.BCELoss(), train_loader=train_iter,
    valid_loader=valid_iter, test_loader=test_iter, num_epochs=5,
    eval_every=len(train_iter) // 2, file_path='training_process',
    best_valid_loss=float('Inf')):
    try:
        flor.namespace_stack.new()
        running_loss = 0.0
        flor.namespace_stack.test_force(running_loss, 'running_loss')
        valid_running_loss = 0.0
        flor.namespace_stack.test_force(valid_running_loss,
            'valid_running_loss')
        global_step = 0
        flor.namespace_stack.test_force(global_step, 'global_step')
        train_loss_list = []
        flor.namespace_stack.test_force(train_loss_list, 'train_loss_list')
        valid_loss_list = []
        flor.namespace_stack.test_force(valid_loss_list, 'valid_loss_list')
        global_steps_list = []
        flor.namespace_stack.test_force(global_steps_list, 'global_steps_list')
        best_loss = float('inf')
        flor.namespace_stack.test_force(best_loss, 'best_loss')
        model.train()
        flor.skip_stack.new(2)
        if flor.skip_stack.peek().should_execute(not flor.SKIP):
            for epoch in flor.it(range(num_epochs)):
                flor.log('learning_rate', str(optimizer.param_groups[0]['lr']))
                if flor.SkipBlock.step_into('batchwise-loop'):
                    flor.skip_stack.new(1)
                    if flor.skip_stack.peek().should_execute(not flor.SKIP):
                        for ((words, words_len), labels), _ in train_loader:
                            labels = labels.to(device)
                            flor.namespace_stack.test_force(labels, 'labels')
                            words = words.to(device)
                            flor.namespace_stack.test_force(words, 'words')
                            words_len = words_len.detach().cpu()
                            flor.namespace_stack.test_force(words_len,
                                'words_len')
                            output = model(words, words_len)
                            flor.namespace_stack.test_force(output, 'output')
                            loss = criterion(output, labels)
                            flor.namespace_stack.test_force(loss, 'loss')
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            running_loss += loss.item()
                            global_step += 1
                            if global_step % eval_every == 0:
                                model.eval()
                                with torch.no_grad():
                                    flor.skip_stack.new(0)
                                    if flor.skip_stack.peek().should_execute(
                                        not flor.SKIP):
                                        for ((words, words_len), labels
                                            ), _ in valid_loader:
                                            labels = labels.to(device)
                                            flor.namespace_stack.test_force(labels,
                                                'labels')
                                            words = words.to(device)
                                            flor.namespace_stack.test_force(words,
                                                'words')
                                            words_len = words_len.detach().cpu()
                                            flor.namespace_stack.test_force(words_len,
                                                'words_len')
                                            output = model(words, words_len)
                                            flor.namespace_stack.test_force(output,
                                                'output')
                                            loss = criterion(output, labels)
                                            flor.namespace_stack.test_force(loss,
                                                'loss')
                                            valid_running_loss += float(loss.item())
                                    (valid_running_loss, _, _, words_len,
                                        output, loss) = (flor.skip_stack.
                                        pop().proc_side_effects(
                                        valid_running_loss, labels, words,
                                        words_len, output, loss))
                                average_train_loss = running_loss / eval_every
                                flor.namespace_stack.test_force(
                                    average_train_loss, 'average_train_loss')
                                average_valid_loss = valid_running_loss / len(
                                    valid_loader)
                                flor.namespace_stack.test_force(
                                    average_valid_loss, 'average_valid_loss')
                                if average_valid_loss < best_loss:
                                    best_loss = average_valid_loss
                                    flor.namespace_stack.test_force(best_loss,
                                        'best_loss')
                                    torch.save(model.state_dict(),
                                        'best-model.pt')
                                train_loss_list.append(average_train_loss)
                                valid_loss_list.append(average_valid_loss)
                                global_steps_list.append(global_step)
                                running_loss = 0.0
                                flor.namespace_stack.test_force(running_loss,
                                    'running_loss')
                                valid_running_loss = 0.0
                                flor.namespace_stack.test_force(
                                    valid_running_loss, 'valid_running_loss')
                                model.train()
                                print(
                                    'Epoch [{}/{}], LR: {:.3f}, Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                                    .format(epoch + 1, num_epochs,
                                    optimizer.param_groups[0]['lr'],
                                    global_step, num_epochs * len(
                                    train_loader), average_train_loss,
                                    average_valid_loss))
                                flor.log('avg_train_loss', average_train_loss)
                                flor.log('average_valid_loss',
                                    average_valid_loss)
                            clr_scheduler.step()
                    (_, _, running_loss, valid_running_loss, global_step, _,
                        _, _, best_loss, _, _, _) = (flor.skip_stack.pop().
                        proc_side_effects(model, optimizer, running_loss,
                        valid_running_loss, global_step, train_loss_list,
                        valid_loss_list, global_steps_list, best_loss,
                        torch, flor, clr_scheduler))
                flor.SkipBlock.end(model, optimizer, clr_scheduler)
        (_, _, running_loss, valid_running_loss, global_step, _, _, _,
            best_loss, _, _, _) = (flor.skip_stack.pop().proc_side_effects(
            model, optimizer, running_loss, valid_running_loss, global_step,
            train_loss_list, valid_loss_list, global_steps_list, best_loss,
            flor, torch, clr_scheduler))
        y_pred = []
        flor.namespace_stack.test_force(y_pred, 'y_pred')
        model.eval()
        with torch.no_grad():
            flor.skip_stack.new(3)
            if flor.skip_stack.peek().should_execute(not flor.SKIP):
                for (words, words_len), _ in test_loader:
                    words = words.to(device)
                    flor.namespace_stack.test_force(words, 'words')
                    words_len = words_len.detach().cpu()
                    flor.namespace_stack.test_force(words_len, 'words_len')
                    output = model(words, words_len)
                    flor.namespace_stack.test_force(output, 'output')
                    output = (output > 0.5).int()
                    flor.namespace_stack.test_force(output, 'output')
                    y_pred.extend(output.tolist())
            _, words_len, output, _ = flor.skip_stack.pop().proc_side_effects(
                words, words_len, output, y_pred)
        print('Finished Training!')
        return y_pred
    finally:
        flor.namespace_stack.pop()


EPOCHS = 80
flor.namespace_stack.test_force(EPOCHS, 'EPOCHS')
MIN_LR = 0.0001
flor.namespace_stack.test_force(MIN_LR, 'MIN_LR')
model = LSTM(8).to(device)
flor.namespace_stack.test_force(model, 'model')
optimizer = optim.SGD(model.parameters(), lr=MIN_LR)
flor.namespace_stack.test_force(optimizer, 'optimizer')
flor.log('optimizer', str(type(optimizer)))
clr_scheduler = CLR_Scheduler(optimizer, net_steps=len(train_iter) * EPOCHS,
    min_lr=MIN_LR, max_lr=4.0, tail_frac=0.0)
flor.namespace_stack.test_force(clr_scheduler, 'clr_scheduler')
pred = train(model=model, optimizer=optimizer, num_epochs=EPOCHS)
flor.namespace_stack.test_force(pred, 'pred')
if not flor.SKIP:
    flor.flush()

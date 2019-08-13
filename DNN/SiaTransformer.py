import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm
import math,time

import codecs, json, os

import pandas as pd

from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

//数据装载代码部分
class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_ifrn_examples(self, text_list):
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_json(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        dicts = []
        with codecs.open(input_file, 'r', 'utf-8') as infs:
            for inf in infs:
                inf = inf.strip()
                dicts.append(json.loads(inf))
        return dicts

    @classmethod
    def _read_csv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        file_in = pd.read_csv(input_file, encoding='utf-8')['0']
        lines = []
        for line in file_in:
            lines.append(line.split(","))
        return lines



class SiameseProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "train.csv")), 'train')

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "valid.csv")), 'dev')

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "test.csv")), 'test')

    def get_labels(self):
        """See base class."""
        return [0,1]

    def _create_examples(self, dicts, set_type):
        examples = []
        for (i, infor) in enumerate(dicts):
            guid = "%s-%s" % (set_type, i)
            text_a = int(infor[0])
            text_b = int(infor[1])
            label = int(infor[2])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b,label=label))
        return examples

def get_class(task_name):
    """ Mapping from task string to Dataset Class """
    processors = {"siamese-model": SiameseProcessor}
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    return processors[task_name]


//模型代码部分
class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout, is_cuda=True):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        if is_cuda:
            self.cuda()
            self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden).cuda()
            self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout).cuda()
            self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout).cuda()
            self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout).cuda()
        else:
            self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
            self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
            self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
            self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.hidden = hidden
        self.is_cuda = is_cuda




    def forward(self, x, mask):
        self.position = PositionalEmbedding(self.hidden, x.size()[1]).cuda()
        if self.is_cuda:
            self.position = self.position.cuda()

        x = x + self.position(x)
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)



class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))





######################################
###
###  A Dummy Model of Siamese Net
###
######################################

'''
Initiation && HyperParameter
'''
np.random.seed(123)

SAMPLE_SIZE = 10000
EPOCH = 20
BATCH_SIZE = 128
LR = 1e-4
BATCH_NUMS = int(math.ceil(SAMPLE_SIZE/BATCH_SIZE))
RAND_ID = np.random.permutation(list(range(SAMPLE_SIZE)))

RNN_TYPE = 'LSTM'  ## OPTIONAL: LSTM/GRU
#RNN_TIMESTEPS = 10
RNN_TIMESTEPS = 4
RNN_EMB_SIZE = 300
#RNN_HIDDEN_SIZE = 256
RNN_HIDDEN_SIZE = 768
TRANSFORMER_HIDDEN_SIZE = RNN_HIDDEN_SIZE
TRANSFORMER_HEADS = 6
TRANSFORMER_DROPOUT = 0.1
RNN_MERGE_MODE = 'CONCATE'

DEVICE = 'gpu'


'''
Data Preparation
'''
x1 = np.random.rand(SAMPLE_SIZE, RNN_TIMESTEPS, TRANSFORMER_HIDDEN_SIZE).astype(np.float16)
x2 = np.random.rand(SAMPLE_SIZE, RNN_TIMESTEPS, TRANSFORMER_HIDDEN_SIZE).astype(np.float16)

y = np.random.randint(low=0, high=2, size=SAMPLE_SIZE)


'''
Define Network
'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.transformer = TransformerBlock(TRANSFORMER_HIDDEN_SIZE, TRANSFORMER_HEADS, TRANSFORMER_HIDDEN_SIZE * 4, TRANSFORMER_DROPOUT, is_cuda=True)
        '''
        self.FC_mul = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128, 1),
            nn.ReLU()
        )

        self.FC_minus = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128, 1),
            nn.ReLU()
        )
        '''
        self.FC_mul = nn.Sequential(
            nn.Linear(RNN_HIDDEN_SIZE, 256),
            nn.BatchNorm1d(256, 1),
            nn.ReLU()
        )

        self.FC_minus = nn.Sequential(
            nn.Linear(RNN_HIDDEN_SIZE, 256),
            nn.BatchNorm1d(256, 1),
            nn.ReLU()
        )

        self.final = nn.Sequential(
            nn.Linear(256*2, 2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, input):
        text_1 = input[0]
        text_2 = input[1]

        rnn1 = self.transformer(text_1, mask=None)
        rnn2 = self.transformer(text_2, mask=None)

        # Approach 1: Use Final State as Context
        context_1 = rnn1[:, -1, :].clone()  # Warning: clone should be added while slicing
        context_2 = rnn2[:, -1, :].clone()


        # Interaction
        mul = context_1.mul_(context_2)
        minus = context_1.add_(-context_2)


        interation_feature = torch.cat(
            [
                self.FC_mul(mul),
                self.FC_minus(minus),
            ], 1)

        output = self.final(interation_feature)

        return output


class RNN(nn.Module):
    def __init__(self, rnn_type='LSTM'):
        super(RNN, self).__init__()

        self.rnn_type = rnn_type
        self.rnn_hidden_state = None

        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=RNN_EMB_SIZE,
                hidden_size=RNN_HIDDEN_SIZE,  # rnn hidden unit
                num_layers=1,  # 有几层 RNN layers
                batch_first=True,  # input & output 会是以 batch size 为第一维度的特征集 e.g. (batch, time_step, input_size)
                bidirectional=True,
            )
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(
                input_size=RNN_EMB_SIZE,
                hidden_size=RNN_HIDDEN_SIZE,  # rnn hidden unit
                num_layers=1,  # 有几层 RNN layers
                batch_first=True,  # input & output 会是以 batch size 为第一维度的特征集 e.g. (batch, time_step, input_size)
                bidirectional=True,
            )

    def forward(self, input):

        output, self.rnn_hidden_state = self.rnn(input, None)

        return output



def summary(mode='console'):

    params_values  = [RNN_TYPE, RNN_HIDDEN_SIZE, LR]
    params_keys    = ['RNN_TYPE', 'RNN_HIDDEN_SIZE', 'LR']

    params_values = [str(item) for item in params_values]
    max_len = max([len(item) for item in params_keys]) + 1

    def format(key, value, max_len):
        return key + ':' + ' ' * (max_len-len(key)) + value

    if mode == 'console':
        print('#' * 30)
        print('#' * 5)
        print('#' * 5 + '  Model Summary')
        print('#' * 5 + '  ' + '-' * 23)
        for i in range(len(params_keys)):
            print('#' * 5 + '  ' + format(params_keys[i], params_values[i], max_len))
        print('#' * 5)
        print('#' * 30)

    print('-' * 30)


def convert_features_to_tensors_siamese(features, batch_size, embedding):
    all_question_a = torch.tensor([embedding[f.text_a] for f in features], dtype=torch.float32)
    all_question_b = torch.tensor([embedding[f.text_b] for f in features], dtype=torch.float32)
    all_label_ids = torch.tensor([f.label for f in features], dtype=torch.long)

    data = TensorDataset(all_question_a, all_question_b, all_label_ids)
    eval_sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=eval_sampler, batch_size=batch_size)

    return dataloader

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs==labels)

'''
Training Phrase
'''
net = Net().cuda()
summary(mode='console')
loss_func = nn.NLLLoss().cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=LR)


'''
get data
'''
processor = get_class('siamese-model')()

train_examples = processor.get_train_examples('data')

data_len = int(0.9 * len(train_examples))
trains_data = train_examples[:data_len]
valid_data = train_examples[data_len:]

embedding = np.load('embedding_matrix.npy')
print('Loaded existing embedding.')

trains_loader = convert_features_to_tensors_siamese(trains_data, BATCH_SIZE, embedding)
valid_loader = convert_features_to_tensors_siamese(valid_data, BATCH_SIZE, embedding)

BATCH_NUMSs = int(math.ceil(len(train_examples)/BATCH_SIZE))

best_record = 10.0

for epoch in range(EPOCH):

    time.sleep(0.1)
    loss_history = 0
    for step, batch in enumerate(tqdm(trains_loader, desc="Iteration")):
        batch = tuple(t.cuda() for t in batch)
        batch_x1, batch_x2, batch_y = batch

        output = net([batch_x1, batch_x2])
        loss = loss_func(output, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history += loss.item()

    time.sleep(0.1)
    print('EPOCH: {}, Loss: {} \n'.format(epoch, loss_history/BATCH_NUMSs))

    #accuracy
    count = 0
    for step, batch in enumerate(tqdm(valid_loader, desc="Iteration")):
        batch = tuple(t.cuda() for t in batch)
        batch_x1, batch_x2, batch_y = batch

        logits = net([batch_x1, batch_x2]).detach().cpu().numpy()
        label_ids = batch_y.cpu().numpy()
        tmp_eval_accuracy =  accuracy(logits, label_ids)

        count = count + tmp_eval_accuracy


    time.sleep(0.1)
    print("accuracy is :", 100 * count/len(valid_data))

    if (loss_history / BATCH_NUMSs) < best_record :
        best_record = loss_history / BATCH_NUMSs
        #state_dict = {
        #    'epoch': epoch,
        #    'siamese': net.state_dict(),
        #    'optimizer': optimizer.state_dict(),
        #}
        print("save model")
        #torch.save(state_dict, 'siamese_net.pt')
		torch.save(net, 'siamese_net.pt')

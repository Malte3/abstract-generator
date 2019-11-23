import time
import math
import torch
import torchtext
import random
import os
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
from dataset import PaperAbstractDataset

# initialize random seed so the train, evaluation and test split stay the same
random.seed(42)

# Do computations on gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset inherited from torchtext.data.Dataset either creates data by downloading from arxiv, 500 is maximum of samples and the number of words needs to be reduced to fit inside the memory
# The dataset is loaded from files if they exist
train_data, val_data = PaperAbstractDataset.splits(max_results=80, reduced_words=1000, random_state=random.getstate())

# Initialize Interator for the data with batch size
batch_size = 1
train_iter, val_iter = torchtext.data.BucketIterator.splits(
                        (train_data, val_data), batch_sizes=(batch_size, batch_size),
                        device=device, 
                        sort_key=lambda x: len(x.paper),
                        shuffle=True, sort_within_batch=False, repeat=False)
# Put paper and abstract (source, target) in one batch tuple
class BatchTuple():
    def __init__(self, dataset, x_var, y_var):
        self.dataset, self.x_var, self.y_var = dataset, x_var, y_var
        
    def __iter__(self):
        for batch in self.dataset:
            x = getattr(batch, self.x_var) 
            y = getattr(batch, self.y_var)                 
            yield (x, y)
            
    def __len__(self):
        return len(self.dataset)

train_iter_tuple = BatchTuple(train_iter, "abstract", "paper")
val_iter_tuple = BatchTuple(val_iter, "abstract", "paper")

# put out one example batch tuple
next(iter(train_iter_tuple))

# compute maximal paper length
MAX_LENGTH = 0
for abstract, paper in train_iter_tuple:
    if len(paper) > MAX_LENGTH:
        MAX_LENGTH = len(paper)
for abstract, paper in val_iter_tuple:
    if len(paper) > MAX_LENGTH:
        MAX_LENGTH = len(paper)

# The encoder and decoder model from NLP From Scratch: Translation with a Sequence to Sequence Network and Attention (https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html), chosen because of its simplicity
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

# Normal Decoder
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

# Attention Decoder
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

teacher_forcing_ratio = 0.5

# The training process is also from NLP From Scratch: Translation with a Sequence to Sequence Network and Attention (https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_input = target_tensor[di]  # Teacher forcing
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])

    else:
        # Without teacher forcing: use its own predictions as the next input
        decoder_input = target_tensor[0]
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def trainIters(encoder, decoder, epochs, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    for epoch in range(epochs):
        for target, source in train_iter_tuple:
            input_tensor = source
            target_tensor = target

            loss = train(input_tensor, target_tensor, encoder,
                        decoder, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, (epoch+1) / epochs),
                                        epoch, epoch / epochs * 100, print_loss_avg))

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

plt.switch_backend('agg')

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = sentence
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = sentence[0]  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == train_data.fields['paper'].eos_token:
                decoded_words.append('<EOS>')
                break
            else:
                if topi.item() != train_data.fields['paper'].pad_token:
                    decoded_words.append(topi.item())

            decoder_input = topi.squeeze().detach()

        return torch.tensor(decoded_words, dtype=torch.int32).unsqueeze(-1), decoder_attentions[:di + 1]

# function to make words from numericalized text adapted from torchtext ReverseField
def reverse(field, batch):
        if not field.batch_first:
            batch = batch.t()
        with torch.cuda.device_of(batch):
            batch = batch.tolist()
        batch = [[field.vocab.itos[ind] for ind in ex] for ex in batch]  # denumericalize

        def trim(s, t):
            sentence = []
            for w in s:
                if w == t:
                    break
                sentence.append(w)
            return sentence

        batch = [trim(ex, field.eos_token) for ex in batch]  # trim past frst eos

        def filter_special(tok):
            return tok not in (field.init_token, field.pad_token)

        batch = [filter(filter_special, ex) for ex in batch]
        return [' '.join(ex) for ex in batch]

def evaluateRandomly(encoder, decoder, n=2):
    for i in range(n):
        pair = next(iter(val_iter_tuple))
        print('>', reverse(train_data.fields['paper'], pair[0]))
        print('=', reverse(train_data.fields['paper'], pair[1]))
        output_words, attentions = evaluate(encoder, decoder, pair[1])
        output_sentence = reverse(train_data.fields['paper'], output_words)
        print('<', output_sentence)
        print('')

# Initialize Network and Parameters
hidden_size = 256
ntokens = len(next(iter(train_data.fields.values())).vocab.stoi) 
encoder1 = EncoderRNN(ntokens, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, ntokens, dropout_p=0.1).to(device)

# Train new parameters or load them from previous trainings, if you want to restart training delte or rename old .pth files 
if os.path.exists('encoder.pth') and os.path.exists('decoder.pth'):
    encoder1.load_state_dict(torch.load('encoder.pth'))
    attn_decoder1.load_state_dict(torch.load('decoder.pth'))
else:
    trainIters(encoder1, attn_decoder1, 250, print_every=1)
    torch.save(encoder1.state_dict(), 'encoder.pth')
    torch.save(attn_decoder1.state_dict(), 'decoder.pth')

# Show some randomly chosen evaluations
evaluateRandomly(encoder1, attn_decoder1)
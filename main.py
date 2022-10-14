# This block handles some basic setup and data loading.
# You shouldn't need to edit this, but if you want to
# import other standard python packages, that is fine.

from collections import defaultdict, Counter
import numpy as np
import math
import tqdm
import random
import pdb

import torch
from torch import nn
import torch.nn.functional as F

# We'll use HuggingFace's datasets and tokenizers libraries, which are a bit
# heavy-duty for what we're doing, but it's worth getting to know them.

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import WhitespaceSplit
import os

os.system('')
dataset = load_dataset("wikitext", "wikitext-103-v1")
tokenizer = Tokenizer(WordLevel(unk_token='<unk>'))
tokenizer.pre_tokenizer = WhitespaceSplit()  # should be equivalent to split()

# "Training" a tokenizer below just feeds it all the tokens so it can map from
# word type to id.
trainer = WordLevelTrainer(  # should only be 33,278 distinct types in Wikitext-2
    vocab_size=267735, special_tokens=["<unk>", "<eos>"])
generator_bsz = 512
all_splits_generator = (dataset[split][i:i + generator_bsz]["text"]
                        for split in ["train", "validation", "test"]
                        for i in range(0, len(dataset[split]), generator_bsz))
tokenizer.train_from_iterator(all_splits_generator, trainer)

# If desired, we could make a transformers tokenizer object now with:
# fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)

orig_vocab = tokenizer.get_vocab()  # The tokenizer reserves a <pad> id, which we'll ignore.
word_types = sorted(list(orig_vocab.keys()), key=lambda w: orig_vocab[w])  # no <pad>
vocab = {w: i for i, w in enumerate(word_types)}  # no <pad>
vocab_size = len(vocab)
print(vocab_size)
# Make a single stream of tokens, with an <eos> after each newline.

train_text = []
for example in dataset["train"]["text"]:
    train_text.extend(tokenizer.encode(example).tokens + ["<eos>"])

validation_text = []
for example in dataset["validation"]["text"]:
    validation_text.extend(tokenizer.encode(example).tokens + ["<eos>"])

print(train_text[:30])


class UnigramModel:
    def __init__(self, train_text):
        self.counts = Counter(train_text)
        self.total_count = len(train_text)

    def probability(self, word):
        return self.counts[word] / self.total_count

    def next_word_probabilities(self, text_prefix):
        """Return a list of probabilities for each word in the vocabulary."""
        return [self.probability(word) for word in word_types]

    def perplexity(self, full_text):
        """Return the perplexity of the model on a text as a float.

        full_text -- a list of string tokens
        """
        log_probabilities = []
        for word in full_text:
            # Note that the base of the log doesn't matter
            # as long as the log and exp use the same base.
            log_probabilities.append(math.log(self.probability(word), 2))
        return 2 ** -np.mean(log_probabilities)


# unigram_demonstration_model = UnigramModel(train_text)
# print('unigram validation perplexity:',
#       unigram_demonstration_model.perplexity(validation_text))


def check_validity(model):
    """Performs several sanity checks on your model:
    1) That next_word_probabilities returns a valid distribution
    2) That perplexity matches a perplexity calculated from next_word_probabilities

    Although it is possible to calculate perplexity from next_word_probabilities,
    it is still good to have a separate more efficient method that only computes
    the probabilities of observed words.
    """

    log_probabilities = []
    for i in range(10):
        prefix = validation_text[:i]
        probs = model.next_word_probabilities(prefix)
        assert min(probs) >= 0, "Negative value in next_word_probabilities"
        assert max(probs) <= 1 + 1e-8, "Value larger than 1 in next_word_probabilities"
        assert abs(sum(probs) - 1) < 1e-4, "next_word_probabilities do not sum to 1"

        word_id = vocab[validation_text[i]]
        selected_prob = probs[word_id]
        log_probabilities.append(math.log(selected_prob))

    perplexity = math.exp(-np.mean(log_probabilities))
    your_perplexity = model.perplexity(validation_text[:10])
    assert abs(perplexity - your_perplexity) < 0.1, "your perplexity does not " + \
                                                    "match the one we calculated from `next_word_probabilities`,\n" + \
                                                    "at least one of `perplexity` or `next_word_probabilities` is incorrect.\n" + \
                                                    f"we calcuated {perplexity} from `next_word_probabilities`,\n" + \
                                                    f"but your perplexity function returned {your_perplexity} (on a small sample)."


# check_validity(unigram_demonstration_model)


def generate_text(model, n=20, prefix=('<eos>', '<eos>')):
    prefix = list(prefix)
    for _ in range(n):
        probs = model.next_word_probabilities(prefix)
        word = random.choices(word_types, probs)[0]
        prefix.append(word)
    return ' '.join(prefix)


# print(generate_text(unigram_demonstration_model))


def save_truncated_distribution(model, filename, short=True):
    """Generate a file of truncated distributions.

    Probability distributions over the full vocabulary are large,
    so we will truncate the distribution to a smaller vocabulary.

    Please do not edit this function
    """
    vocab_name = 'nu_eval_output_vocab'
    prefixes_name = 'nu_eval_prefixes'

    if short:
        vocab_name += '_short'
        prefixes_name += '_short'

    with open('{}.txt'.format(vocab_name), 'r') as eval_vocab_file:
        eval_vocab = [w.strip() for w in eval_vocab_file]
    eval_vocab_ids = [vocab[s] for s in eval_vocab]

    all_selected_probabilities = []
    with open('{}.txt'.format(prefixes_name), 'r') as eval_prefixes_file:
        lines = eval_prefixes_file.readlines()
        for line in tqdm.notebook.tqdm(lines, leave=False):
            prefix = line.strip().split(' ')
            probs = model.next_word_probabilities(prefix)
            selected_probs = np.array([probs[i] for i in eval_vocab_ids], dtype=np.float32)
            all_selected_probabilities.append(selected_probs)

    all_selected_probabilities = np.stack(all_selected_probabilities)
    np.save(filename, all_selected_probabilities)
    print('saved', filename)


class NLLwithUIDLoss(nn.Module):
    def __init__(self, beta):
        super(NLLwithUIDLoss, self).__init__()
        self.nll_loss = nn.NLLLoss(reduction='none')
        self.final_loss = nn.NLLLoss()
        self.beta = beta
        self.mse = nn.MSELoss()

    def forward(self, prediction, target, need_nll=True):
        if need_nll:
            surprisal = self.nll_loss(prediction, target)
        else:
            surprisal = prediction
        squared_error = self.mse(surprisal, torch.full_like(surprisal, torch.mean(surprisal).item()))
        if need_nll:
            return self.beta * squared_error + self.final_loss(prediction, target)
        else:
            return self.beta * squared_error + torch.mean(surprisal)


## Adaptive input representations:

class AdaptiveInputEmbedding(nn.Module):

    def __init__(self, corpus: list, embedding_dim: int = 128, buckets: list = (0, 300, 3000, 30000, 267734),
                 decay: int = 2):
        """
        paramters

          corpus: list
            tokenized list of input words from a corpus
          buckets: int
            number of buckets for seperate embeddings
          decay: int
            the rate of decay of the number of embeddings
        """
        super(AdaptiveInputEmbedding, self).__init__()
        self.embed_size = embedding_dim
        words = np.unique(corpus)
        self.buckets = [
            torch.LongTensor(words[bucket: buckets[i + 1]])
            for i, bucket in enumerate(buckets[:-1])
        ]
        self.map = [torch.LongTensor(list(range(len(bucket)))) for bucket in self.buckets]

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(
                    len(bucket),
                    embedding_dim // decay ** i
                )
                for i, bucket in enumerate(self.buckets)
            ]
        )
        self.linears = nn.ParameterList(
            [
                nn.Parameter(torch.Tensor(
                    embedding_dim // decay ** i,
                    embedding_dim
                ).uniform_(-0.01, 0.01)).to(device)
                for i in range(len(self.buckets))
            ]
        )

    def forward(self, x: torch.Tensor):
        """
        parameters
          x: torch.Tensor
            Shape: `(batch_size, max_len)
        """
        # Get the bucket each word is in

        x_flat = x.flatten()
        embed_flat = torch.zeros([x_flat.shape[0], self.embed_size]).to(device)
        for i, bucket in enumerate(self.buckets):
            in_bucket = ((x_flat >= bucket[0]) & (x_flat <= bucket[-1])).nonzero().squeeze()
            if in_bucket.numel() == 0:
                continue

            x_i = x_flat.index_select(0, in_bucket)
            x_i = x_i - bucket[0]

            z = self.embeddings[i](x_i)
            z = F.linear(z, self.linears[i].T)

            embed_flat.index_copy_(0, in_bucket, z)

        embed = embed_flat.reshape(*x.shape, self.embed_size)
        return embed


class AdaptiveLogSoftmax(nn.Module):
    def __init__(self, embedding_dim: int = 128, buckets: list = (0, 300, 3000, 30000, 267734), decay: int = 2):
        super().__init__()
        self.embed_dim = embedding_dim
        self.cluster_weight = nn.Parameter(torch.zeros(len(buckets) - 1, embedding_dim)).to(device)
        self.cluster_bias = nn.Parameter(torch.zeros(len(buckets) - 1)).to(device)
        self.out_projs = nn.ParameterList()
        self.out_layers = nn.ModuleList()
        self.buckets = buckets
        for i in range(len(self.buckets) - 1):
            self.out_projs.append(
                nn.Parameter(torch.Tensor(embedding_dim, embedding_dim // decay ** i))
            )
            self.out_layers.append(
                nn.Linear(
                    embedding_dim // decay ** i,
                    self.buckets[i + 1] - self.buckets[i]
                )
            )

    def _logits(self, x, weight, bias, proj):

        hidden = F.linear(x, proj)

        logit = F.linear(hidden, weight, bias=bias)
        return logit

    def forward(self, x, tgt):

        weights = [layer.weight for layer in self.out_layers]
        biases = [layer.bias for layer in self.out_layers]

        weights[0] = torch.cat(
            [
                weights[0],
                self.cluster_weight
            ],
            dim=0
        )

        biases[0] = torch.cat(
            [
                biases[0],
                self.cluster_bias
            ]
        )

        x_flat = x.reshape(-1, self.embed_dim)
        tgt = tgt.reshape(-1)

        head = self._logits(x_flat, weights[0], biases[0], self.out_projs[0])
        head = F.log_softmax(head, dim=1)
        if self.training:

            nll = torch.zeros_like(tgt, dtype=x.dtype).to(device)
            for i, cutoff in enumerate(self.buckets[:-1]):
                in_bucket = ((tgt >= cutoff) & (tgt < self.buckets[i + 1])).nonzero().squeeze()

                if in_bucket.numel() == 0:
                    continue
                tgt_i = tgt.index_select(0, in_bucket) - cutoff
                head_i = head.index_select(0, in_bucket)

                if i == 0:

                    logmax = head_i.gather(1, tgt_i[:, None]).squeeze(1)

                else:
                    x_i = x_flat.index_select(0, in_bucket)
                    tail = F.log_softmax(self._logits(x_i, weights[i], biases[i], self.out_projs[i]), dim=1)
                    logmax = tail.gather(1, tgt_i[:, None]).squeeze(1) + head_i[:, -i]

                if self.training:
                    nll.index_copy_(0, in_bucket, -logmax)
            return nll
        if not self.training:
            all_prob = torch.zeros(x_flat.shape[0], self.buckets[-1]).to(device)
            for i, cutoff in enumerate(self.buckets[:-1]):
                if i == 0:
                    prob = head[:, cutoff:self.buckets[i + 1]]
                else:

                    prob = F.log_softmax(
                        self._logits(x_flat, weights[i], biases[i], self.out_projs[i])
                        , dim=1
                    )

                    prob += head[:, -i].unsqueeze(1)

                all_prob[:, cutoff:self.buckets[i + 1]].copy_(prob)

            return all_prob


def ids(tokens):
    return [vocab[t] for t in tokens]


import transformers

assert torch.cuda.is_available(), "no GPU found, in Colab go to 'Edit->Notebook settings' and choose a GPU hardware accelerator"

device = torch.device("cuda")


class TransformerLMDataset:
    """Returns a batch of sequences with their corresponding target word ids.
    Note that the returned tensors are of shapes `max_len x batch_size`, hence
    the sequence length dimension appears before the batch dimension."""

    def __init__(self, text_token_ids, bsz, max_len=128):
        self.bsz = bsz
        self.max_len = max_len
        token_ids = torch.tensor(text_token_ids)
        ncontig = token_ids.size(0) // bsz
        token_ids = token_ids[:ncontig * bsz].view(bsz, -1)  # bsz x ncontig
        self.token_ids = token_ids.t().contiguous()  # ncontig x bsz

    def __len__(self):
        return int(math.ceil(self.token_ids.size(0) / self.max_len))

    def __iter__(self):
        for i in range(0, self.token_ids.size(0) - 1, self.max_len):
            seqlen = min(self.max_len, self.token_ids.size(0) - i - 1)
            x = self.token_ids[i:i + seqlen]  # max_len x bsz
            y = self.token_ids[i + 1:i + seqlen + 1]  # max_len x bsz
            yield x, y


class TransformerNetwork(nn.Module):
    # a PyTorch Module that holds the neural network for your model

    def __init__(self, corpus, max_len=128, embed_size=128):  # feel free to add more parameters
        super().__init__()

        # Initialize the different layers needed in the computation graph.
        # A full list of available layers in Pytorch is given here:
        # https://pytorch.org/docs/stable/nn.html

        # In PyTorch, to construct a TransformerEncoder layer, we first need
        # to construct a nn.TransformerEncoderLayer and then pass it to
        # nn.TransformerEncoder.
        # Besides these you will need linear and dropout layers for this model.

        # Below we have initialized the positional embeddings for you.
        super().__init__()
        self.pe = nn.Parameter(torch.FloatTensor(max_len, embed_size).uniform_(-0.01, 0.01)).to(device)

        # YOUR CODE HERE

        self.max_len = max_len
        self.embedding = AdaptiveInputEmbedding(
            corpus,
            embed_size,
            decay=4
        )
        self.second = nn.Linear(embed_size, embed_size * 2)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                embed_size * 2,
                8,
                embed_size * 4,
                .3,
                batch_first=True
            ).to(device),
            9
        ).to(device)
        self.down = nn.Linear(embed_size * 2, embed_size)
        self.softmax = AdaptiveLogSoftmax(
            embed_size,
            decay=4
        )

        for i in range(len(self.softmax.out_layers)):
            self.softmax.out_layers[i].weight = self.embedding.embeddings[i].weight

        for i, tie_proj in enumerate(self.softmax.out_projs):
            self.softmax.out_projs[i] = self.embedding.linears[i]
        self.dropout = nn.Dropout(.3)
        self.last_hidden = nn.Sequential(
            nn.Linear(embed_size * 2, embed_size),
        )
        self.relu = nn.ReLU()

    def _generate_mask(self, sz):
        """Return an attention mask with shape (max_len, max_len)."""

        # Note: you can use torch.triu (https://pytorch.org/docs/stable/generated/torch.triu.html) to build a upper triangular matrix.
        # For passing to the TransformerEncoder, we will need to replace 0s with -inf and 1s with 0

        # YOUR CODE HERE
        tri = torch.triu(torch.ones(sz, sz))
        tri[tri == 0] = -np.infty
        tri[tri == 1] = 0
        return tri.permute(1, 0).to(device)

    def forward(self, x, tgt):
        """ x - a tensor with shape (bsz, max_len=128)
            returns a tensor of log probabilities with shape (bsz, max_len, len(word_types)).
        """

        # Make sure you add the positional embeddings initialized above to the word embeddings
        # extracted from the inputs.
        # Make sure you provide an attention mask to the TransformerEncoder (after moving it to the GPU).
        # YOUR CODE HERE
        mx_len, bsz = x.shape
        x = x.permute(1, 0)
        x = self.dropout(self.embedding(x) + self.pe[:mx_len])
        x = self.dropout(self.transformer(self.second(x), mask=self._generate_mask(mx_len)))
        return self.softmax(self.down(x), tgt)


class TransformerModel:
    "A class that wraps TransformerNetwork to handle training and evaluation."

    def __init__(self, corpus, batch_size, max_len, embed_size):
        self.network = TransformerNetwork(corpus, max_len, embed_size).to(device)
        self.batch_size = batch_size
        self.max_len = max_len

    def train(self, epochs):
        transformer_dataset = TransformerLMDataset(ids(train_text[:int(len(train_text) / 2)]), self.batch_size,
                                                   self.max_len)
        # Iterate over transformer_dataset with a for loop (optionally with tqdm).
        # Looping thru this dataset gives (x, y) tuples,
        # where x is a seqlen x batch_size token id tensor, and y is a seqlen x batch_size token id tensor.
        # The token ids in y are the next word targets for the sequence up till that position
        # in x.

        # The basic recipe for training the network will be the same as the NeuralNGramModel,
        # but note that the network will return the next word predictions of the entire
        # sequence together. You will need to reshape this in order to use the nll_loss.

        # Note: You will need a learning rate scheduler in addition to the optimizer.
        # Note: Initialize your optimizer and scheduler before starting the training loop.
        # Note: Make sure you update the parameters of your optimizer and scheduler after each training step.
        # Note: You should zero out previous gradients before each backpropragation:
        # https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html

        # You should also print the perplexity on the validation set after each epoch by
        # calling self.perplexity(validation_text). You can do early stopping by
        # comparing this perplexity to the perplexity from the previous epoch
        # and stop training when it gets larger.

        # You should also clip the gradients before performing an optimizer.step()
        # using torch.nn.utils.clip_grad_norm.

        # YOUR CODE HERE
        optim = torch.optim.Adam(self.network.parameters(), lr=0.001)
        lr_sched = transformers.get_linear_schedule_with_warmup(optim, int(.1 * epochs), epochs)
        criterion = NLLwithUIDLoss(0.001)
        prev_val = np.infty
        three_prev = []
        lr_sched.step()
        for epoch in range(epochs):
            self.network.train()
            print(lr_sched.get_last_lr(), epoch)
            with tqdm.tqdm(transformer_dataset) as training_bar:
                for seq, pred in training_bar:
                    seq, pred = seq.to(device), pred.to(device)
                    max_len, batch = seq.shape
                    out = self.network(seq, pred.permute(1, 0))
                    out = out.reshape(max_len * batch, -1)
                    loss = criterion(out, pred, False)
                    optim.zero_grad()

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                    optim.step()
                    training_bar.set_postfix(perplexity=math.exp(loss.detach()))
            lr_sched.step()
            perplexity = self.perplexity(validation_text)
            print(perplexity)
            if perplexity < prev_val:
                torch.save(self.network.state_dict(), f"model.pth")
                prev_val = perplexity
            if len(three_prev) == 6 and all(prev < perplexity for prev in three_prev):
                break

            three_prev.append(perplexity)
            if len(three_prev) > 6:
                three_prev = three_prev[1:]
        network_dict = torch.load("model.pth")
        self.network.load_state_dict(network_dict)

    def next_word_probabilities(self, text_prefix):
        "Return a list of probabilities for each word in the vocabulary."

        # you will need to convert text_prefix from strings to numbers with the `ids` function
        # We won't be calling check_validity on the Transformer so you don't need
        # worry about the empty text_prefix.

        # Make sure you return probabilities instead of log probabilities!

        # YOUR CODE HERE
        # don't forget self.network.eval()
        # don't forget to move tensors to the GPU
        self.network.eval()

        text_prefix = ids(text_prefix)
        text_prefix = torch.LongTensor(text_prefix).to(device).unsqueeze(0)
        with torch.no_grad():
            pred = self.network.forward(text_prefix)
        return list(pred.squeeze().cpu()[1].numpy())

    def perplexity(self, text):
        "Return perplexity as a float."
        # Use torch.no_grad() for extra speed.

        # Note that the nll_loss function, by default, computes the losses
        # averaged over each loss element in the batch.

        bsz = self.batch_size if len(text) > self.batch_size else 1
        transformer_dataset = TransformerLMDataset(ids(text), bsz, min(self.max_len, len(text)))

        # Loop over the transformer_dataset and get the outputs from your network.
        # Use nll_loss after reshaping these outputs and the corresponding targets
        # to get the average log probabilities.

        # Note that nll_loss by default will give you an average over all the elements
        # in a batch, keep track of the total number of elements for computing the
        # overall average for the perplexity.

        # YOUR CODE HERE
        # don't forget self.network.eval()
        # don't forget to move tensors to the GPU
        val_loss = []
        batch_size = []
        self.network.eval()
        with torch.no_grad():
            for data in transformer_dataset:
                seq, tgt = data[0].to(device), data[1].to(device)
                max_len, batch = seq.shape

                pred = self.network(seq, tgt.permute(1, 0))

                tgt = tgt.permute(1, 0).reshape(max_len * batch)
                loss = F.nll_loss(pred, tgt)
                val_loss.append(loss.detach().cpu())
                batch_size.append(batch * max_len)

        mean = sum(loss * batch for loss, batch in zip(val_loss, batch_size)) / sum(batch_size)
        print(pred)
        return math.exp(mean)


transformer_model = TransformerModel(ids(train_text), 64, 240, 512)

transformer_model.train(100)
print('transformer validation perplexity:', transformer_model.perplexity(validation_text))
save_truncated_distribution(transformer_model, 'transformer_predictions_adapt.npy')

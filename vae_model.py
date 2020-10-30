import torch
import torch.nn as nn
import torch.nn.functional as F

from data import *

# vae parameters
q_bidir = True
enc_hidden = 125
dec_hidden = 512
enc_layers = 6
dec_layers = 1
enc_dropout = 0.1
dec_dropout = 0.01
d_z = 128


class VAE(nn.Module):
    def __init__(self, vocab, vector):
        super().__init__()
        self.vocabulary = vocab
        self.vector = vector

        n_vocab, d_emb = len(vocab), vector.size(1)
        self.x_emb = nn.Embedding(n_vocab, d_emb, c2i['<pad>'])
        self.x_emb.weight.data.copy_(vector)

        # Encoder
        self.encoder_rnn = nn.GRU(d_emb, enc_hidden, num_layers=enc_layers, batch_first=True,
                                  dropout=enc_dropout if enc_layers > 1 else 0, bidirectional=q_bidir)
        q_d_last = enc_hidden * (2 if q_bidir else 1)
        self.q_mu = nn.Linear(q_d_last, d_z)
        self.q_logvar = nn.Linear(q_d_last, d_z)

        # Decoder
        self.decoder_rnn = nn.GRU(d_emb + d_z, dec_hidden, num_layers=dec_layers, batch_first=True,
                                  dropout=dec_dropout if dec_layers > 1 else 0)
        self.decoder_latent = nn.Linear(d_z, dec_hidden)
        self.decoder_fullyc = nn.Linear(dec_hidden, n_vocab)

        # Grouping the model's parameters
        self.encoder = nn.ModuleList([self.encoder_rnn, self.q_mu, self.q_logvar])
        self.decoder = nn.ModuleList([self.decoder_rnn, self.decoder_latent, self.decoder_fullyc])
        self.vae = nn.ModuleList([self.x_emb, self.encoder, self.decoder])

    @property
    def device(self):
        return next(self.parameters()).device

    def string2tensor(self, string, device='model'):
        ids = string2ids(string, add_bos=True, add_eos=True)
        tensor = torch.tensor(ids, dtype=torch.long, device=self.device if device == 'model' else device)
        return tensor

    def tensor2string(self, tensor):
        ids = tensor.tolist()
        string = ids2string(ids, rem_bos=True, rem_eos=True)
        return string

    def forward(self, x):
        z, kl_loss = self.forward_encoder(x)
        recon_loss = self.forward_decoder(x, z)
        # print("forward")
        return kl_loss, recon_loss

    def forward_encoder(self, x):
        x = [self.x_emb(i_x) for i_x in x]
        x = nn.utils.rnn.pack_sequence(x)
        _, h = self.encoder_rnn(x, None)
        h = h[-(1 + int(self.encoder_rnn.bidirectional)):]
        h = torch.cat(h.split(1), dim=-1).squeeze(0)
        mu, logvar = self.q_mu(h), self.q_logvar(h)
        eps = torch.randn_like(mu)
        z = mu + (logvar / 2).exp() * eps
        kl_loss = 0.5 * (logvar.exp() + mu ** 2 - 1 - logvar).sum(1).mean()
        return z, kl_loss

    def forward_decoder(self, x, z):
        lengths = [len(i_x) for i_x in x]
        x = nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=c2i['<pad>'])
        x_emb = self.x_emb(x)
        z_0 = z.unsqueeze(1).repeat(1, x_emb.size(1), 1)
        x_input = torch.cat([x_emb, z_0], dim=-1)
        x_input = nn.utils.rnn.pack_padded_sequence(x_input, lengths, batch_first=True)
        h_0 = self.decoder_latent(z)
        h_0 = h_0.unsqueeze(0).repeat(self.decoder_rnn.num_layers, 1, 1)
        output, _ = self.decoder_rnn(x_input, h_0)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        y = self.decoder_fullyc(output)

        recon_loss = F.cross_entropy(y[:, :-1].contiguous().view(-1, y.size(-1)), x[:, 1:].contiguous().view(-1),
                                     ignore_index=c2i['<pad>'])
        return recon_loss

    def sample_z_prior(self, n_batch):
        return torch.randn(n_batch, self.q_mu.out_features, device=self.x_emb.weight.device)

    def sample(self, n_batch, max_len=100, z=None, temp=1.0):
        with torch.no_grad():
            if z is None:
                z = self.sample_z_prior(n_batch)
                z = z.to(self.device)
                z_0 = z.unsqueeze(1)
                h = self.decoder_latent(z)
                h = h.unsqueeze(0).repeat(self.decoder_rnn.num_layers, 1, 1)
                w = torch.tensor(c2i['<$>'], device=self.device).repeat(n_batch)
                x = torch.tensor([c2i['<pad>']], device=device).repeat(n_batch, max_len)
                x[:, 0] = c2i['<$>']
                end_pads = torch.tensor([max_len], device=self.device).repeat(n_batch)
                eos_mask = torch.zeros(n_batch, dtype=torch.uint8, device=self.device)
                # eos_mask = torch.zeros(n_batch, dtype=torch.bool, device=self.device)

                for i in range(1, max_len):
                    x_emb = self.x_emb(w).unsqueeze(1)
                    x_input = torch.cat([x_emb, z_0], dim=-1)

                    o, h = self.decoder_rnn(x_input, h)
                    y = self.decoder_fullyc(o.squeeze(1))
                    y = F.softmax(y / temp, dim=-1)

                    w = torch.multinomial(y, 1)[:, 0]
                    x[~eos_mask, i] = w[~eos_mask]
                    i_eos_mask = ~eos_mask & (w == c2i['<#>'])
                    end_pads[i_eos_mask] = i + 1
                    eos_mask = eos_mask | i_eos_mask

                    new_x = []
                    for i in range(x.size(0)):
                        new_x.append(x[i, :end_pads[i]])

        return [self.tensor2string(i_x) for i_x in new_x]

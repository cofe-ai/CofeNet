import torch
import torch.nn as nn


class SeqCNN(nn.Module):
    def __init__(self, in_hidden_size, channels, out_hidden_size, kernel_sizes, pool='avg'):
        super(SeqCNN, self).__init__()
        self.pool = pool
        assert self.pool in ['avg', 'max', 'none']
        if self.pool == 'max':
            self.layer_convs = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(1, channels, (ks, in_hidden_size), padding=(ks - 1, 0)),
                    nn.ReLU(True),
                    nn.MaxPool2d((ks, 1), stride=1)
                )
                for ks in kernel_sizes
            ])
        elif self.pool == 'avg':
            self.layer_convs = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(1, channels, (ks, in_hidden_size), padding=(ks - 1, 0)),
                    nn.ReLU(True),
                    nn.AvgPool2d((ks, 1), stride=1)
                )
                for ks in kernel_sizes
            ])
        elif self.pool == 'none':
            self.layer_convs = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(1, channels, (ks, in_hidden_size), padding=(ks - 1, 0)),
                    nn.ReLU(True)
                )
                for ks in kernel_sizes
            ])

        self.layer_fc = nn.Linear(channels * len(kernel_sizes), out_hidden_size)

    def forward(self, input):
        """
        :param input: [batch, seq, hidden]
        :return:
        """
        if self.pool == 'none':
            outputs = torch.cat([conv(input.unsqueeze(1)).squeeze(-1).transpose(-1, -2)[:, :input.size(1), :] for conv in self.layer_convs], -1)
        else:
            outputs = torch.cat([conv(input.unsqueeze(1)).squeeze(-1).transpose(-1, -2) for conv in self.layer_convs], -1)
        return self.layer_fc(outputs)

    @staticmethod
    def test():
        hidden_size = 7
        out_hidden_size = 3
        kernel_sizes = [1, 2, 3]

        layer_emb = nn.Embedding(5, hidden_size)
        obj = SeqCNN(hidden_size, 5, out_hidden_size, kernel_sizes)

        batch_seq_ch = [
            [1, 2, 3, 4],
            [4, 1, 3, 4],
            [1, 1, 4, 4],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
        ]
        batch_seq_ch = torch.tensor(batch_seq_ch)

        batch_seq_hidden = layer_emb(batch_seq_ch)
        out = obj(batch_seq_hidden)
        print(out)
        print(out.size())


if __name__ == '__main__':
    SeqCNN.test()

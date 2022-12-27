import torch

class CrossAttention(torch.nn.Module):
    def __init__(self, in_channels, out_channels, query_channels):
        super(CrossAttention, self).__init__()
        self.key_projection = torch.nn.Linear(in_channels, out_channels)
        self.value_projection = torch.nn.Linear(in_channels, out_channels)
        self.query_projection = torch.nn.Linear(query_channels, out_channels)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x, query):
        keys = self.key_projection(x)
        values = self.value_projection(x)
        query = self.query_projection(query)
        attention_weights = torch.matmul(query, keys.transpose(-1, -2))
        attention_weights = torch.nn.functional.softmax(attention_weights, dim=-1)
        attention_weights = torch.nn.functional.interpolate(attention_weights, size=(256, 256))
        output = torch.mul(attention_weights, values)
        output = self.dropout(output)
        return output
import torch 
from torch import nn
import torch.nn.functional as F



class SelfAttention(nn.Module):
    
    def __init__(self, in_channels:int):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        self.key_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        self.value_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        self.final_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        self.alpha = nn.Parameter(torch.rand(1))
        self.beta = nn.Parameter(torch.rand(1))
    
    def forward(self, x):
        query = self.query_conv(x)
        key = self.key_conv(x)
        value = self.value_conv(x)
        energy = torch.softmax(query * key, dim=1)
        attention = self.final_conv(value * energy)
        return (self.alpha * attention) + (self.beta * x)
    
    

class ResBlock(nn.Module):
    
    def __init__(self, in_channels:int, out_channels:int):
        super(ResBlock, self).__init__()
        self.res_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            SelfAttention(in_channels=out_channels)
        )
    
    def forward(self, x):
        return self.conv(x) + self.res_conv(x)
    
    
    
class ResNet(nn.Module):
    
    def __init__(self, in_channels:int, out_channels:int):
        super(ResNet, self).__init__()
        self.block1 = ResBlock(in_channels=in_channels, out_channels=out_channels//16)
        self.block2 = ResBlock(in_channels=out_channels//16, out_channels=out_channels//8)
        self.block3 = ResBlock(in_channels=out_channels//8, out_channels=out_channels//4)
        self.block4 = ResBlock(in_channels=out_channels//4, out_channels=out_channels//2)
        self.block5 = ResBlock(in_channels=out_channels//2, out_channels=out_channels)
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x
    
    
    
class PatchEmbedding(nn.Module):
    
    def __init__(self, img_channels:int, embedding_dim:int, patch_size:int):
        super(PatchEmbedding, self).__init__()
        self.conv = nn.Conv2d(in_channels=img_channels, out_channels=embedding_dim, kernel_size=patch_size, stride=patch_size, padding=0)
    
    def forward(self, x):
        x = self.conv(x)
        return x.view(x.shape[0], x.shape[1], -1).permute(0,2,1)       
    


class ResformerEncoder(nn.Module):
    
    def __init__(self, img_size:int=400, img_channels:int=3, resnet_channels:int=256, embedding_dim:int=2048, patch_size:int=16, num_text_tokens:int=8, num_transformer_layers:int=12, num_heads:int=16, fc_hidden:int=1024):
        super(ResformerEncoder, self).__init__()       
        self.cnn_resnet = ResNet(in_channels=img_channels, out_channels=resnet_channels)
        
        num_patches = int((img_size/patch_size)**2)
        self.patch_embedding = PatchEmbedding(img_channels=resnet_channels, embedding_dim=embedding_dim, patch_size=patch_size)
        self.text_embedding = nn.Embedding(num_embeddings=num_text_tokens, embedding_dim=embedding_dim)
        self.class_token = nn.Parameter(torch.rand(1, 1, embedding_dim))
        self.position_embedding = nn.Parameter(torch.rand(1, num_patches+2, embedding_dim))
        self.embedding_dropout = nn.Dropout(p=0.1)
        
        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=fc_hidden, dropout=0.1, activation='relu', batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers=num_transformer_layers)
        
        
    def forward(self, x, field_type):
        x = self.cnn_resnet(x)
        patch_embed = self.patch_embedding(x)
        text_embed = self.text_embedding(field_type.int()).unsqueeze(dim=1)
        class_token = self.class_token.repeat(x.shape[0], 1, 1)
        transformer_input = torch.cat([patch_embed, text_embed, class_token], dim=1)
        transformer_input = transformer_input + self.position_embedding.repeat(x.shape[0], 1, 1)
        
        transformer_input = self.embedding_dropout(transformer_input)
        transformer_output = self.transformer_encoder(transformer_input)
        
        return transformer_output         
    
    


class ResformerDecoder(nn.Module):
    
    def __init__(self, num_text_tokens:int=50, embedding_dim:int=2048, max_seq_len:int=64, num_transformer_layers:int=12, num_heads:int=16, fc_hidden:int=1024):
        super(ResformerDecoder, self).__init__()
        self.text_embedding = nn.Embedding(num_embeddings=num_text_tokens, embedding_dim=embedding_dim)
        self.positional_embedding = nn.Parameter(torch.rand(1, max_seq_len, embedding_dim))
        self.decoder = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=fc_hidden, dropout=0.1, activation='relu', batch_first=True)
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim, out_features=num_text_tokens*4),
            nn.Dropout(p=0.1),
            nn.LayerNorm(normalized_shape=num_text_tokens*4),
            nn.Linear(in_features=num_text_tokens*4, out_features=num_text_tokens),
            nn.Softmax()    
        )
    
    
    def forward(self, x, memory, output_mask=None):
        x = self.text_embedding(x.int())
        x = x + self.positional_embedding[:, :x.shape[1], :].repeat(x.shape[0], 1, 1)
        decoder_output = self.decoder(x, memory, tgt_key_padding_mask=output_mask)
        pred_probs = self.classifier(decoder_output)
        return pred_probs
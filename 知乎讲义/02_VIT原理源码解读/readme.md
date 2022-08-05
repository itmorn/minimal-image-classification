本文以pytorch提供的官方 ViT_B_16 为例进行梳理。
# 0.网络结构总览

VIT和BERT的结构基本是一致的。
- 加CLS token
- 加位置编码
- 最后有一个单层线性分类头

VIT的思想：

把图像切成很多小块，然后对每个小块编码成768维度的向量，这一点和BERT对每个token的编码过程是很相似的。

下面总览一下VIT的结构，图片有些长，关键部位的尺寸和信息在图上做了备注：

![](vit_b_16.onnx.png)

# 1.网络前向传播
```python
def forward(self, x: torch.Tensor):
    # Reshape and permute the input tensor
    x = self._process_input(x)
    n = x.shape[0]

    # Expand the class token to the full batch
    batch_class_token = self.class_token.expand(n, -1, -1)
    x = torch.cat([batch_class_token, x], dim=1)

    x = self.encoder(x)

    # Classifier "token" as used by standard language architectures
    x = x[:, 0]

    x = self.heads(x)

    return x
```

## 1.1.输入图像预处理
![img.png](img.png)

预处理的方式和前面写过的ResNet的方式是一样的，这里不加赘述。

## 1.2.VIT的预处理
x = self._process_input(x)
```python
def _process_input(self, x: torch.Tensor) -> torch.Tensor:
    n, c, h, w = x.shape
    p = self.patch_size
    torch._assert(h == self.image_size, "Wrong image height!")
    torch._assert(w == self.image_size, "Wrong image width!")
    n_h = h // p
    n_w = w // p

    # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
    x = self.conv_proj(x)
    # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
    x = x.reshape(n, self.hidden_dim, n_h * n_w)

    # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
    # The self attention layer expects inputs in the format (N, S, E)
    # where S is the source sequence length, N is the batch size, E is the
    # embedding dimension
    x = x.permute(0, 2, 1)

    return x
```
### 1.2.1.对图像进行分块

patch_size我们设置的是16，因此，

n_h = h // p = 224 // 16 = 14； 

n_w = w // p = 224 // 16 = 14； 

### 1.2.2.对patch进行编码
x = self.conv_proj(x)
```python
self.conv_proj = nn.Conv2d(
    in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
)
```
其中hidden_dim=768，patch_size=16。

因此，x会由(n, c, h, w) -> (n, hidden_dim, n_h, n_w),即(n, 3, 224, 224) -> (n, 768, 14, 14)

然后，reshape拉直，permute调整通道顺序为(n, n_h*n_w, hidden_dim),即(n, 196, 768)

## 1.3.增加CLS token
```python
# Expand the class token to the full batch
batch_class_token = self.class_token.expand(n, -1, -1)
x = torch.cat([batch_class_token, x], dim=1)
```
其中，self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

那么，x的维度就(n, 196, 768) -> (n, 197, 768)

## 1.4.进入VIT的Encoder部分
x = self.encoder(x)
```python
def forward(self, input: torch.Tensor):
    torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
    input = input + self.pos_embedding
    return self.ln(self.layers(self.dropout(input)))
```

### 1.4.1.对输入加 pos_embedding
input = input + self.pos_embedding
self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
self.pos_embedding的维度为(n, 197, 768),位置编码是随机初始化的。

### 1.4.2.对输入进行 dropout
self.dropout(input)
pytorch中默认p=0

### 1.4.3.输入 encoder layers
![img_1.png](img_1.png)

self.layers(self.dropout(input))

其中，self.layers的定义如下：
```python
layers: OrderedDict[str, nn.Module] = OrderedDict()
for i in range(num_layers):
    layers[f"encoder_layer_{i}"] = EncoderBlock(
        num_heads,
        hidden_dim,
        mlp_dim,
        dropout,
        attention_dropout,
        norm_layer,
    )
self.layers = nn.Sequential(layers)
```
其中，num_layers=12，对应上图中的L×；num_heads=12；mlp_dim=3072；dropout=0；attention_dropout=0

其中，EncoderBlock的定义如下：
```python
class EncoderBlock(nn.Module):
    """Transformer encoder block."""
    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x, _ = self.self_attention(query=x, key=x, value=x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y
```
我们知道，self_attention是不会改变输入维度的，

因此，x = self.encoder(x)之后，x的维度仍然为 (n, 197, 768)


## 1.5.提取CLS token的编码
```python
# Classifier "token" as used by standard language architectures
x = x[:, 0]
```
CLS token的编码维度为(n, 768)

## 1.6.送入分类头
```python
x = self.heads(x)

return x
```

其中，self.heads的定义如下：
```python
heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
heads_layers["head"] = nn.Linear(hidden_dim, num_classes)
self.heads = nn.Sequential(heads_layers)
```
其中，hidden_dim=768；num_classes=1000；就是一个全连接层，就可以计算交叉熵Loss了







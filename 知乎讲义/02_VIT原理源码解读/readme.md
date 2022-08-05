本文以pytorch提供的官方 ViT_B_16 为例进行梳理。

## 1.前向传播
![img.png](img.png)
### 1.1.输入图像预处理
预处理的方式和前面写过的ResNet的方式是一样的，这里不加赘述。

### 1.2.VIT的预处理
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
#### 1.2.1.对图像进行分块

patch_size我们设置的是16，因此，

n_h = h // p = 224 // 16 = 14； 

n_w = w // p = 224 // 16 = 14； 

#### 1.2.2.对patch进行编码
x = self.conv_proj(x)
```python
self.conv_proj = nn.Conv2d(
    in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
)
```
其中hidden_dim=768，patch_size=16。

因此，x会由(n, c, h, w) -> (n, hidden_dim, n_h, n_w),即(n, 3, 224, 224) -> (n, 768, 14, 14)

然后，reshape拉直，permute调整通道顺序为(n, n_h*n_w, hidden_dim),即(n, 196, 768)

#### 1.2.3.增加CLS token




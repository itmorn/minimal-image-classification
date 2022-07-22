"""
@Auth: itmorn
@Date: 2022/7/22-15:22
@Email: 12567148@qq.com
"""
import re
import sys
from torch.distributed.run import main
if __name__ == '__main__':
    print(sys.argv)
    a = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    print(a)
    sys.argv[0] = a
    sys.exit(main())
# ;OMP_NUM_THREADS=20
# --nproc_per_node=2 train.py --model=resnet18 --data-path=/data01/zhaoyichen/data/ImageNet1k/ --weights=ResNet18_Weights.IMAGENET1K_V1

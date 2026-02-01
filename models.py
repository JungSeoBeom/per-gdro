import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# -------------------------------------------------------------------------
# Algorithm Metadata
# -------------------------------------------------------------------------
algorithm_attributes = {
    'ERM': {
        'description': 'Empirical Risk Minimization',
        'paper': 'Vapnik (1998)'
    },
    'GroupDRO': {
        'description': 'Group Distributionally Robust Optimization',
        'paper': 'Sagawa et al. (2020)'
    },
    'PerGroupDRO': {
        'description': 'Per-Group Distributionally Robust Optimization',
        'paper': 'Custom Implementation'
    }
}

# -------------------------------------------------------------------------
# 1. MLP (Multi-Layer Perceptron)
# -------------------------------------------------------------------------
class MLP(nn.Module):
    """
    Simple Multi-Layer Perceptron for tabular datasets.
    (Synthetic, COMPAS, Adult, etc.)
   
    Args:
        input_dim (int): Input feature dimension.
        hidden_dim (int): Number of units in hidden layers.
        num_hidden_layers (int): Number of hidden layers.
        output_dim (int): Output dimension (default 1 for binary classification logits).
    """
    def __init__(self, input_dim, hidden_dim=64, num_hidden_layers=2, output_dim=1):
        super(MLP, self).__init__()
       
        layers = []
        in_dim = input_dim
       
        # Hidden Layers
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
           
        # Output Layer
        layers.append(nn.Linear(in_dim, output_dim))
       
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# -------------------------------------------------------------------------
# 2. Simple CNN (for CMNIST)
# -------------------------------------------------------------------------
class SimpleCNN(nn.Module):
    """
    A simple Convolutional Neural Network typically used for Colored MNIST (CMNIST).
    Structure:
    - Conv2d (3 -> 64, 3x3) -> ReLU -> BatchNorm -> MaxPool
    - Conv2d (64 -> 128, 3x3) -> ReLU -> BatchNorm -> MaxPool
    - Conv2d (128 -> 256, 3x3) -> ReLU -> BatchNorm -> MaxPool
    - Global Avg Pool -> Linear
    """
    def __init__(self, num_classes=1):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(256 * 3 * 3, num_classes) # Assuming 28x28 input -> 3x3 feature map

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# -------------------------------------------------------------------------
# 3. Model Factory
# -------------------------------------------------------------------------
def define_model(args, input_dim=None):
    """
    Creates and returns the appropriate PyTorch model based on the dataset.
    """
    # 1. Tabular Datasets -> MLP
    if args.dataset in ['synthetic', 'compas', 'adult', 'insurance']:
        if input_dim is None:
            raise ValueError(f"input_dim must be provided for MLP models (dataset: {args.dataset})")
        
        print(f"[Model] Initializing MLP (In: {input_dim}, Hidden: {args.hsize}, Layers: {args.n_layers})")
        return MLP(
            input_dim=input_dim,
            hidden_dim=args.hsize,
            num_hidden_layers=args.n_layers,
            output_dim=1 
        )

    # 2. Image Datasets (Large) -> ResNet50
    # Waterbirds & CelebA
    elif args.dataset in ['waterbirds', 'celeba']:
        print(f"[Model] Initializing ResNet50 (Pretrained) for {args.dataset}")
        try:
            from torchvision.models import ResNet50_Weights
            model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        except ImportError:
            model = models.resnet50(pretrained=True)
        
        d_features = model.fc.in_features
        model.fc = nn.Linear(d_features, 1)
        return model

    elif args.dataset == 'cmnist':
        print(f"[Model] Initializing ResNet18 (Pretrained) for CMNIST")
        
        try:
            from torchvision.models import ResNet18_Weights
            model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        except ImportError:
            model = models.resnet18(pretrained=True)
            

        old_conv = model.conv1
        new_conv = nn.Conv2d(2, old_conv.out_channels, kernel_size=old_conv.kernel_size, 
                             stride=old_conv.stride, padding=old_conv.padding, bias=old_conv.bias)
        
        with torch.no_grad():
            new_conv.weight[:] = old_conv.weight[:, :2, :, :]
            
        model.conv1 = new_conv

        d_features = model.fc.in_features
        model.fc = nn.Linear(d_features, 1)
        
        return model

    else:
        raise NotImplementedError(f"Model architecture for dataset '{args.dataset}' is not implemented.")
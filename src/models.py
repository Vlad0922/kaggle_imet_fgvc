import torch.nn as nn
import pretrainedmodels


class AvgPool(nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, x.shape[2:])

class Resnet50Body(nn.Module):
    def __init__(self):
        super(Resnet50Body, self).__init__()
        
        self.model = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(self.model.children())[:-2])
        self.avg_pool = AvgPool()
        
        self.input_size = list(self.model.children())[~0].in_features  
        
    def forward(self, x):
        out = self.features(x)
        out = self.avg_pool(out)
        return out.view(out.size(0), -1)
    
class SeResnet50Body(nn.Module):
    def __init__(self):
        super(SeResnet50Body, self).__init__()
        
        model = pretrainedmodels.__dict__['se_resnet50'](num_classes=1000, pretrained='imagenet')
        self.features = nn.Sequential(*list(model.children())[:-2])
        self.avg_pool = AvgPool()
        self.input_size = list(model.children())[~0].in_features
        
    def forward(self, x):
        out = self.features(x)
        out = self.avg_pool(out)
        return out.view(out.size(0), -1)
    
class SeResnext50Body(nn.Module):
    def __init__(self):
        super(SeResnext50Body, self).__init__()
        
        model = pretrainedmodels.__dict__['se_resnext50_32x4d'](pretrained='imagenet')
        self.features = nn.Sequential(*list(model.children())[:-2])
        self.avg_pool = AvgPool()
        self.input_size = list(model.children())[~0].in_features
    
    def forward(self, x):
        out = self.features(x)
        out = self.avg_pool(out)
        return out.view(out.size(0), -1)
    
class SeResnext101Body(nn.Module):
    def __init__(self):
        super(SeResnext101Body, self).__init__()
        
        model = pretrainedmodels.__dict__['se_resnext101_32x4d'](pretrained='imagenet')
        self.features = nn.Sequential(*list(model.children())[:-2])
        self.avg_pool = AvgPool()
        self.input_size = list(model.children())[~0].in_features
    
    def forward(self, x):
        out = self.features(x)
        out = self.avg_pool(out)
        return out.view(out.size(0), -1)
        
class InceptionResnetV2Body(nn.Module):
    def __init__(self):
        super(InceptionResnetV2Body, self).__init__()
        
        model = pretrainedmodels.__dict__['inceptionresnetv2'](pretrained='imagenet')
        self.features = nn.Sequential(*list(model.children())[:-2])
        self.avg_pool = AvgPool()
        self.input_size = list(model.children())[~0].in_features
    
    def forward(self, x):
        out = self.features(x)
        out = self.avg_pool(out)
        return out.view(out.size(0), -1)
        
class Densenet161Body(nn.Module):
    def __init__(self):
        super(Densenet161Body, self).__init__()
        
        model = models.densenet161(pretrained=True)
        self.features = model.features
        self.input_size = model.classifier.in_features
        
    def forward(self, x):
        out = self.features(x)
        out = F.relu(out, inplace=True)
        return F.adaptive_avg_pool2d(out, (1, 1)).view(out.size(0), -1)
    
class Densenet201Body(nn.Module):
    def __init__(self):
        super(Densenet201Body, self).__init__()
        
        self.model = models.densenet201(pretrained=True)
        self.features = self.model.features
        self.input_size = self.model.classifier.in_features
        
    def forward(self, x):
        out = self.features(x)
        out = F.relu(out, inplace=True)
        return F.adaptive_avg_pool2d(out, (1, 1)).view(out.size(0), -1)
    
class InceptionV3Body(nn.Module):
    def __init__(self, use_dropout=False):
        super(InceptionV3Body, self).__init__()
        
        self.model = pretrainedmodels.inceptionv3(pretrained='imagenet')
        self.features = self.model.features
        self.input_size = list(self.model.children())[-1].in_features
        
        self.use_dropout=use_dropout
    
    def forward(self, x):
        out = self.features(x)
        out = F.avg_pool2d(out, kernel_size=8)
        if self.use_dropout:
            out = F.dropout(out, training=self.training)
        out = out.view(out.size(0), -1) 
        
        return out
        
class NeuralNetBaseline(nn.Module):
    def __init__(self, body_func, nclasses):
        super(NeuralNetBaseline, self).__init__()

        self.body, input_size = body_func()
        self.freeze_body(True)
        
        self.head = nn.Linear(input_size, nclasses)
    
    def freeze_body(self, freeze=True):
        for param in self.body.parameters():
            param.requires_grad = not(freeze)
        
    def forward(self, x):
        features = self.body(x)
        features = features.view(features.size(0), -1)
        out = torch.sigmoid(self.head(features))

        return out


class NeuralNetComplexHead(nn.Module):
    def __init__(self, body, nclasses, sigmoid=True, use_drop=False, use_batchnorm=False, input_size=None):
        super(NeuralNetComplexHead, self).__init__()
        self.use_drop = use_drop
        self.use_batchnorm = use_batchnorm
        self.sigmoid = sigmoid
        self.nclasses = nclasses
        
        self.body = body
        
        if input_size is None:
            input_size = self.body.input_size
            
        if not(type(input_size) is list):
            input_size = [input_size]
            
        self._init_head(input_size)
            
        self.freeze_body(True)
        

    def _init_head(self, input_sizes):
        self.fc_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.drop_layers = nn.ModuleList()
        
        for (sz, sz_next) in zip(input_sizes[:-1], input_sizes[1:]):
            bn = nn.BatchNorm1d(sz)
            fc = nn.Linear(sz, sz_next)    
            drop = nn.Dropout(0.5)
            
            self.drop_layers.append(drop)            
            self.fc_layers.append(fc)
            self.bn_layers.append(bn)
        
        self.head_bn = nn.BatchNorm1d(input_sizes[~0])
        self.head_drop = nn.Dropout(0.5)
        self.head = nn.Linear(input_sizes[~0], self.nclasses)
    
    def freeze_body(self, freeze=True):
        for param in self.body.parameters():
            param.requires_grad = not(freeze)
        
    def forward(self, x):
        out = self.body(x)
        
        for (fc, bn, drop) in zip(self.fc_layers, self.bn_layers, self.drop_layers):
            if self.use_batchnorm:
                out = bn(out)
                
            out = F.relu(fc(out))
            
            if self.use_drop:
                out = drop(out)
        
        if self.use_drop:
            out = self.head_drop(out)

        out = self.head(out)
        
        if self.sigmoid:
            out = torch.sigmoid(out)

        return out
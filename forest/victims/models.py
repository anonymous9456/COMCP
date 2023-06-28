import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision.models.resnet import BasicBlock, Bottleneck
from transformers import BertTokenizer, BertModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import OrderedDict

from ..consts import EPSILON


def get_model(name, vocab_size, pretrained=False, pretrained_model=None):
    if name == "textcnn":
        if pretrained:
            raise ValueError("nonsupport")
        else:
            embed_size, kernel_sizes, nums_channels = 100, [3, 4, 5], [100, 100, 100] 
            model = TextCNN(vocab_size, embed_size, kernel_sizes, nums_channels)
    elif name == "LSTM":
        if pretrained:
            raise ValueError("nonsupport")
        else:
            embed_size, num_hiddens, num_layers = 100, 100, 2
            model = BiRNN(vocab_size, embed_size, num_hiddens, num_layers)
    elif "bert" in name:
        model = BERTClassifier(name, 768, 2)
        model.resize_token_embeddings(vocab_size)
    elif name == "defend":
        num_classes, embed_size, latent_dim = 2, 100, 200
        if pretrained: 
            raise NotImplementedError("nonsupport")
        else:
            embedding_matrix = None
            model = dEFENDNet(embedding_matrix, num_classes, vocab_size, embed_size, latent_dim)
    elif name == "discriminator":
        model = Discriminator(class_size=2, pretrained_model=pretrained_model, cached_mode=False)
    else:
        raise NotImplementedError("unsupported classifier")

    return model

def get_model_image(name, dataset_name, pretrained=False):
    if 'CIFAR' in dataset_name:
        if pretrained:
            raise ValueError('Loading pretrained models is only supported for ImageNet.')
        in_channels = 1 if dataset_name == 'MNIST' else 3
        num_classes = 10 if dataset_name in ['CIFAR10', 'MNIST'] else 100
        if 'ResNet' in name:
            model = resnet_picker(name, dataset_name)
        elif name == 'ConvNet64':
            model = convnet(width=64, in_channels=in_channels, num_classes=num_classes)
    elif 'DogVsCat' in dataset_name:
        if name == 'AlexNet':
            model = AlexNet(num_classes=2)

    return model

def resnet_picker(arch, dataset):
    """Pick an appropriate resnet architecture for MNIST/CIFAR."""
    in_channels = 1 if dataset == 'MNIST' else 3
    num_classes = 10
    if dataset in ['CIFAR10', 'MNIST']:
        num_classes = 10
        initial_conv = [3, 1, 1]
    elif dataset == 'CIFAR100':
        num_classes = 100
        initial_conv = [3, 1, 1]
    elif dataset == 'TinyImageNet':
        num_classes = 200
        initial_conv = [7, 2, 3]
    else:
        raise ValueError(f'Unknown dataset {dataset} for ResNet.')

    if arch == 'ResNet20':
        return ResNet(torchvision.models.resnet.BasicBlock, [3, 3, 3], num_classes=num_classes, base_width=16, initial_conv=initial_conv)

class ResNet(torchvision.models.ResNet):
    """ResNet generalization for CIFAR-like thingies.

    This is a minor modification of
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py,
    adding additional options.
    """
    def __init__(self, block, layers, num_classes=10, zero_init_residual=False,
                 groups=1, base_width=64, replace_stride_with_dilation=[False, False, False, False],
                 norm_layer=torch.nn.BatchNorm2d, strides=[1, 2, 2, 2], initial_conv=[3, 1, 1]):
        """Initialize as usual. Layers and strides are scriptable."""
        super(torchvision.models.ResNet, self).__init__()  # torch.nn.Module
        self._norm_layer = norm_layer

        self.dilation = 1
        if len(replace_stride_with_dilation) != 4:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 4-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups

        self.inplanes = base_width
        self.base_width = 64  # Do this to circumvent BasicBlock errors. The value is not actually used.
        self.conv1 = torch.nn.Conv2d(3, self.inplanes, kernel_size=initial_conv[0],
                                     stride=initial_conv[1], padding=initial_conv[2], bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = torch.nn.ReLU(inplace=True)

        layer_list = []
        width = self.inplanes
        for idx, layer in enumerate(layers):
            layer_list.append(self._make_layer(block, width, layer, stride=strides[idx], dilate=replace_stride_with_dilation[idx]))
            width *= 2
        self.layers = torch.nn.Sequential(*layer_list)

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.decoder = torch.nn.Linear(width // 2 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the arch by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    torch.nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    torch.nn.init.constant_(m.bn2.weight, 0)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layers(x)

        x = self.avgpool(x)
        encoding = torch.flatten(x, 1)
        outputs = self.decoder(encoding)

        return { "logits": outputs,
                 "feats": encoding}

def convnet(width=32, in_channels=3, num_classes=10, **kwargs):
    """Define a simple ConvNet. This architecture only really works for CIFAR10."""
    model = torch.nn.Sequential(OrderedDict([
        ('conv0', torch.nn.Conv2d(in_channels, 1 * width, kernel_size=3, padding=1)),
        ('relu0', torch.nn.ReLU()),
        ('conv1', torch.nn.Conv2d(1 * width, 2 * width, kernel_size=3, padding=1)),
        ('relu1', torch.nn.ReLU()),
        ('conv2', torch.nn.Conv2d(2 * width, 2 * width, kernel_size=3, padding=1)),
        ('relu2', torch.nn.ReLU()),
        ('conv3', torch.nn.Conv2d(2 * width, 4 * width, kernel_size=3, padding=1)),
        ('relu3', torch.nn.ReLU()),
        ('pool3', torch.nn.MaxPool2d(3)),
        ('conv4', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
        ('relu4', torch.nn.ReLU()),
        ('pool4', torch.nn.MaxPool2d(3)),
        ('flatten', torch.nn.Flatten()),
        ('linear', torch.nn.Linear(36 * width, num_classes))
    ]))
    return model

class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()

        # load a pre-trained model for the feature extractor
        self.feature_extractor = nn.Sequential(*list(torchvision.models.alexnet(pretrained=True).children())[:-1]).eval()
        self.decoder = nn.Linear(9216, num_classes) # binary classification (num_of_class == 2)

        # fix the pre-trained network
        # for param in self.feature_extractor.parameters():
        #   param.requires_grad = False

    def forward(self, images):
        features = self.feature_extractor(images)
        encoding = torch.flatten(features, 1)
        outputs = self.decoder(encoding)
        return {"logits": outputs,
                "feats": encoding}

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, kernel_sizes, num_channels,
                 **kwargs):
        super(TextCNN, self).__init__(**kwargs)

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.constant_embedding = nn.Embedding(vocab_size, embed_size)

        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Linear(sum(num_channels), 2)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU()

        self.convs = nn.ModuleList()

        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(2 * embed_size, c, k))

    def forward(self, inputs):
        # inputs=[batch_size,sen_len]
        # embeddings=[batch_size,sen_len,emb_dim]
        embeddings = torch.cat((
                    self.embedding(inputs), self.constant_embedding(inputs)), dim=2)

        embeddings = embeddings.permute(0, 2, 1)

        encoding = torch.cat([
            torch.squeeze(self.relu(self.pool(conv(embeddings))), dim=-1)
            for conv in self.convs], dim=1) #encoding.shape=[1,300]
        outputs = self.decoder(self.dropout(encoding))

        return { "logits": outputs,
                 "feats": encoding }

class BiRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, 
                 **kwargs):
        super(BiRNN, self).__init__(**kwargs)
    
        self.embedding = nn.Embedding(vocab_size, embed_size)

        self.encoder = nn.LSTM(embed_size, num_hiddens, num_layers=num_layers,
                                bidirectional=True)
        self.decoder = nn.Linear(4 * num_hiddens, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, inputs):
        # inputs [batch_size,sen_len]
        # embeddings [batch_size,sen_len,emd_dim]
        
        embeddings = self.embedding(inputs.T)

        self.encoder.flatten_parameters()
        
        outputs, _ = self.encoder(embeddings)

        encoding = torch.cat((outputs[0], outputs[-1]), dim=1)
        outs = self.decoder(self.dropout(encoding))
        return { "logits":outs,
                 "feats":encoding }

def create_embeddeding_layer(num_embeddings, embedding_dim, weights_matrix=None, non_trainable=False):
    '''
        function to create a embedding layer from given weight matrix.

        Inputs: 
            weight_matrix (numpy.array) : a weight matrix, with shape = (vocab_size + 1, embedding_dim)
            non_trainable (bool):   arg to set weights for non-training (default: False) 
    '''

    if weights_matrix:
        num_embeddings, embedding_dim = weights_matrix.shape
        # convert weight_matrix numpy.array --> torch.Tensor
        weights_matrix = torch.from_numpy(weights_matrix)

    emb_layer = nn.Embedding(num_embeddings, embedding_dim) #weight.dtype=torch.float32
    # add weights to layer
    if weights_matrix:
        emb_layer.load_state_dict({'weight': weights_matrix})

    if non_trainable:
        emb_layer.weight.requires_grad = False
    else:
        emb_layer.weight.requires_grad = True

    return emb_layer

class AttLayer(nn.Module):
    def __init__(self, input_last=200 ,attention_dim=100):
        """
            Attention layer used for the calcualting attention in word and sentence levels
        """
        super(AttLayer, self).__init__()
        
        self.attention_dim = 100
        self.input_last = 200 
        self.epsilon = 1e-07
        
        #initialize parametres
        self.W = nn.Parameter(torch.Tensor((input_last, attention_dim)))
        self.b = nn.Parameter(torch.Tensor((attention_dim)))
        self.u = nn.Parameter(torch.Tensor((attention_dim, 1)))

        #register params
        self.register_parameter("W", self.W)
        self.register_parameter("b", self.b)
        self.register_parameter("u", self.u)

        #initialize param data
        self.W.data = torch.randn((input_last, attention_dim))
        self.b.data = torch.randn((attention_dim))
        self.u.data = torch.randn((attention_dim, 1))

    def forward(self, x):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        
        uit = torch.tanh(torch.matmul(x, self.W)+ self.b)
        ait = torch.matmul(uit, self.u)
        ait = torch.squeeze(ait, -1)
        ait = torch.exp(ait)
        
        ait = ait/(torch.sum(ait, dim=1, keepdims=True) + self.epsilon)
        ait = torch.unsqueeze(ait, -1)
        weighted_input = x * ait
        output = torch.sum(weighted_input, dim=1)

        return output

class CoAttention(nn.Module):
    """
    Co-attention layer which accept content and comment states and computes co-attention between them and returns the
    weighted sum of the content and the comment states
    """
    def __init__(self, latent_dim = 200):
        super(CoAttention, self).__init__()
        
        self.latent_dim = latent_dim
        self.k = 80
        self.Wl = nn.Parameter(torch.Tensor((self.latent_dim, self.latent_dim)))
        self.Wc = nn.Parameter(torch.Tensor((self.k, self.latent_dim)))
        self.Ws = nn.Parameter(torch.Tensor((self.k, self.latent_dim)))
        self.whs = nn.Parameter(torch.Tensor((1, self.k)))
        self.whc = nn.Parameter(torch.Tensor((1, self.k)))

        #register weights and bias as params
        self.register_parameter("Wl", self.Wl)
        self.register_parameter("Wc", self.Wc)
        self.register_parameter("Ws", self.Ws)
        self.register_parameter("whs", self.whs)
        self.register_parameter("whc", self.whc)

        #initialize data of parameters
        self.Wl.data = torch.randn((self.latent_dim, self.latent_dim))
        self.Wc.data = torch.randn((self.k, self.latent_dim))
        self.Ws.data = torch.randn((self.k, self.latent_dim))
        self.whs.data = torch.randn((1, self.k))
        self.whc.data = torch.randn((1, self.k))
        
    def forward(self, sentence_rep, comment_rep):
        sentence_rep_trans = sentence_rep.transpose(2, 1)
        comment_rep_trans = comment_rep.transpose(2, 1)
        
        L = torch.tanh(torch.matmul(torch.matmul(comment_rep, self.Wl), sentence_rep_trans))
        L_trans = L.transpose(2, 1)

        Hs = torch.tanh(torch.matmul(self.Ws, sentence_rep_trans) + torch.matmul(torch.matmul(self.Wc, comment_rep_trans), L))
        Hc = torch.tanh(torch.matmul(self.Wc, comment_rep_trans)+ torch.matmul(torch.matmul(self.Ws, sentence_rep_trans), L_trans))
        
        As = F.softmax(torch.matmul(self.whs, Hs), dim = 2) 
        Ac = F.softmax(torch.matmul(self.whc, Hc), dim=2)
        
        As = As.transpose(2, 1)
        Ac = Ac.transpose(2, 1)

        co_s = torch.matmul(sentence_rep_trans, As)
        co_c = torch.matmul(comment_rep_trans, Ac)
        co_sc = torch.cat([co_s, co_c], dim=1)

        return torch.squeeze(co_sc, -1)


class dEFENDNet(nn.Module):
    def __init__(self, weight_matrix = None , num_classes = 2, vocab_size = 50257, embedding_dim = 100, latent_dim = 200):
        '''
        Contains Architecture of the dEFEND.

        torch Embedding is independent of input dims, so we can use same embedding
        matrix for both comment and article section.
        
        '''
        super(dEFENDNet,self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.attention_dim = 100
        self.input_last = 200
        self.latent_dim = latent_dim

        self.dropout = nn.Dropout(0.5)

        self.embedding = create_embeddeding_layer(vocab_size, embedding_dim, weight_matrix)

        self.sentence_encoder = nn.GRU(embedding_dim, 100, batch_first=True, bidirectional=True)
        self.comment_encoder = nn.GRU(embedding_dim, 100, batch_first=True, bidirectional=True)
        self.content_encoder = nn.GRU(input_size=2*embedding_dim, hidden_size=100, batch_first = True, bidirectional= True)
        self.attention = AttLayer(self.input_last,self.attention_dim)

        self.coattention = CoAttention(self.latent_dim)
        self.decoder = nn.Linear(2*self.latent_dim, num_classes) 
        self.softamx = nn.Softmax(dim = 1)

        #coattention weight
        self.k = 80
        self.Wl = Variable(torch.rand((self.latent_dim, self.latent_dim), requires_grad = True))
        
        self.Wc = Variable(torch.rand((self.k, self.latent_dim), requires_grad = True))
        self.Ws = Variable(torch.rand((self.k, self.latent_dim), requires_grad = True))
        
        self.whs = Variable(torch.rand((1, self.k), requires_grad = True))
        self.whc = Variable(torch.rand((1, self.k), requires_grad = True))
        

    def forward(self, content, comment):
        # inputs [batch_size, sen_cnt, sen_len]
        # embedding_out [batch_size, sen_cnt, sen_len, emb_dim]

        self.sentence_encoder.flatten_parameters()
        self.comment_encoder.flatten_parameters()
        self.content_encoder.flatten_parameters()

        embedded_content = self.embedding(content)
        embedded_comment = self.embedding(comment)

        embedded_content = embedded_content.transpose(1, 0)
        embedded_comment = embedded_comment.transpose(1, 0)
        
        xa_cache = []
        xc_cache = []

        for sentence, comment in zip(embedded_content,embedded_comment):
            x1, word_lstm_weight = self.sentence_encoder(sentence)
            xa = self.attention(x1) 
            x2, comment_lstm_weight = self.comment_encoder(comment)
            xc = self.attention(x2)

            xa_cache.append(xa)
            xc_cache.append(xc)
            
        xa = torch.stack(xa_cache)
        xc = torch.stack(xc_cache)

        xa = xa.transpose(1, 0)
        xc = xc.transpose(1, 0)
 
        x3, content_lstm_weight = self.content_encoder(xa)

        coatten = self.coattention(x3, xc)

        preds = self.decoder(self.dropout(coatten))

        preds = self.softamx(preds)

        return {"logits": preds,
                "feats": coatten}

    def initHidden(self):
        word_lstm_weight = Variable(torch.zeros(2, self.batch_size, self.embedding_dim))
        comment_lstm_weight = Variable(torch.zeros(2, self.batch_size, self.embedding_dim))
        content_lstm_weight = Variable(torch.zeros(2, self.batch_size, self.embedding_dim))

        return (word_lstm_weight, comment_lstm_weight, content_lstm_weight)

class ClassificationHead(torch.nn.Module):
    def __init__(self, class_size, embed_size):
        super(ClassificationHead, self).__init__()
        
        self.class_size = class_size
        self.embed_size = embed_size
        self.mlp = torch.nn.Linear(embed_size, class_size)

    def forward(self, hidden_state):
        logits = self.mlp(hidden_state)
        return logits


class Discriminator(nn.Module):
    def __init__(
            self,
            class_size=None,
            pretrained_model="gpt2-medium",
            classifier_head=None,
            cached_mode=False,
            ):
        super(Discriminator, self).__init__()

        if "gpt2" in pretrained_model:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
            self.encoder = AutoModelForCausalLM.from_pretrained(pretrained_model)
            self.embed_size = self.encoder.transformer.config.hidden_size
        elif pretrained_model.startswith("bert"):
            self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
            self.encoder = BertModel.from_pretrained(pretrained_model)
            self.embed_size = self.encoder.config.hidden_size
        else:
            raise ValueError(
                    "{} model not yet supported".format(pretrained_model))
        
        if classifier_head:
            self.decoder = classifier_head
        else:
            if not class_size:
                raise ValueError("must specify class_size")
            self.decoder = ClassificationHead(class_size=class_size, embed_size=self.embed_size)
        self.cached_mode = cached_mode
        self.dropout = nn.Dropout(0.5)

    def avg_representation(self, x):
        mask = x.ne(0).unsqueeze(2).repeat(1, 1, self.embed_size).float().detach()
        if hasattr(self.encoder, 'transformer'):# for gpt2
            hidden = self.encoder.transformer(x)[0] #hidden [batch_size, sen_len, emb_size]
        else: # for bert
            hidden = self.encoder(x)[0]
        masked_hidden = hidden * mask
        avg_hidden = torch.sum(masked_hidden, dim=1) / (
                torch.sum(mask, dim=1).detach() + EPSILON)
        
        return masked_hidden, avg_hidden 

    def forward(self, x):
        if self.cached_mode:
            avg_hidden = x
        else:
            avg_hidden = self.avg_representation(x)[1].float()

        logits = self.decoder(self.dropout(avg_hidden))

        return {"logits": logits,
                "feats": avg_hidden}
    

class BERTClassifier(nn.Module):
    def __init__(self, name, hidden_size, num_labels):
        super(BERTClassifier, self).__init__()
        
        self.bert = BertModel.from_pretrained(name)
        classifier_dropout = (0.5)
        self.dropout = nn.Dropout(classifier_dropout)
        self.decoder = nn.Linear(hidden_size, num_labels) 

    def forward(self, input_ids):
        outputs = self.bert(input_ids)
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.decoder(pooled_output)

        return { "logits": logits,
                 "feats": pooled_output}

    def resize_token_embeddings(self, new_num_tokens):
        model_embeds = self.bert.resize_token_embeddings(new_num_tokens)
        return model_embeds
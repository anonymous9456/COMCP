

def training_strategy(model_name, args):
    """Parse training strategy."""
    if model_name == "textcnn":
        defs = TextCNNStrategy(model_name, args)
    elif "bert" in model_name:
        defs = BertStrategy(model_name, args)
    elif model_name == "LSTM":
        defs = BiRNNStrategy(model_name, args)
    elif model_name == "defend":
        defs = DefendStrategy(model_name, args)
    elif model_name == "xlnet":
        defs = XLNetStrategy(model_name, args)
    elif model_name == "AlexNet":
        defs = AlexNetStrategy(model_name, args)
    elif model_name == "discriminator":
        defs = DiscriminatorStrategy(model_name, args)


    return defs


class Strategy:
    """Default usual parameters, not intended for parsing."""

    epochs : int
    batch_size : int
    optimizer : str
    lr : float
    ft_lr: float
    scheduler : str
    weight_decay : float
    validate : int

    def __init__(self, model_name, args):
        """Defaulted parameters. Apply overwrites from args."""
        if args.lr:
            self.lr = args.lr
        if args.epochs:
            self.epochs = args.epochs
        if args.ft_lr:
            self.ft_lr = args.ft_lr
        if args.weight_decay:
            self.weight_decay = args.weight_decay
        if args.ft_epochs:
            self.ft_epochs = args.ft_epochs
        if args.scheduler:
            self.scheduler = args.scheduler

class TextCNNStrategy(Strategy):
    """Most simple stochastic gradient descent.
    """

    def __init__(self, model_name, args):
        """Initialize training hyperparameters."""
        if args.dataset == "pheme":
            self.lr = 0.0005
            self.epochs = 50
            self.ft_lr = 0.001
            self.ft_epochs_clean = 100
            self.ft_epochs = 100
            self.scheduler = 'none'
            self.weight_decay = 1e-4
            self.optimizer = 'Adam'
        elif args.dataset == "fakenewsnet":
            self.lr = 0.0005
            self.epochs = 60
            self.ft_lr = 0.001
            self.ft_epochs_clean = 100
            self.ft_epochs = 100
            self.scheduler = 'linear'
            self.weight_decay = 0
            self.optimizer = 'Adam'
        elif args.dataset == "CED":
            self.lr = 0.01
            self.epochs = 60
            self.ft_lr = 0.001
            self.ft_epochs = 120
            self.ft_epochs_clean = 120
            self.weight_decay = 1e-4
            self.scheduler = 'none'
            self.optimizer = 'AdamW'

        self.batch_size = args.batch_size
        self.validate = 10 
        super().__init__(model_name, args)
    
class BertStrategy(Strategy):
    def __init__(self, model_name, args):
        """Initialize training hyperparameters."""
        if args.dataset == "pheme":
            self.lr = 0.00001
            self.epochs = 10
            self.ft_lr = 0.01 
            self.ft_epochs = 30
            self.ft_epochs_clean = 30
            self.optimizer = 'AdamW'
            self.weight_decay = 1e-4
            self.scheduler = 'none'
        elif args.dataset == "fakenewsnet":
            self.lr = 0.00002
            self.epochs = 15
            self.ft_lr = 0.0001 
            self.ft_epochs = 120
            self.ft_epochs_clean = 120
            self.optimizer = 'AdamW'
            self.weight_decay = 0.1
            self.scheduler = 'none'
        elif args.dataset == "CED":
            self.lr = 0.00001
            self.epochs = 20
            self.ft_lr = 0.0001
            self.ft_epochs = 120
            self.ft_epochs_clean = 120
            self.optimizer = 'AdamW'
            self.weight_decay = 0.1
            self.scheduler = 'none'
        self.batch_size = args.batch_size
        self.validate = 10 
        super().__init__(model_name, args)
    
class BiRNNStrategy(Strategy):
    def __init__(self, model_name, args):
        """Initialize training hyperparameters."""
        if args.dataset=="pheme":
            self.lr = 0.005 
            self.epochs = 40 
            self.ft_lr = 0.02 
            self.ft_epochs = 50
            self.ft_epochs_clean = 50
            self.weight_decay = 1e-4
            self.scheduler = 'none'
        elif args.dataset=="fakenewsnet":
            self.lr = 0.005 
            self.epochs = 20
            self.ft_lr = 0.01 
            self.ft_epochs = 50
            self.ft_epochs_clean = 50
            self.weight_decay = 0.1
            self.scheduler = 'linear'
        elif args.dataset == "CED":
            self.lr = 0.001
            self.epochs = 30
            self.ft_lr = 0.01
            self.ft_epochs = 65
            self.ft_epochs_clean = 65
            self.weight_decay = 1e-4
            self.scheduler = 'linear'
        
        self.batch_size = args.batch_size
        self.optimizer = 'AdamW'
        self.validate = 10
        super().__init__(model_name, args)


class DefendStrategy(Strategy):
    def __init__(self, model_name, args):
        """Initialize training hyperparameters."""
        if args.dataset=="pheme":
            self.lr = 0.01
            self.epochs = 100
            self.ft_lr = 0.01 
            self.ft_epochs = 200
            self.ft_epochs_clean = 200
            self.weight_decay = 0.1
            self.scheduler = 'linear'
        elif args.dataset=="fakenewsnet":
            self.lr = 0.01 
            self.epochs = 80
            self.ft_lr = 0.001
            self.ft_epochs = 100
            self.ft_epochs_clean = 100
            self.weight_decay = 1e-3
            self.scheduler = 'none'
        elif args.dataset=="CED":
            self.lr = 0.001 
            self.epochs = 100
            self.ft_lr = 0.01
            self.ft_epochs = 120
            self.ft_epochs_clean = 120
            self.weight_decay = 1e-3
            self.scheduler = 'none'
        self.batch_size = args.batch_size
        self.optimizer = 'AdamW'
        self.validate = 10 
        super().__init__(model_name, args)


class XLNetStrategy(Strategy):
    def __init__(self, model_name, args):
        """Initialize training hyperparameters."""
        self.lr = 5e-5
        self.ft_lr = 0.01
        self.epochs = 80
        self.batch_size = args.batch_size
        self.optimizer = 'Adam'
        self.scheduler = 'none'
        self.weight_decay = 1e-4
        self.validate = 10
        super().__init__(model_name, args)


class AlexNetStrategy(Strategy):
    def __init__(self, model_name, args):
        """Initialize training hyperparameters."""
        self.lr = 0.1
        self.epochs = 40
        self.ft_lr = 0.01
        self.ft_epochs = 50
        self.batch_size = args.batch_size
        self.optimizer = 'Adam'
        self.scheduler = 'none'
        self.weight_decay = 0
        self.validate = 10
        super().__init__(model_name, args)

class DiscriminatorStrategy(Strategy):
    """Most simple stochastic gradient descent.
    """

    def __init__(self, model_name, args):
        """Initialize training hyperparameters."""
        if args.dataset == "pheme":
            if args.pretrained_model == "gpt2":
                self.lr = 0.0005
                self.epochs = 10
                self.ft_lr = 0.01
                self.ft_epochs = 100
                self.scheduler = 'none'
                self.weight_decay = 1e-4
            elif args.pretrained_model == "bert-base-uncased":
                self.lr = 0.0005
                self.epochs = 30
                self.ft_lr = 0.001
                self.ft_epochs = 100
                self.scheduler = 'linear'
                self.weight_decay = 1e-4
            elif args.pretrained_model == "bert-large-uncased":
                self.lr = 0.0005
                self.epochs = 30
                self.ft_lr = 0.001
                self.ft_epochs = 100
                self.scheduler = 'linear'
                self.weight_decay = 1e-4
        elif args.dataset == "fakenewsnet":
            if args.pretrained_model == "gpt2":
                self.lr = 0.00005
                self.epochs = 10
                self.ft_lr = 0.0001
                self.ft_epochs = 100
                self.scheduler = 'linear'
                self.weight_decay = 0
            elif args.pretrained_model == "bert-base-uncased":
                self.lr = 0.00005
                self.epochs = 10
                self.ft_lr = 0.0005
                self.ft_epochs = 81
                self.scheduler = 'linear'
                self.weight_decay = 1e-4
            elif args.pretrained_model == "bert-large-uncased":
                self.lr = 0.00005
                self.epochs = 10
                self.ft_lr = 0.0005
                self.ft_epochs = 81
                self.scheduler = 'linear'
                self.weight_decay = 1e-4
        elif args.dataset == "CED":
            if args.pretrained_model == "uer/gpt2-chinese-cluecorpussmall":
                self.lr = 0.0005
                self.epochs = 30
                self.ft_lr = 0.001
                self.ft_epochs = 100
                self.scheduler = 'linear'
                self.weight_decay = 1e-4

        self.batch_size = args.batch_size
        self.optimizer = 'Adam'
        self.validate = 10 
        super().__init__(model_name, args)
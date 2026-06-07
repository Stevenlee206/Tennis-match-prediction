import src.models.preco.optim as optim
from src.models.preco.structure import *

class PCgraph(PCmodel):
    """
    Predictive Coding Graph (PCG) class.
    """
    def __init__(self, lr_x: float, T_train: int, structure: PCGStructure,
                 incremental: bool = False, min_delta: float = 0, early_stop: bool = False,
                 use_feedforward_init: bool = True, node_init_std: float = None,
                 ):
        super().__init__(structure, lr_x, T_train, incremental, min_delta, early_stop)

        self.use_feedforward_init = use_feedforward_init
        self.node_init_std = node_init_std

        if use_feedforward_init:
            self.init_hidden = self.init_hidden_feedforward
            if node_init_std is not None:
                raise ValueError('Standard deviation should not be provided when using feedforward.')
        else:
            self.init_hidden = self.init_hidden_random
            if node_init_std is None:
                raise ValueError('Standard deviation must be provided when not using feedforward.')

        self._reset_grad()
        self._reset_params()

    @property
    def hparams(self):
        return {"lr_x": self.lr_x, "T_train": self.T_train, "incremental": self.incremental, "min_delta": self.min_delta,
                "early_stop": self.early_stop, "use_feedforward_init": self.use_feedforward_init, "node_init_std": self.node_init_std}

    @property
    def params(self):
        return {"w": self.w, "b": self.b, "use_bias": self.structure.use_bias}

    @property
    def grads(self):
        return {"w": self.dw, "b": self.db}

    def _reset_grad(self):
        self.dw = None
        self.db = None

    def _reset_params(self):
        self.w = torch.empty(self.structure.N, self.structure.N, device=DEVICE) 
        self.weight_init(self.w)
        if self.structure.mask is not None:
            self.w *= self.structure.mask

        if self.structure.use_bias:
            self.b = torch.empty(self.structure.N, device=DEVICE)
            self.bias_init(self.b)

        self.no_weigths = self.w.shape[0]*self.w.shape[1]
        if self.structure.use_bias:
            self.no_weigths += self.b.shape[0]

    def reset_nodes(self):
        self.e = []
        self.x = []

    def clamp_input(self, inp):
        self.x[:,:self.structure.shape[0]] = inp.clone()

    def clamp_target(self, target):
        self.x[:,-self.structure.shape[2]:] = target.clone()

    def init_hidden_random(self, batch_size):
        lower = self.structure.shape[0]
        upper = -self.structure.shape[2]
        self.x[:,lower:upper] = torch.normal(0, self.node_init_std, size=(batch_size, sum(self.structure.shape[1:-1])), device=DEVICE)

    def init_hidden_feedforward(self, batch_size):
        nodes_partition = get_nodes_partition(self.structure.layers)
        
        for l in range(1, len(nodes_partition)-1): 
            # We assume feedforward-like structure, so we predict using the previous layers
            # nodes up to the current layer
            nodes_prev = np.concatenate(nodes_partition[:l])
            nodes_curr = nodes_partition[l]

            w_l = self.w[nodes_curr,:][:,nodes_prev]
            b_l = self.b[nodes_curr] if self.structure.use_bias else 0
            
            x_prev = self.x[:,nodes_prev]
            
            # Predict values for the current layer
            if isinstance(self.structure, PCG_AMB):
                pred = torch.matmul(self.structure.f(x_prev), w_l.T) + b_l
            elif isinstance(self.structure, PCG_MBA):
                pred = self.structure.f(torch.matmul(x_prev, w_l.T) + b_l)
            
            self.x[:,nodes_curr] = pred

    def forward(self, batch_size):
        # Iterative update for feedforward during test
        for _ in range(self.T_train):
            dEdx = self.structure.grad_x(self.x, self.e, self.w, self.b, train=False)
            self.x[:,self.structure.shape[0]:] -= self.lr_x*dEdx
            self.e = self.x - self.structure.pred(self.x, self.w, self.b)

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def train_updates(self, batch_no=None):
        self.e = self.x - self.structure.pred(self.x, self.w, self.b)

        if self.early_stop:
            early_stopper = optim.EarlyStopper(patience=0, min_delta=self.min_delta)

        for t in range(self.T_train):
            dEdx = self.structure.grad_x(self.x, self.e, self.w, self.b, train=True)
            
            lower = self.structure.shape[0]
            upper = -self.structure.shape[2]
            self.x[:,lower:upper] -= self.lr_x*dEdx

            self.e = self.x - self.structure.pred(self.x, self.w, self.b)

            if self.incremental:
                self.optimizer.step(self.params, self.grads, batch_size=self.x.shape[0])

            if self.early_stop:
                if early_stopper.early_stop( self.get_energy() ):
                    print(f"\nEarly stopping inference at t={t}.")          
                    break

    def update_w(self):
        dEdw = self.structure.grad_w(self.x, self.e, self.w, self.b)
        self.dw = dEdw

        if self.structure.use_bias:
            dEdb = self.structure.grad_b(self.x, self.e, self.w, self.b)
            self.db = dEdb

    def train_supervised(self, X_batch, y_batch, batch_no=None):
        X_batch = to_vector(X_batch)
        y_batch = onehot(y_batch, N=self.structure.shape[-1])

        batch_size = X_batch.shape[0]
        self.reset_nodes()
        self.x = torch.empty(batch_size, self.structure.N, device=DEVICE)
        
        self.clamp_input(X_batch)
        self.init_hidden(batch_size)
        self.clamp_target(y_batch)
        
        self.train_updates(batch_no=batch_no)
        self.update_w()
        if not self.incremental:
            self.optimizer.step(self.params, self.grads, batch_size=batch_size)

    def test_supervised(self, X_batch):
        X_batch = to_vector(X_batch)
        batch_size = X_batch.shape[0]
        
        self.reset_nodes()
        self.x = torch.empty(batch_size, self.structure.N, device=DEVICE)
        self.clamp_input(X_batch)

        if self.structure.num_layers is not None:
            self.init_hidden_feedforward(batch_size)
            # Predict output layer
            nodes_partition = get_nodes_partition(self.structure.layers)
            nodes_prev = np.concatenate(nodes_partition[:-1])
            nodes_curr = nodes_partition[-1]
            w_l = self.w[nodes_curr,:][:,nodes_prev]
            b_l = self.b[nodes_curr] if self.structure.use_bias else 0
            x_prev = self.x[:,nodes_prev]
            if isinstance(self.structure, PCG_AMB):
                pred = torch.matmul(self.structure.f(x_prev), w_l.T) + b_l
            elif isinstance(self.structure, PCG_MBA):
                pred = self.structure.f(torch.matmul(x_prev, w_l.T) + b_l)
            self.x[:,nodes_curr] = pred
        else:
            self.node_init_std = 0.1
            self.init_hidden_random(batch_size)
            self.x[:,-self.structure.shape[2]:] = torch.zeros(batch_size, self.structure.shape[2], device=DEVICE)
            self.e = self.x - self.structure.pred(self.x, self.w, self.b)
            self.forward(batch_size)

        return self.x[:,-self.structure.shape[2]:]

    def get_errors(self):
        return torch.mean(self.e, axis=0)

    def get_energy(self):
        return torch.sum( self.get_errors()**2 ).item()

    def get_weights(self):
        return self.w.clone()
    
    def get_mean_weights(self):
        return torch.mean(self.w)

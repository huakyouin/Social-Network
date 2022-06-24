import math

from typing import Optional, Tuple, Union
from torch_geometric.typing import Adj, Size, OptTensor, Tensor
import torch
from torch import nn
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing

import torch.nn.functional as F

from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import (
    Adj,
    NoneType,
    OptPairTensor,
    OptTensor,
    Size,
)
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax

from torch_geometric.nn.inits import glorot, zeros

def bp():
    import pdb;pdb.set_trace()

class PEGConv(MessagePassing):
    r"""The PEG layer from the `"Equivariant and Stable Positional Encoding for More Powerful Graph Neural Networks" <https://arxiv.org/abs/2203.00199>`_ paper
    
    
    Args:
        in_feats_dim (int): Size of input node features.
        pos_dim (int): Size of positional encoding.
        out_feats_dim (int): Size of output node features.
        edge_mlp_dim (int): We use MLP to make one to one mapping between the relative information and edge weight. 
                            edge_mlp_dim represents the hidden units dimension in the MLP. (default: 32)
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        normalize (bool, optional): Whether to add self-loops and compute
            symmetric normalization coefficients on the fly.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        use_formerinfo (bool): Whether to use previous layer's output to update node features.
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self,
                 in_feats_dim: int,
                 pos_dim: int,
                 out_feats_dim: int,
                 edge_mlp_dim: int = 32,
                 improved: bool = False,
                 cached: bool = False,
                 add_self_loops: bool = True,
                 normalize: bool = True,
                 bias: bool = True,
                 use_formerinfo: bool = False,
                 **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(PEGConv, self).__init__(**kwargs)

        self.in_feats_dim = in_feats_dim
        self.out_feats_dim = out_feats_dim
        self.pos_dim = pos_dim
        self.use_formerinfo = use_formerinfo
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.edge_mlp_dim = edge_mlp_dim

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.weight_withformer = Parameter(
            torch.Tensor(in_feats_dim + in_feats_dim, out_feats_dim))
        self.weight_noformer = Parameter(
            torch.Tensor(in_feats_dim, out_feats_dim))
        self.edge_mlp = nn.Sequential(nn.Linear(1, edge_mlp_dim),
                                      nn.Linear(edge_mlp_dim, 1), nn.Sigmoid())

        if bias:
            self.bias = Parameter(torch.Tensor(out_feats_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.glorot(self.weight_withformer)
        self.glorot(self.weight_noformer)
        self.zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self,
                x: Tensor,
                edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""

        coors, feats = x[:, :self.pos_dim], x[:, self.pos_dim:]

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, feats.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, feats.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache
        else:
            print('We normalize the adjacency matrix in PEG.')

        if isinstance(edge_index, Tensor):
            rel_coors = coors[edge_index[0]] - coors[edge_index[1]]
        elif isinstance(edge_index, SparseTensor):
            rel_coors = coors[edge_index.to_torch_sparse_coo_tensor()._indices()[0]] - coors[edge_index.to_torch_sparse_coo_tensor()._indices()[1]]
        rel_dist = (rel_coors**2).sum(dim=-1, keepdim=True)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        # pos: l2 norms
        hidden_out, coors_out = self.propagate(edge_index,
                                               x=feats,
                                               edge_weight=edge_weight,
                                               pos=rel_dist,
                                               coors=coors,
                                               size=None)

        if self.bias is not None:
            hidden_out += self.bias

        return torch.cat([coors_out, hidden_out], dim=-1)

    def message(self, x_i: Tensor, x_j: Tensor, edge_weight: OptTensor,
                pos) -> Tensor:
        PE_edge_weight = self.edge_mlp(pos)
        return x_j if edge_weight is None else PE_edge_weight * edge_weight.view(
            -1, 1) * x_j

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        """The initial call to start propagating messages.
            Args:
            `edge_index` holds the indices of a general (sparse)
                assignment matrix of shape :obj:`[N, M]`.
            size (tuple, optional) if none, the size will be inferred
                and assumed to be quadratic.
            **kwargs: Any additional data which is needed to construct and
                aggregate messages, and to update node embeddings.
        """
        size = self.__check_input__(edge_index, size)
        coll_dict = self.__collect__(self.__user_args__, edge_index, size,
                                     kwargs)
        msg_kwargs = self.inspector.distribute('message', coll_dict)
        aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
        update_kwargs = self.inspector.distribute('update', coll_dict)

        # get messages
        m_ij = self.message(**msg_kwargs)

        m_i = self.aggregate(m_ij, **aggr_kwargs)

        coors_out = kwargs["coors"]
        hidden_feats = kwargs["x"]
        if self.use_formerinfo:
            hidden_out = torch.cat([hidden_feats, m_i], dim=-1)
            hidden_out = hidden_out @ self.weight_withformer
        else:
            hidden_out = m_i
            hidden_out = hidden_out @ self.weight_noformer

        # return tuple
        return self.update((hidden_out, coors_out), **update_kwargs)

    def glorot(self, tensor):
        if tensor is not None:
            stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
            tensor.data.uniform_(-stdv, stdv)

    def zeros(self, tensor):
        if tensor is not None:
            tensor.data.fill_(0)

    def __repr__(self):
        return '{}({},{},{})'.format(self.__class__.__name__, self.in_feats_dim, self.pos_dim,
                                   self.out_feats_dim)

class GATConv(MessagePassing):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.

    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    _alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.0,
                 add_self_loops: bool = True, bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GATConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        if isinstance(in_channels, int):
            self.lin_l = Linear(in_channels, heads * out_channels, bias=False)
            self.lin_r = self.lin_l
        else:
            self.lin_l = Linear(in_channels[0], heads * out_channels, False)
            self.lin_r = Linear(in_channels[1], heads * out_channels, False)

        self.att_l = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_r = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_l.weight)
        glorot(self.lin_r.weight)
        glorot(self.att_l)
        glorot(self.att_r)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None, return_attention_weights=None):
        # type: (Union[Tensor, OptPairTensor], Tensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], Tensor, Size, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, Size, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        alpha_l: OptTensor = None
        alpha_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = x_r = self.lin_l(x).view(-1, H, C)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            alpha_r = (x_r * self.att_r).sum(dim=-1)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = self.lin_l(x_l).view(-1, H, C)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)
                alpha_r = (x_r * self.att_r).sum(dim=-1)

        assert x_l is not None
        assert alpha_l is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                if size is not None:
                    num_nodes = min(size[0], size[1])
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out = self.propagate(edge_index, x=(x_l, x_r),
                             alpha=(alpha_l, alpha_r), size=size)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


# class GATConv(MessagePassing):
#     r"""The graph attentional operator from the `"Graph Attention Networks"
#     <https://arxiv.org/abs/1710.10903>`_ paper

#     .. math::
#         \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
#         \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

#     where the attention coefficients :math:`\alpha_{i,j}` are computed as

#     .. math::
#         \alpha_{i,j} =
#         \frac{
#         \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
#         [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
#         \right)\right)}
#         {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
#         \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
#         [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
#         \right)\right)}.

#     If the graph has multi-dimensional edge features :math:`\mathbf{e}_{i,j}`,
#     the attention coefficients :math:`\alpha_{i,j}` are computed as

#     .. math::
#         \alpha_{i,j} =
#         \frac{
#         \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
#         [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j
#         \, \Vert \, \mathbf{\Theta}_{e} \mathbf{e}_{i,j}]\right)\right)}
#         {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
#         \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
#         [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k
#         \, \Vert \, \mathbf{\Theta}_{e} \mathbf{e}_{i,k}]\right)\right)}.

#     Args:
#         in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
#             derive the size from the first input(s) to the forward method.
#             A tuple corresponds to the sizes of source and target
#             dimensionalities.
#         out_channels (int): Size of each output sample.
#         heads (int, optional): Number of multi-head-attentions.
#             (default: :obj:`1`)
#         concat (bool, optional): If set to :obj:`False`, the multi-head
#             attentions are averaged instead of concatenated.
#             (default: :obj:`True`)
#         negative_slope (float, optional): LeakyReLU angle of the negative
#             slope. (default: :obj:`0.2`)
#         dropout (float, optional): Dropout probability of the normalized
#             attention coefficients which exposes each node to a stochastically
#             sampled neighborhood during training. (default: :obj:`0`)
#         add_self_loops (bool, optional): If set to :obj:`False`, will not add
#             self-loops to the input graph. (default: :obj:`True`)
#         edge_dim (int, optional): Edge feature dimensionality (in case
#             there are any). (default: :obj:`None`)
#         fill_value (float or Tensor or str, optional): The way to generate
#             edge features of self-loops (in case :obj:`edge_dim != None`).
#             If given as :obj:`float` or :class:`torch.Tensor`, edge features of
#             self-loops will be directly given by :obj:`fill_value`.
#             If given as :obj:`str`, edge features of self-loops are computed by
#             aggregating all features of edges that point to the specific node,
#             according to a reduce operation. (:obj:`"add"`, :obj:`"mean"`,
#             :obj:`"min"`, :obj:`"max"`, :obj:`"mul"`). (default: :obj:`"mean"`)
#         bias (bool, optional): If set to :obj:`False`, the layer will not learn
#             an additive bias. (default: :obj:`True`)
#         **kwargs (optional): Additional arguments of
#             :class:`torch_geometric.nn.conv.MessagePassing`.

#     Shapes:
#         - **input:**
#           node features :math:`(|\mathcal{V}|, F_{in})` or
#           :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
#           if bipartite,
#           edge indices :math:`(2, |\mathcal{E}|)`,
#           edge features :math:`(|\mathcal{E}|, D)` *(optional)*
#         - **output:** node features :math:`(|\mathcal{V}|, H * F_{out})` or
#           :math:`((|\mathcal{V}_t|, H * F_{out})` if bipartite.
#           If :obj:`return_attention_weights=True`, then
#           :math:`((|\mathcal{V}|, H * F_{out}),
#           ((2, |\mathcal{E}|), (|\mathcal{E}|, H)))`
#           or :math:`((|\mathcal{V_t}|, H * F_{out}), ((2, |\mathcal{E}|),
#           (|\mathcal{E}|, H)))` if bipartite
#     """
#     def __init__(
#         self,
#         in_channels: Union[int, Tuple[int, int]],
#         out_channels: int,
#         heads: int = 1,
#         concat: bool = True,
#         negative_slope: float = 0.2,
#         dropout: float = 0.0,
#         add_self_loops: bool = True,
#         edge_dim: Optional[int] = None,
#         fill_value: Union[float, Tensor, str] = 'mean',
#         bias: bool = True,
#         **kwargs,
#     ):
#         kwargs.setdefault('aggr', 'add')
#         super().__init__(node_dim=0, **kwargs)

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.heads = heads
#         self.concat = concat
#         self.negative_slope = negative_slope
#         self.dropout = dropout
#         self.add_self_loops = add_self_loops
#         self.edge_dim = edge_dim
#         self.fill_value = fill_value

#         # In case we are operating in bipartite graphs, we apply separate
#         # transformations 'lin_src' and 'lin_dst' to source and target nodes:
#         if isinstance(in_channels, int):
#             self.lin_src = Linear(in_channels, heads * out_channels,
#                                   bias=False, weight_initializer='glorot')
#             self.lin_dst = self.lin_src
#         else:
#             self.lin_src = Linear(in_channels[0], heads * out_channels, False,
#                                   weight_initializer='glorot')
#             self.lin_dst = Linear(in_channels[1], heads * out_channels, False,
#                                   weight_initializer='glorot')

#         # The learnable parameters to compute attention coefficients:
#         self.att_src = Parameter(torch.Tensor(1, heads, out_channels))
#         self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))

#         if edge_dim is not None:
#             self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
#                                    weight_initializer='glorot')
#             self.att_edge = Parameter(torch.Tensor(1, heads, out_channels))
#         else:
#             self.lin_edge = None
#             self.register_parameter('att_edge', None)

#         if bias and concat:
#             self.bias = Parameter(torch.Tensor(heads * out_channels))
#         elif bias and not concat:
#             self.bias = Parameter(torch.Tensor(out_channels))
#         else:
#             self.register_parameter('bias', None)

#         self.reset_parameters()

#     def reset_parameters(self):
#         self.lin_src.reset_parameters()
#         self.lin_dst.reset_parameters()
#         if self.lin_edge is not None:
#             self.lin_edge.reset_parameters()
#         glorot(self.att_src)
#         glorot(self.att_dst)
#         glorot(self.att_edge)
#         zeros(self.bias)


#     def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
#                 edge_attr: OptTensor = None, size: Size = None,
#                 return_attention_weights=None):
#         # type: (Union[Tensor, OptPairTensor], Tensor, OptTensor, Size, NoneType) -> Tensor  # noqa
#         # type: (Union[Tensor, OptPairTensor], SparseTensor, OptTensor, Size, NoneType) -> Tensor  # noqa
#         # type: (Union[Tensor, OptPairTensor], Tensor, OptTensor, Size, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
#         # type: (Union[Tensor, OptPairTensor], SparseTensor, OptTensor, Size, bool) -> Tuple[Tensor, SparseTensor]  # noqa
#         r"""
#         Args:
#             return_attention_weights (bool, optional): If set to :obj:`True`,
#                 will additionally return the tuple
#                 :obj:`(edge_index, attention_weights)`, holding the computed
#                 attention weights for each edge. (default: :obj:`None`)
#         """
#         # NOTE: attention weights will be returned whenever
#         # `return_attention_weights` is set to a value, regardless of its
#         # actual value (might be `True` or `False`). This is a current somewhat
#         # hacky workaround to allow for TorchScript support via the
#         # `torch.jit._overload` decorator, as we can only change the output
#         # arguments conditioned on type (`None` or `bool`), not based on its
#         # actual value.

#         H, C = self.heads, self.out_channels

#         # We first transform the input node features. If a tuple is passed, we
#         # transform source and target node features via separate weights:
#         if isinstance(x, Tensor):
#             assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
#             x_src = x_dst = self.lin_src(x).view(-1, H, C)
#         else:  # Tuple of source and target node features:
#             x_src, x_dst = x
#             assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"
#             x_src = self.lin_src(x_src).view(-1, H, C)
#             if x_dst is not None:
#                 x_dst = self.lin_dst(x_dst).view(-1, H, C)

#         x = (x_src, x_dst)

#         # Next, we compute node-level attention coefficients, both for source
#         # and target nodes (if present):
#         alpha_src = (x_src * self.att_src).sum(dim=-1)
#         alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
#         alpha = (alpha_src, alpha_dst)

#         if self.add_self_loops:
#             if isinstance(edge_index, Tensor):
#                 # We only want to add self-loops for nodes that appear both as
#                 # source and target nodes:
#                 num_nodes = x_src.size(0)
#                 if x_dst is not None:
#                     num_nodes = min(num_nodes, x_dst.size(0))
#                 num_nodes = min(size) if size is not None else num_nodes
#                 edge_index, edge_attr = remove_self_loops(
#                     edge_index, edge_attr)
#                 edge_index, edge_attr = add_self_loops(
#                     edge_index, edge_attr, fill_value=self.fill_value,
#                     num_nodes=num_nodes)
#             elif isinstance(edge_index, SparseTensor):
#                 if self.edge_dim is None:
#                     edge_index = set_diag(edge_index)
#                 else:
#                     raise NotImplementedError(
#                         "The usage of 'edge_attr' and 'add_self_loops' "
#                         "simultaneously is currently not yet supported for "
#                         "'edge_index' in a 'SparseTensor' form")

#         # edge_updater_type: (alpha: OptPairTensor, edge_attr: OptTensor)
#         alpha = self.edge_update(alpha_j=alpha_dst, alpha_i=alpha_src, edge_attr=edge_attr)

#         # propagate_type: (x: OptPairTensor, alpha: Tensor)
#         out = self.propagate(edge_index, x=x, alpha=alpha, size=size)

#         if self.concat:
#             out = out.view(-1, self.heads * self.out_channels)
#         else:
#             out = out.mean(dim=1)

#         if self.bias is not None:
#             out += self.bias

#         if isinstance(return_attention_weights, bool):
#             if isinstance(edge_index, Tensor):
#                 return out, (edge_index, alpha)
#             elif isinstance(edge_index, SparseTensor):
#                 return out, edge_index.set_value(alpha, layout='coo')
#         else:
#             return out


#     def edge_update(self, alpha_j: Tensor, alpha_i: OptTensor,
#                     edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
#                     size_i: Optional[int]) -> Tensor:
#         # Given edge-level attention coefficients for source and target nodes,
#         # we simply need to sum them up to "emulate" concatenation:
#         alpha = alpha_j if alpha_i is None else alpha_j + alpha_i

#         if edge_attr is not None and self.lin_edge is not None:
#             if edge_attr.dim() == 1:
#                 edge_attr = edge_attr.view(-1, 1)
#             edge_attr = self.lin_edge(edge_attr)
#             edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
#             alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
#             alpha = alpha + alpha_edge

#         alpha = F.leaky_relu(alpha, self.negative_slope)
#         alpha = softmax(alpha, index, ptr, size_i)
#         alpha = F.dropout(alpha, p=self.dropout, training=self.training)
#         return alpha


#     def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
#         return alpha.unsqueeze(-1) * x_j

#     def __repr__(self) -> str:
#         return (f'{self.__class__.__name__}({self.in_channels}, '
#                 f'{self.out_channels}, heads={self.heads})')



# class GATConv_PE(MessagePassing):
#     r"""The graph attentional operator from the `"Graph Attention Networks"
#     <https://arxiv.org/abs/1710.10903>`_ paper

#     .. math::
#         \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
#         \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

#     where the attention coefficients :math:`\alpha_{i,j}` are computed as

#     .. math::
#         \alpha_{i,j} =
#         \frac{
#         \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
#         [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
#         \right)\right)}
#         {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
#         \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
#         [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
#         \right)\right)}.

#     If the graph has multi-dimensional edge features :math:`\mathbf{e}_{i,j}`,
#     the attention coefficients :math:`\alpha_{i,j}` are computed as

#     .. math::
#         \alpha_{i,j} =
#         \frac{
#         \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
#         [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j
#         \, \Vert \, \mathbf{\Theta}_{e} \mathbf{e}_{i,j}]\right)\right)}
#         {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
#         \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
#         [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k
#         \, \Vert \, \mathbf{\Theta}_{e} \mathbf{e}_{i,k}]\right)\right)}.

#     Args:
#         in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
#             derive the size from the first input(s) to the forward method.
#             A tuple corresponds to the sizes of source and target
#             dimensionalities.
#         out_channels (int): Size of each output sample.
#         heads (int, optional): Number of multi-head-attentions.
#             (default: :obj:`1`)
#         concat (bool, optional): If set to :obj:`False`, the multi-head
#             attentions are averaged instead of concatenated.
#             (default: :obj:`True`)
#         negative_slope (float, optional): LeakyReLU angle of the negative
#             slope. (default: :obj:`0.2`)
#         dropout (float, optional): Dropout probability of the normalized
#             attention coefficients which exposes each node to a stochastically
#             sampled neighborhood during training. (default: :obj:`0`)
#         add_self_loops (bool, optional): If set to :obj:`False`, will not add
#             self-loops to the input graph. (default: :obj:`True`)
#         edge_dim (int, optional): Edge feature dimensionality (in case
#             there are any). (default: :obj:`None`)
#         fill_value (float or Tensor or str, optional): The way to generate
#             edge features of self-loops (in case :obj:`edge_dim != None`).
#             If given as :obj:`float` or :class:`torch.Tensor`, edge features of
#             self-loops will be directly given by :obj:`fill_value`.
#             If given as :obj:`str`, edge features of self-loops are computed by
#             aggregating all features of edges that point to the specific node,
#             according to a reduce operation. (:obj:`"add"`, :obj:`"mean"`,
#             :obj:`"min"`, :obj:`"max"`, :obj:`"mul"`). (default: :obj:`"mean"`)
#         bias (bool, optional): If set to :obj:`False`, the layer will not learn
#             an additive bias. (default: :obj:`True`)
#         **kwargs (optional): Additional arguments of
#             :class:`torch_geometric.nn.conv.MessagePassing`.

#     Shapes:
#         - **input:**
#           node features :math:`(|\mathcal{V}|, F_{in})` or
#           :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
#           if bipartite,
#           edge indices :math:`(2, |\mathcal{E}|)`,
#           edge features :math:`(|\mathcal{E}|, D)` *(optional)*
#         - **output:** node features :math:`(|\mathcal{V}|, H * F_{out})` or
#           :math:`((|\mathcal{V}_t|, H * F_{out})` if bipartite.
#           If :obj:`return_attention_weights=True`, then
#           :math:`((|\mathcal{V}|, H * F_{out}),
#           ((2, |\mathcal{E}|), (|\mathcal{E}|, H)))`
#           or :math:`((|\mathcal{V_t}|, H * F_{out}), ((2, |\mathcal{E}|),
#           (|\mathcal{E}|, H)))` if bipartite
#     """
#     def __init__(
#         self,
#         in_channels: Union[int, Tuple[int, int]],
#         pos_dim: int,
#         out_channels: int,
#         heads: int = 1,
#         edge_mlp_dim: int = 32,
#         concat: bool = True,
#         negative_slope: float = 0.2,
#         dropout: float = 0.0,
#         add_self_loops: bool = True,
#         edge_dim: Optional[int] = None,
#         fill_value: Union[float, Tensor, str] = 'mean',
#         bias: bool = True,
#         **kwargs,
#     ):
#         kwargs.setdefault('aggr', 'add')
#         super().__init__(node_dim=0, **kwargs)

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.heads = heads
#         self.concat = concat
#         self.negative_slope = negative_slope
#         self.dropout = dropout
#         self.add_self_loops = add_self_loops
#         self.edge_dim = edge_dim
#         self.fill_value = fill_value
#         self.pos_dim = pos_dim
#         self.edge_mlp_dim = edge_mlp_dim

#         self.edge_mlp = nn.Sequential(nn.Linear(1, edge_mlp_dim),
#                                       nn.Linear(edge_mlp_dim, 1), nn.Sigmoid())

#         # In case we are operating in bipartite graphs, we apply separate
#         # transformations 'lin_src' and 'lin_dst' to source and target nodes:
#         if isinstance(in_channels, int):
#             self.lin_src = Linear(in_channels, heads * out_channels,
#                                   bias=False, weight_initializer='glorot')
#             self.lin_dst = self.lin_src
#         else:
#             self.lin_src = Linear(in_channels[0], heads * out_channels, False,
#                                   weight_initializer='glorot')
#             self.lin_dst = Linear(in_channels[1], heads * out_channels, False,
#                                   weight_initializer='glorot')

#         # The learnable parameters to compute attention coefficients:
#         self.att_src = Parameter(torch.Tensor(1, heads, out_channels))
#         self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))

#         if edge_dim is not None:
#             self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
#                                    weight_initializer='glorot')
#             self.att_edge = Parameter(torch.Tensor(1, heads, out_channels))
#         else:
#             self.lin_edge = None
#             self.register_parameter('att_edge', None)

#         if bias and concat:
#             self.bias = Parameter(torch.Tensor(heads * out_channels))
#         elif bias and not concat:
#             self.bias = Parameter(torch.Tensor(out_channels))
#         else:
#             self.register_parameter('bias', None)

#         self.reset_parameters()

#     def reset_parameters(self):
#         self.lin_src.reset_parameters()
#         self.lin_dst.reset_parameters()
#         if self.lin_edge is not None:
#             self.lin_edge.reset_parameters()
#         glorot(self.att_src)
#         glorot(self.att_dst)
#         glorot(self.att_edge)
#         zeros(self.bias)


#     def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
#                 edge_attr: OptTensor = None, size: Size = None,
#                 return_attention_weights=None):
#         # type: (Union[Tensor, OptPairTensor], Tensor, OptTensor, Size, NoneType) -> Tensor  # noqa
#         # type: (Union[Tensor, OptPairTensor], SparseTensor, OptTensor, Size, NoneType) -> Tensor  # noqa
#         # type: (Union[Tensor, OptPairTensor], Tensor, OptTensor, Size, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
#         # type: (Union[Tensor, OptPairTensor], SparseTensor, OptTensor, Size, bool) -> Tuple[Tensor, SparseTensor]  # noqa
#         r"""
#         Args:
#             return_attention_weights (bool, optional): If set to :obj:`True`,
#                 will additionally return the tuple
#                 :obj:`(edge_index, attention_weights)`, holding the computed
#                 attention weights for each edge. (default: :obj:`None`)
#         """
#         # NOTE: attention weights will be returned whenever
#         # `return_attention_weights` is set to a value, regardless of its
#         # actual value (might be `True` or `False`). This is a current somewhat
#         # hacky workaround to allow for TorchScript support via the
#         # `torch.jit._overload` decorator, as we can only change the output
#         # arguments conditioned on type (`None` or `bool`), not based on its
#         # actual value.

#         H, C = self.heads, self.out_channels

#         coors, feats = x[:, :self.pos_dim], x[:, self.pos_dim:]

#         x = feats
#         # We first transform the input node features. If a tuple is passed, we
#         # transform source and target node features via separate weights:
#         if isinstance(x, Tensor):
#             assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
#             x_src = x_dst = self.lin_src(x).view(-1, H, C)
#         else:  # Tuple of source and target node features:
#             x_src, x_dst = x
#             assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"
#             x_src = self.lin_src(x_src).view(-1, H, C)
#             if x_dst is not None:
#                 x_dst = self.lin_dst(x_dst).view(-1, H, C)

        
#         if isinstance(edge_index, Tensor):
#             rel_coors = coors[edge_index[0]] - coors[edge_index[1]]
#         elif isinstance(edge_index, SparseTensor):
#             rel_coors = coors[edge_index.to_torch_sparse_coo_tensor()._indices()[0]] - coors[edge_index.to_torch_sparse_coo_tensor()._indices()[1]]
#         rel_dist = (rel_coors**2).sum(dim=-1, keepdim=True)

#         x = (x_src, x_dst)

#         # Next, we compute node-level attention coefficients, both for source
#         # and target nodes (if present):
#         alpha_src = (x_src * self.att_src).sum(dim=-1)
#         alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
#         alpha = (alpha_src, alpha_dst)

#         if self.add_self_loops:
#             if isinstance(edge_index, Tensor):
#                 # We only want to add self-loops for nodes that appear both as
#                 # source and target nodes:
#                 num_nodes = x_src.size(0)
#                 if x_dst is not None:
#                     num_nodes = min(num_nodes, x_dst.size(0))
#                 num_nodes = min(size) if size is not None else num_nodes
#                 edge_index, edge_attr = remove_self_loops(
#                     edge_index, edge_attr)
#                 edge_index, edge_attr = add_self_loops(
#                     edge_index, edge_attr, fill_value=self.fill_value,
#                     num_nodes=num_nodes)
#             elif isinstance(edge_index, SparseTensor):
#                 if self.edge_dim is None:
#                     edge_index = set_diag(edge_index)
#                 else:
#                     raise NotImplementedError(
#                         "The usage of 'edge_attr' and 'add_self_loops' "
#                         "simultaneously is currently not yet supported for "
#                         "'edge_index' in a 'SparseTensor' form")

#         # edge_updater_type: (alpha: OptPairTensor, edge_attr: OptTensor)
#         alpha = self.edge_updater(edge_index, alpha=alpha, edge_attr=edge_attr)

#         # propagate_type: (x: OptPairTensor, alpha: Tensor)
#         hidden_out, coors_out  = self.propagate(edge_index, x=x, alpha=alpha, size=size, pos=rel_dist, coors=coors)

#         if self.concat:
#             hidden_out = hidden_out.view(-1, self.heads * self.out_channels)
#         else:
#             hidden_out = hidden_out.mean(dim=1)

#         if self.bias is not None:
#             hidden_out += self.bias

#         if isinstance(return_attention_weights, bool):
#             if isinstance(edge_index, Tensor):
#                 return torch.cat([coors_out, hidden_out], dim=-1), (edge_index, alpha)
#             elif isinstance(edge_index, SparseTensor):
#                 return torch.cat([coors_out, hidden_out], dim=-1), edge_index.set_value(alpha, layout='coo')
#         else:
#             return torch.cat([coors_out, hidden_out], dim=-1)


#     def edge_update(self, alpha_j: Tensor, alpha_i: OptTensor,
#                     edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
#                     size_i: Optional[int]) -> Tensor:
#         # Given edge-level attention coefficients for source and target nodes,
#         # we simply need to sum them up to "emulate" concatenation:
#         alpha = alpha_j if alpha_i is None else alpha_j + alpha_i

#         if edge_attr is not None and self.lin_edge is not None:
#             if edge_attr.dim() == 1:
#                 edge_attr = edge_attr.view(-1, 1)
#             edge_attr = self.lin_edge(edge_attr)
#             edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
#             alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
#             alpha = alpha + alpha_edge

#         alpha = F.leaky_relu(alpha, self.negative_slope)
#         alpha = softmax(alpha, index, ptr, size_i)
#         alpha = F.dropout(alpha, p=self.dropout, training=self.training)
#         return alpha


#     def message(self, x_j: Tensor, alpha: Tensor, pos) -> Tensor:
#         PE_edge_weight = self.edge_mlp(pos)
#         x_j_tmp = alpha.unsqueeze(-1) * x_j
#         x_j_tmp = PE_edge_weight * x_j_tmp
#         return x_j_tmp


#     def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
#         """The initial call to start propagating messages.
#             Args:
#             `edge_index` holds the indices of a general (sparse)
#                 assignment matrix of shape :obj:`[N, M]`.
#             size (tuple, optional) if none, the size will be inferred
#                 and assumed to be quadratic.
#             **kwargs: Any additional data which is needed to construct and
#                 aggregate messages, and to update node embeddings.
#         """
#         size = self.__check_input__(edge_index, size)
#         coll_dict = self.__collect__(self.__user_args__, edge_index, size,
#                                      kwargs)
#         msg_kwargs = self.inspector.distribute('message', coll_dict)
#         aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
#         update_kwargs = self.inspector.distribute('update', coll_dict)

#         # get messages
#         m_ij = self.message(**msg_kwargs)

#         m_i = self.aggregate(m_ij, **aggr_kwargs)

#         coors_out = kwargs["coors"]
#         hidden_feats = kwargs["x"]
#         if self.use_formerinfo:
#             hidden_out = torch.cat([hidden_feats, m_i], dim=-1)
#             hidden_out = hidden_out @ self.weight_withformer
#         else:
#             hidden_out = m_i
#             hidden_out = hidden_out @ self.weight_noformer

#         # return tuple
#         return self.update((hidden_out, coors_out), **update_kwargs)

#     def __repr__(self) -> str:
#         return (f'{self.__class__.__name__}({self.in_channels}, '
#                 f'{self.out_channels}, heads={self.heads})')

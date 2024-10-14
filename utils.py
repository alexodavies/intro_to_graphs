import matplotlib.pyplot as plt
import networkx as nx
from rdkit import Chem
from rdkit.Chem import Draw
import numpy as np
import torch
from torch.nn import Sequential, Linear, ReLU
import torch.nn.functional as F
from torch_geometric.nn import GINEConv
from torch_geometric.nn import global_add_pool
from tqdm import tqdm




import numpy as np


def nx_to_rdkit(graph, labels):
    m = Chem.MolFromSmiles('')
    mw = Chem.RWMol(m)
    atom_index = {}
    for n, d in graph.nodes(data=True):
        label = labels[n]
        atom_index[n] = mw.AddAtom(Chem.Atom(int(label) + 1))
    for a, b, d in graph.edges(data=True):
        start = atom_index[a]
        end = atom_index[b]
        mw.AddBond(start, end, Chem.BondType.SINGLE)


    mol = mw.GetMol()
    return mol

def vis_molecule(molecule):
    im = Draw.MolToImage(molecule, size=(600, 600))

    return im

def vis_from_pyg(data, filename = None, ax = None, save = True):
    """
    Visualise a pytorch_geometric.data.Data object
    Args:
        data: pytorch_geometric.data.Data object
        filename: if passed, this is the filename for the saved image. Ignored if ax is not None
        ax: matplotlib axis object, which is returned if passed

    Returns:

    """
    g, labels = better_to_nx(data)
    if ax is None:
        fig, ax = plt.subplots(figsize = (7,7))
        ax_was_none = True
    else:
        ax_was_none = False

    if "ogbg" not in filename:
        pos = nx.kamada_kawai_layout(g)

        nx.draw_networkx_edges(g, pos = pos, ax = ax)
        if np.unique(labels).shape[0] != 1:
            nx.draw_networkx_nodes(g, pos=pos, node_color=labels,
                                   edgecolors="black",
                                   cmap="Dark2", node_size=64,
                                   vmin=0, vmax=10, ax=ax)
    else:
        im = vis_molecule(nx_to_rdkit(g, labels))
        ax.imshow(im)

    # ax.axis('off')
    # ax.set_title(f"|V|: {g.order()}, |E|: {g.number_of_edges()}")

    plt.tight_layout()

    if not ax_was_none:
        return ax
    elif filename is None:
        plt.show()
    elif save:
        plt.savefig(filename, dpi = 300)
        plt.close()
    else:
        plt.show()

    plt.close()

def better_to_nx(data):
    """
    Converts a pytorch_geometric.data.Data object to a networkx graph,
    robust to nodes with no edges, unlike the original pytorch_geometric version

    Args:
        data: pytorch_geometric.data.Data object

    Returns:
        g: a networkx.Graph graph
        labels: torch.Tensor of node labels
    """
    edges = data.edge_index.T.cpu().numpy()
    if data.x is not None:
        labels = data.x[:,0].cpu().numpy()
    else:
        labels = None

    g = nx.Graph()
    g.add_edges_from(edges)
    if data.x is not None:
        for ilabel in range(labels.shape[0]):
            if ilabel not in np.unique(edges):
                g.add_node(ilabel)

    return g, labels

def vis_grid(datalist, filename, save = True):
    """
    Visualise a set of graphs, from pytorch_geometric.data.Data objects
    Args:
        datalist: list of pyg.data.Data objects
        filename: the visualised grid is saved to this path

    Returns:
        None
    """

    # Trim to square root to ensure square grid
    grid_dim = int(np.sqrt(len(datalist)))

    fig, axes = plt.subplots(grid_dim, grid_dim, figsize=(8,8))

    # Unpack axes
    axes = [num for sublist in axes for num in sublist]

    for i_axis, ax in enumerate(axes):
        ax = vis_from_pyg(datalist[i_axis], ax = ax, filename=filename, save = False)
        ax.axis('off')

    if save:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()



# Generic N-Layer Graph Isomorphism Network encoder
class GraphModel(torch.nn.Module):
    def __init__(self,
                 node_encoder,
                 edge_encoder,
                 gnn_block,
                 output_layer,
                 pooling_type = "standard"):
        super(GraphModel, self).__init__()

        self.node_encoder = node_encoder
        self.edge_encoder = edge_encoder
        self.gnn_block = gnn_block
        self.output_layer = output_layer
        self.pooling_type = pooling_type

        self.init_emb()

    def init_emb(self):
        """
        Initializes the node embeddings.
        """
        for m in self.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, data):
        # print(data.x, data.edge_attr)
        x = data.x
        edge_attr = data.edge_attr
        edge_index = data.edge_index
        batch = data.batch

        x = self.node_encoder.forward(x)
        edge_attr = self.edge_encoder.forward(edge_attr)


        x, xs = self.gnn_block.forward(batch, x, edge_index, edge_attr)

        # compute graph embedding using pooling
        if self.pooling_type == "standard":
            xpool = global_add_pool(x, batch)
            

        elif self.pooling_type == "layerwise":
            xpool = [global_add_pool(x, batch) for x in xs]
            xpool = torch.cat(xpool, 1)
            

        x = self.output_layer.forward(xpool)

        return x



# Generic N-Layer Graph Isomorphism Network encoder
class GNNBlock(torch.nn.Module):
    def __init__(self, emb_dim=300, num_gc_layers=5, drop_ratio=0.0):
        super(GNNBlock, self).__init__()

        self.emb_dim = emb_dim
        self.num_gc_layers = num_gc_layers
        self.drop_ratio = drop_ratio
        self.out_node_dim = self.emb_dim

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        self.convolution = GINEConv

        for i in range(num_gc_layers):
            nn = Sequential(Linear(emb_dim, 2 * emb_dim), torch.nn.BatchNorm1d(2 * emb_dim), ReLU(),
                            Linear(2 * emb_dim, emb_dim))
            conv = GINEConv(nn)
            bn = torch.nn.BatchNorm1d(emb_dim)
            self.convs.append(conv)
            self.bns.append(bn)

    def init_emb(self):
        """
        Initializes the node embeddings.
        """
        for m in self.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, batch, x, edge_index, edge_attr):
        # compute node embeddings using GNN
        xs = []
        for i in range(self.num_gc_layers):
            x = self.convs[i](x, edge_index, edge_attr)
            x = self.bns[i](x)

            if i == self.num_gc_layers - 1:
                # remove relu for the last layer
                x = F.dropout(x, self.drop_ratio, training=self.training)
            else:
                x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)
            xs.append(x)
        return x, xs


class GenericEncoder(torch.nn.Module):
	"""
	A generic encoder module that transforms input features into embeddings.

	Args:
		emb_dim (int): The dimensionality of the output embeddings.
		feat_dim (int): The dimensionality of the input features.
		n_layers (int, optional): The number of layers in the encoder. Defaults to 1.
	"""

	def __init__(self, emb_dim, feat_dim, n_layers=1):
		super(GenericEncoder, self).__init__()
		self.layers = []
		spread_layers = [min(emb_dim, feat_dim) + np.abs(feat_dim - emb_dim) * i for i in range(n_layers - 1)]

		layer_sizes = [feat_dim] + spread_layers + [emb_dim]
		for i in range(n_layers):
			lin = Linear(layer_sizes[i], layer_sizes[i + 1])
			torch.nn.init.xavier_uniform_(lin.weight.data)
			self.layers.append(lin)

			if i != n_layers:
				self.layers.append(ReLU())

		self.model = Sequential(*self.layers)

	def forward(self, x):
		"""
		Forward pass of the encoder.

		Args:
			x (torch.Tensor): Input features.

		Returns:
			torch.Tensor: Output embeddings.
		"""
		return self.model(x.float())

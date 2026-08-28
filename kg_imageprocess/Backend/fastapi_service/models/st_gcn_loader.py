import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional, Union, Dict, Any

# ============================================================================
# 1. Graph Definition for Skeleton Connections
# ============================================================================

class Graph:
    """
    Class to build and handle adjacency matrices for skeleton graph representations.
    Supports popular skeleton layouts: 'coco_17', 'openpose_18', 'ntu_25'.
    """
    def __init__(self, layout: str = 'coco_17', strategy: str = 'spatial', max_hop: int = 1, dilation: int = 1):
        self.max_hop = max_hop
        self.dilation = dilation
        self.get_edge(layout)
        self.hop_dis = self.get_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def get_edge(self, layout: str):
        if layout == 'coco_17':
            self.num_node = 17
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [
                (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7), (7, 9),
                (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13),
                (13, 15), (12, 14), (14, 16)
            ]
            self.center = 0
        elif layout == 'openpose_18':
            self.num_node = 18
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [
                (4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11),
                (10, 9), (9, 8), (11, 8), (12, 8), (14, 0), (15, 0),
                (2, 1), (5, 1), (8, 1), (16, 14), (17, 15), (1, 0)
            ]
            self.center = 1
        elif layout == 'ntu_25':
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [
                (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
                (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
                (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
                (19, 18), (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)
            ]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_link]
            self.center = 20
        else:
            raise ValueError(f"Unsupported layout: {layout}")

        self.edge = self_link + neighbor_link

    def get_hop_distance(self, num_node: int, edge: List[Tuple[int, int]], max_hop: int = 1):
        adj = np.zeros((num_node, num_node))
        for i, j in edge:
            adj[i, j] = 1
            adj[j, i] = 1

        hop_dis = np.zeros((num_node, num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(adj, d) for d in range(max_hop + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)

        for d in range(max_hop, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis

    def get_adjacency(self, strategy: str):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = self.normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[i, j] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                a_root[i, j] = normalize_adjacency[i, j]
                            elif self.hop_dis[j, self.center] < self.hop_dis[i, self.center]:
                                a_close[i, j] = normalize_adjacency[i, j]
                            else:
                                a_further[i, j] = normalize_adjacency[i, j]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            self.A = np.stack(A)
        else:
            raise ValueError(f"Unsupported partitioning strategy: {strategy}")

    @staticmethod
    def normalize_digraph(A):
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i] ** (-1)
        return np.dot(A, Dn)


# ============================================================================
# 2. ST-GCN Modules
# ============================================================================

class ConvTemporalGraphical(nn.Module):
    """Spatial Graph Convolution Layer"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(1, 1),
            padding=(0, 0),
            stride=(1, 1),
            bias=True
        )

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        assert A.size(0) == self.kernel_size
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        return x.contiguous()


class st_gcn_block(nn.Module):
    """Spatio-Temporal Graph Convolutional Block"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int],
                 stride: int = 1, dropout: float = 0.0, residual: bool = True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=(1, 1),
                    stride=(stride, 1)
                ),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, A: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        x = self.gcn(x, A * M)
        x = self.tcn(x) + res
        return self.relu(x)


class STGCNModel(nn.Module):
    """
    ST-GCN Architecture for Skeleton-Based Action Recognition
    Input shape: (N, C, T, V, M)
    """
    def __init__(
        self, 
        in_channels: int = 3, 
        num_class: int = 60, 
        graph_args: Optional[Dict[str, Any]] = None,
        edge_importance_weighting: bool = True, 
        dropout: float = 0.5
    ):
        super().__init__()

        resolved_graph_args = graph_args if graph_args is not None else {'layout': 'coco_17', 'strategy': 'spatial'}

        self.graph = Graph(**resolved_graph_args) # type: ignore
        A_tensor = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A_tensor)

        spatial_kernel_size = A_tensor.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)

        # BatchNorm for input data
        self.data_bn = nn.BatchNorm1d(in_channels * A_tensor.size(1))

        # ST-GCN Layers
        self.st_gcn_networks = nn.ModuleList((
            st_gcn_block(in_channels, 64, kernel_size, 1, residual=False),
            st_gcn_block(64, 64, kernel_size, 1),
            st_gcn_block(64, 64, kernel_size, 1),
            st_gcn_block(64, 64, kernel_size, 1),
            st_gcn_block(64, 128, kernel_size, 2),
            st_gcn_block(128, 128, kernel_size, 1),
            st_gcn_block(128, 128, kernel_size, 1),
            st_gcn_block(128, 256, kernel_size, 2),
            st_gcn_block(256, 256, kernel_size, 1),
            st_gcn_block(256, 256, kernel_size, 1),
        ))

        # Edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(A_tensor.clone().fill_(1.0))
                for _ in self.st_gcn_networks
            ])
        else:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(A_tensor.clone().fill_(1.0), requires_grad=False)
                for _ in self.st_gcn_networks
            ])

        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected input shape: (N, C, T, V, M)
        if x.dim() == 4:
            x = x.unsqueeze(-1)

        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()  # (N, M, V, C, T)
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()  # (N, M, C, T, V)
        x = x.view(N * M, C, T, V)

        # Forward ST-GCN blocks
        A = getattr(self, 'A')
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x = gcn(x, A, importance)

        # Global pooling: (N*M, C, T, V) -> (N*M, C, 1, 1)
        x = F.avg_pool2d(x, x.size()[2:])
        x = self.fcn(x)
        x = x.view(N, M, -1).mean(dim=1)  # Average pooling across persons (M)

        return x


# ============================================================================
# 3. Model Loader and Inference Wrapper Class
# ============================================================================

class STGCNLoader:
    """
    Helper class for loading pre-trained ST-GCN weights and performing inference
    on standard joint coordinates array.
    """
    def __init__(self,
                 checkpoint_path: Optional[str] = None,
                 in_channels: int = 3,
                 num_class: int = 60,
                 layout: str = 'coco_17',
                 strategy: str = 'spatial',
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):

        self.device = torch.device(device)
        self.layout = layout
        self.in_channels = in_channels
        self.num_class = num_class

        graph_args = {'layout': layout, 'strategy': strategy}
        self.model = STGCNModel(
            in_channels=in_channels,
            num_class=num_class,
            graph_args=graph_args
        )

        if checkpoint_path is not None:
            self.load_weights(checkpoint_path)

        self.model.to(self.device)
        self.model.eval()

    def load_weights(self, checkpoint_path: str):
        """Loads weights from checkpoint file (.pt or .pth)"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        cleaned_state_dict = {}
        for k, v in state_dict.items():
            key = k.replace('module.', '')
            cleaned_state_dict[key] = v

        self.model.load_state_dict(cleaned_state_dict, strict=True)
        print(f"[STGCNLoader] Successfully loaded checkpoint from '{checkpoint_path}'")

    def preprocess_input(self, joints: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Converts flexible joint coordinate shapes into target (N, C, T, V, M) tensor format.
        """
        if isinstance(joints, np.ndarray):
            joints = torch.from_numpy(joints).float()

        dims = joints.dim()

        if dims == 3:
            # (T, V, C) -> (1, C, T, V, 1)
            T, V, C = joints.shape
            joints = joints.permute(2, 0, 1).unsqueeze(0).unsqueeze(-1)
        elif dims == 4:
            if joints.shape[0] == self.in_channels:
                # (C, T, V, M) -> (1, C, T, V, M)
                joints = joints.unsqueeze(0)
            else:
                # (N, T, V, C) -> (N, C, T, V, 1)
                N, T, V, C = joints.shape
                joints = joints.permute(0, 3, 1, 2).unsqueeze(-1)
        elif dims == 5:
            pass
        else:
            raise ValueError(f"Invalid input joints shape: {joints.shape}")

        return joints.to(self.device)

    @torch.no_grad()
    def predict(self, joints: Union[np.ndarray, torch.Tensor], return_softmax: bool = True) -> Dict[str, Any]:
        """
        Runs forward inference on joint coordinate sequence.
        """
        x = self.preprocess_input(joints)
        logits = self.model(x)

        if return_softmax:
            probs = F.softmax(logits, dim=-1)
            conf, pred_id = torch.max(probs, dim=-1)
            return {
                'logits': logits.cpu().numpy(),
                'probs': probs.cpu().numpy(),
                'class_id': pred_id.cpu().numpy().tolist(),
                'confidence': conf.cpu().numpy().tolist()
            }
        else:
            pred_id = torch.argmax(logits, dim=-1)
            return {
                'logits': logits.cpu().numpy(),
                'class_id': pred_id.cpu().numpy().tolist()
            }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Testing ST-GCN loader module...")

    N, C, T, V, M = 2, 3, 100, 17, 1
    num_classes = 10

    dummy_joints = np.random.randn(N, C, T, V, M).astype(np.float32)

    loader = STGCNLoader(
        checkpoint_path=None,
        in_channels=C,
        num_class=num_classes,
        layout='coco_17',
        device='cpu'
    )

    results = loader.predict(dummy_joints)
    print("Inference output prediction IDs:", results['class_id'])
    print("Inference confidence scores:", results['confidence'])
    print("Output probabilities shape:", results['probs'].shape)
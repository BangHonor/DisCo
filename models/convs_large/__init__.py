from .gcn_conv import GCNConv
from .sage_conv import SAGEConv
from .gin_conv import GINConv
from .sg_conv import SGConv

__all__ = [
    'GCNConv',
    'SAGEConv',
    'GINConv',
    "SGConv"
]

classes = __all__

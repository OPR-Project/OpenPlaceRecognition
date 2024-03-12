"""Building blocks for the OPR modular system."""
from .cosplace import CosPlace
from .eca import MinkECABasicBlock, MinkECALayer
from .fusion import Add, Concat, GeMFusion
from .gem import GeM, MinkGeM, SeqGeM
from .mixvpr import MixVPR
from .mlp import MLP
from .netvlad import NetVLAD
from .self_attention import SelfAttention

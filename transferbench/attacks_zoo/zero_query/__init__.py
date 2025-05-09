r"""Zero-query attacks.

This module contains the implementation of zero-query ensemble attacks imported from
the TransferAttack library `https://github.com/Trustworthy-AI-Group/TransferAttack.git`.
"""

from functools import partial

from transferbench.types import TransferAttack

from .transfer_attack_wrappers import transfer_attack

EPOCH = 100  # For a fair evaluation.

AdaEA: TransferAttack = partial(transfer_attack, attack_name="adaea", epoch=EPOCH)
CWA: TransferAttack = partial(transfer_attack, attack_name="cwa", epoch=EPOCH)
ENS: TransferAttack = partial(transfer_attack, attack_name="ens", epoch=EPOCH)
LGV: TransferAttack = partial(transfer_attack, attack_name="lgv", epoch=EPOCH)
MBA: TransferAttack = partial(transfer_attack, attack_name="mba", epoch=EPOCH)
SASD_WS: TransferAttack = partial(transfer_attack, attack_name="sasd_ws", epoch=EPOCH)
SMER: TransferAttack = partial(transfer_attack, attack_name="smer", epoch=EPOCH)
SVRE: TransferAttack = partial(transfer_attack, attack_name="svre", epoch=EPOCH)

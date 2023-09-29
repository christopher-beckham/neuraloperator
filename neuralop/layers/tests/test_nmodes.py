import torch

from neuralop.layers.fno_block import FNOBlocks

def test_fnoblock_nmodes():

    sdim = 32
    xfake = torch.randn((4, 8, sdim, sdim))

    block1 = FNOBlocks(
        in_channels=8,
        out_channels=16,
        n_modes=(35,35),
        n_layers=1
    )

    print("Run block1:")
    block1(xfake)


    print("----")

    block2 = FNOBlocks(
        in_channels=8,
        out_channels=16,
        n_modes=(36,36),
        n_layers=1
    )

    print("Run block2:")
    block2(xfake)

    
if __name__ == '__main__':
    test_fnoblock_nmodes()
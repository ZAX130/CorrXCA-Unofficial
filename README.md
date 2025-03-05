# CorrXCA-Unofficial
This repository provides an unofficial implementation of the correlation version framework (CorrXCA) in "Advancing Deformable Medical Image Registration with Multi-axis Cross-covariance Attention" by Meng. 
[[arxiv]](https://arxiv.org/abs/2412.18545)

## Instructions and System Environment
- CUDA 11.8
- Python 3.12
- PyTorch 2.4.1
- Nvidia Tesla V100

For convenience, we are sharing the preprocessed [LPBA](https://drive.usercontent.google.com/download?id=1mFzZDn2qPAiP1ByGZ7EbsvEmm6vrS5WO&export=download&authuser=0) dataset used in our experiments. Once uncompressed, simply modify the "LPBA_path" in `train.py` to the path name of the extracted data. Next, you can execute `train.py` to train the network, and after training, you can run `infer.py` to test the network performance. 

The overall framework and some network components of the code are heavily based on [TransMorph](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration), [VoxelMorph](https://github.com/voxelmorph/voxelmorph), [CorrMLP](https://github.com/MungoMeng/Registration-CorrMLP), [XCiT](https://github.com/facebookresearch/xcit). We are very grateful for their contributions. The file makePklDataset.py shows how to make a pkl dataset from the original LPBA dataset. If you have any other questions about the .pkl format, please refer to the github page of [[TransMorph_on_IXI]](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/IXI/TransMorph_on_IXI.md). 

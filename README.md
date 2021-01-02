# hashNet

## Propose

Through training the hash network, the 121-bit antenna pattern was successfully encoded into a 16-bit hash code. This project uses a paired training method, and the labels used are the similarity matrix between the antenna patterns. The similarity matrix is obtained by calculating the cosine similarity between the two antenna patterns, and the threshold can be selected by oneself.

## Network

The feature extraction is achieved by using a multi-layer 1-dimensional convolutional network or a multi-layer fully connected network, which can be seen in model.py. The main function interface is main.py, which can implement training and testing of the model respectively.

## Reference

> ```
> @inproceedings{zhu2016deep,
>   title={Deep hashing network for efficient similarity retrieval},
>   author={Zhu, Han and Long, Mingsheng and Wang, Jianmin and Cao, Yue},
>   booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
>   volume={30},
>   number={1},
>   year={2016}
> }
> ```


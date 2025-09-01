## Convolutional Neural Networks for Sentence Classification (Python 3.11 Version)

This is a Python 3.11 migration of the code for the paper [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882) (EMNLP 2014).

Original code runs on Pang and Lee's movie review dataset (MR in the paper).
Please cite the original paper when using the data.

### Versions Available

This repository contains two implementations:

1. **`rewrite/` folder**: Python 3.11 compatible versions with Theano (original approach)
2. **NumPy-only version**: Complete NumPy implementation without Theano dependency

### Requirements

#### For Theano Version (rewrite/ folder)

- Python 3.11
- Theano (may require special installation for Python 3.11)
- NumPy
- pandas (for data processing)

#### For NumPy-only Version

- Python 3.11
- NumPy
- SciPy (for convolution operations)
- pandas (for data processing)

Using the pre-trained `word2vec` vectors will also require downloading the binary file from
https://code.google.com/p/word2vec/

### Data Preprocessing

To process the raw data, run

```bash
cd rewrite/
python process_data.py path
```

where path points to the word2vec binary file (i.e. `GoogleNews-vectors-negative300.bin` file).
This will create a pickle object called `mr.p` in the same folder, which contains the dataset
in the right format.

Note: This will create the dataset with different fold-assignments than was used in the paper.
You should still be getting a CV score of >81% with CNN-nonstatic model, though.

### Running the Models

#### Theano Version (Recommended)

```bash
cd rewrite/
python conv_net_sentence.py -nonstatic -rand
python conv_net_sentence.py -static -word2vec
python conv_net_sentence.py -nonstatic -word2vec
```

#### NumPy-only Version (Experimental)

```bash
cd rewrite/
python conv_net_sentence_numpy.py -nonstatic -rand
python conv_net_sentence_numpy.py -static -word2vec
python conv_net_sentence_numpy.py -nonstatic -word2vec
```

### Changes from Original

#### Python 3.11 Migration Changes:

- `import cPickle` → `import pickle`
- `print statements` → `print() functions`
- `xrange()` → `range()`
- `execfile()` → `exec(open().read())`
- `.iteritems()` → `.items()`
- Integer division `/` → `//` where appropriate
- Binary file reading improvements for better Python 3 compatibility

#### NumPy Implementation:

- Replaced Theano tensors with NumPy arrays
- Implemented manual convolution using SciPy
- Added manual max pooling
- Simplified optimizer (basic SGD instead of Adadelta)
- Forward-pass only (no automatic differentiation)

### Performance Notes

- **Theano version**: Should provide same performance as original (if Theano is properly installed)
- **NumPy version**: Significantly slower, intended for educational purposes and environments where Theano cannot be installed

### GPU Support

For the Theano version, GPU support should work the same as the original:

```bash
THEANO_FLAGS=mode=FAST_RUN,device=cuda,floatX=float32 python conv_net_sentence.py -nonstatic -word2vec
```

The NumPy version runs CPU-only.

### File Structure

```
rewrite/
├── process_data.py              # Python 3.11 data preprocessing
├── conv_net_classes.py          # Python 3.11 Theano neural network classes
├── conv_net_sentence.py         # Python 3.11 main training script (Theano)
├── conv_net_sentence_numpy.py   # NumPy-only implementation
└── README.md                    # This file
```

### Known Limitations

1. **Theano compatibility**: Theano is no longer actively maintained and may require special setup for Python 3.11
2. **NumPy implementation**: Missing automatic differentiation, uses simplified optimizer
3. **Performance**: NumPy version is significantly slower than GPU-accelerated Theano

### Recommended Migration Path

For production use, consider migrating to modern frameworks:

- **PyTorch**: Most similar API to Theano
- **TensorFlow/Keras**: Widely supported
- **JAX**: For NumPy-like API with GPU acceleration

### Other Implementations

- **TensorFlow**: https://github.com/dennybritz/cnn-text-classification-tf
- **Torch**: https://github.com/harvardnlp/sent-conv-torch
- **PyTorch**: Multiple implementations available on GitHub

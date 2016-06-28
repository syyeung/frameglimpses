# End-to-end Learning of Action Detection from Frame Glimpses in Videos

By Serena Yeung, Olga Russakovsky, Greg Mori, Li Fei-Fei

This code is a Torch implementation of an end-to-end approach for action detection in videos that learns to directly predict the temporal bounds of actions.  Details of the work can be found [here](http://arxiv.org/abs/1511.06984).

### Citation
    @article{yeung2015end,
      title={End-to-end Learning of Action Detection from Frame Glimpses in Videos},
      author={Yeung, Serena and Russakovsky, Olga and Mori, Greg and Fei-Fei, Li},
      journal={arXiv preprint arXiv:1511.06984},
      year={2015}
    }

### Usage
Run `th train.lua` to train an action detection model for an action class from a video dataset.  The following command line arguments must be specified:

- `train_data_file`: An hdf5 file of training data for all action classes. Each hdf5 dataset `'data/[class_idx]'` should be a `num_examples x seq_len x data_dim` array containing the training data for a class.  `data_dim` is the frame-level input data dimension (4096-dim fc7 features extracted from a VGG16 Net in the paper), `seq_len` is the number of frames in a video chunk (50 in the paper), and `num_examples` is the number of video chunks of the class for training.  The glimpse agent will take `num_glimpses` glimpses in each video chunk of `seq_len` frames.

- `train_meta_file`: A json file containing meta data of the training examples (i.e. the video chunks).  The json file should contain an object `meta` such that for each training example `i`,
  - `meta[i]['vidName'] = video_name`, the name of the video containing the video chunk, e.g. `video_validation_000051`
  - `meta[i]['seq'] = [start_frame, end_frame]`, the start and end frame of the video chunk within the video, e.g. `[509,558]`
  - `meta[i]['dets'][class_idx][det_idx] = [det_start, det_end]`, the start and end frame of all ground truth action instances (clipped) within the video chunk, per-class and *relative to* the video chunk. E.g.,  ``[[[2.0,28.0],[38.0,50.0]],{},{}]`` for the case where class `1` has 2 ground truth instances (clipped to `[1,50]`), and classes `2` and `3` do not have any ground truth instances within the chunk.

- `val_data_file`: An hdf5 file containing validation data for a set of validation videos.  The hdf5 dataset `'data'` should be a `num_examples x seq_len x data_dim array` containing the validation data, broken into `num_examples` video chunks of length `seq_len`.

- `val_meta_file`: A json file containing meta data of the validation video chunks, in the same format as `train_meta_file` above.

- `val_vids_file`: A text file with the names of the videos in the validation set, with one video name per line.

- `class_mapping_file` (optional): A text file mapping contiguous action class indexes in the data and meta files to corresponding non-contiguous action class indexes in the dataset.  This is needed for the Thumos dataset used in the paper, in order to generate the output detection file for Thumos evaluation, since Thumos detection classes are a subset of all Thumos classes.  An example file is included.

For more details, `see util/DataHandler.lua`.  Run `th train.lua --help` to see additional command line options that may be specified.

### Acknowledgments

This code is based partly on code from Wojciech Zaremba's [learning to execute](http://github.com/wojciechz/learning_to_execute), Andrej Karpathy's [char-rnn](http://github.com/karpathy/char-rnn), and Element Research's [dpnn (deep extensions to nn)](http://github.com/Element-Research/dpnn).

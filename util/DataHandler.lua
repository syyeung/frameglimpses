require 'hdf5'
local rapidjson = require('rapidjson')

local DataHandler = torch.class('DataHandler')
function DataHandler:__init(args)
    self.train_data_file = args.train_data_file
    self.train_meta_file = args.train_meta_file
    self.val_data_file = args.val_data_file
    self.val_meta_file = args.val_meta_file
    self.seq_len = args.seq_len
    self.num_classes = args.num_classes
    self.pos_class = args.pos_class
    self.data_dim = args.data_dim

    self.neg_classes = {}
    for i=0,self.num_classes do
        if i ~= self.pos_class then
            table.insert(self.neg_classes, i)
        end
    end

    self.batch_size_pos = args.batch_size_pos
    self.batch_size_neg = args.batch_size_neg
    self.batch_size = self.batch_size_pos + #self.neg_classes*self.batch_size_neg

    self.train_data = hdf5.open(self.train_data_file,'r')
    self.train_meta = rapidjson.load(self.train_meta_file)
    self.examples_per_class = self.train_data:read('data/1'):dataspaceSize()[1]
    self.epoch_batches = math.floor(self.examples_per_class / self.batch_size)

    self.val_data = hdf5.open(self.val_data_file, 'r')
    self.val_meta = rapidjson.load(self.val_meta_file)
    self.num_val_examples = self.val_data:read('data'):dataspaceSize()[1]
    self.num_val_batches = math.ceil(self.num_val_examples / self.batch_size)

    self.data_ptrs = {train_pos=1,train_neg=1,val=1}

    self.batch_data = torch.Tensor()
    self.batch_cont = torch.Tensor()
    self.batch_labels = {}
    self.batch_meta = {}
end


function DataHandler:reset_data_ptr(ptr_name, value)
    self.data_ptrs[ptr_name] = value
end


function DataHandler:next_train_batch()

    self.batch_data:resize(self.batch_size, self.seq_len, self.data_dim):zero()
    self.batch_cont:resize(self.batch_size, self.seq_len, 1):zero()
    self.batch_labels = {}

    local pos_start = self.data_ptrs['train_pos']
    if pos_start + self.batch_size_pos > self.examples_per_class then
        pos_start = 1
    end
    local pos_end = pos_start + self.batch_size_pos - 1

    local neg_start = self.data_ptrs['train_neg']
    if neg_start + self.batch_size_neg > self.examples_per_class then
        neg_start = 1
    end
    local neg_end = neg_start + self.batch_size_neg - 1

    -- fill positives
    -- data
    local batch_idx = 1
    local dset_name = string.format('data/%d', self.pos_class)
    self.batch_data:sub(batch_idx, batch_idx + self.batch_size_pos - 1):copy(self.train_data:read(dset_name):partial({pos_start, pos_end}, {1, self.seq_len}, {1,self.data_dim}))
    batch_idx = batch_idx + self.batch_size_pos
    -- labels
    local class_str = string.format('%d', self.pos_class)
    for i = pos_start, pos_end do
        table.insert(self.batch_labels, self.train_meta[class_str][i]['dets'][self.pos_class])
    end

    -- fill negatives
    for i = 1,#self.neg_classes do
        local neg_class = self.neg_classes[i]
        -- data
        dset_name = string.format('data/%d', neg_class)
        self.batch_data:sub(batch_idx, batch_idx + self.batch_size_neg - 1):copy(self.train_data:read(dset_name):partial({neg_start, neg_end}, {1, self.seq_len}, {1,self.data_dim}))
        batch_idx = batch_idx + self.batch_size_neg
        -- labels
        for i = neg_start, neg_end do
        local neg_class_str = string.format('%d', neg_class)
            table.insert(self.batch_labels, self.train_meta[neg_class_str][i]['dets'][self.pos_class])
        end
    end

    self.batch_cont:fill(1)

    local batch_data = torch.Tensor(self.batch_size, self.seq_len, self.data_dim)
    local batch_labels = {}
    local batch_cont = torch.Tensor(self.batch_size, self.seq_len, 1)
    local shuffle = torch.randperm(self.batch_size)
    for i = 1,self.batch_size do
        batch_data[i] = self.batch_data[shuffle[i]]
        table.insert(batch_labels, self.batch_labels[shuffle[i]])
    end


    self.data_ptrs['train_pos'] = pos_end + 1
    self.data_ptrs['train_neg'] = neg_end + 1

    return batch_data, batch_labels, self.batch_cont
end


function DataHandler:next_val_batch()

    self.batch_data:resize(self.batch_size, self.seq_len, self.data_dim):zero()
    self.batch_cont:resize(self.batch_size, self.seq_len, 1):zero()
    self.batch_labels = {}
    self.batch_meta = {}

    local batch_start = self.data_ptrs['val']
    local batch_end = math.min(self.num_val_examples, batch_start + self.batch_size - 1)
    local num_valid_examples = batch_end - batch_start + 1

    -- data
    self.batch_data:sub(1, num_valid_examples):copy(self.val_data:read('data'):partial({batch_start, batch_end}, {1, self.seq_len}, {1,self.data_dim}))

    -- labels, meta
    for i = batch_start, batch_end do
        table.insert(self.batch_labels, self.val_meta[i]['dets'][self.pos_class])
        table.insert(self.batch_meta, self.val_meta[i])
    end

    self.batch_cont:sub(1, num_valid_examples):fill(1)

    self.data_ptrs['val'] = batch_end + 1

    return self.batch_data, self.batch_labels, self.batch_cont, self.batch_meta

end

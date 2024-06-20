import os
import math
import time
import pickle
import random
import cooler
import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence
from scipy.linalg import block_diag
from scipy.sparse import csr_matrix, coo_matrix, save_npz
from tqdm import tqdm
from multiprocessing import Pool

from va3de.utils.utils import anchor_to_locus, anchor_list_to_dict, sorted_nicely
from va3de.utils.matrix_ops import rebin, convolution, random_walk, OE_norm, VC_SQRT_norm, network_enhance, graph_google


class DataGenerator(Sequence):
    def __init__(self, sparse_matrices, anchor_list, anchor_dict, data_dir, reference, full_reference=None, scool_file=None, scool_contents=None, assembly='hg19', res_name='1M', resolution=1000000,
                 n_clusters=None, class_names=None, dataset_name=None, active_regions=None, preprocessing=[],
                 batch_size=64, normalize=False, standardize=False, binarize=False, downsample=False,
                 simulate_from_bulk=False, bulk_cooler_dir=None, simulate_n=None, simulate_depth=None, real_n=0,
                 depth_norm=False, distance_norm=False, shuffle=True, use_raw_data=True, no_viz=False,
                 rotated_cells=False, resize_amount=1, limit2Mb=8, rotated_offset=0, depth=4, verbose=True, filter_cells_by_depth=True, ignore_chr_filter=False, color_config=None):
        # dictionary of rotated cells represented as sparse matrices
        self.sparse_matrices = sparse_matrices
        self.cell_anchor_vectors = None
        self.anchor_list = anchor_list  # DataFrame of anchors across whole genome
        self.resize_amount = resize_amount  # fraction to resize each cell
        # compute length of matrix diagonal
        # this becomes the width of the rotated matrix
        matrix_len = int(len(self.anchor_list) * self.resize_amount *
                         (math.sqrt(2) if rotated_cells else 1))
        # find next closest power of 2 so we can more easily define downsampling and upsampling
        # the matrix is padded with zeros to reach this new length
        next = matrix_len + (2 ** depth - matrix_len % (2 ** depth))
        # amount to pad matrix to reach new length
        self.matrix_pad = int(next - matrix_len)
        self.matrix_len = int(matrix_len)
        # dictionary m apping each anchor name to its genomic index
        self.anchor_dict = anchor_dict
        self.data_dir = data_dir  # directory containing all cell anchor to anchor files
        self.reference = reference  # DataFrame of cell, cluster, and depth information
        if full_reference is None:
            self.full_reference = reference
        else:
            self.full_reference = full_reference
        self.scool_file = scool_file
        self.scool_contents = scool_contents
        self.assembly = assembly
        self.res_name = res_name
        self.resolution = resolution
        self.verbose = verbose
        self.depth = depth  # depth of autoencoder model
        if dataset_name is None:
            self.dataset_name = 'schic'
        else:
            self.dataset_name = dataset_name
        if filter_cells_by_depth and not simulate_from_bulk:
            # remove cells which do not fit sequencing depth criteria
            self.filter_cells(ignore_chr_filter=ignore_chr_filter)
        # list of preprocessing operations to apply to each matrix generated
        self.preprocessing = preprocessing
        self.max_read_depth = self.reference['depth'].max()
        self.batch_size = batch_size  # size of each batch during training
        # list of all cell file names
        self.cell_list = sorted(list(self.reference.index))
        self.n_cells = len(self.reference)  # total cells to train on
        if class_names is None:
            # list of cluster names
            self.classes = np.array(self.reference['cluster'].unique())
        else:
            self.classes = np.array(class_names)
        self.n_classes = len(self.classes)  # number of classes/clusters
        if n_clusters is None:
            self.n_clusters = self.n_classes
        else:
            self.n_clusters = n_clusters
        self.normalize = normalize  # option to normalize cell matrices to range 0-1
        self.standardize = standardize  # option to convert cells to mean zero unit variance
        self.binarize = binarize
        self.downsample = downsample
        self.simulate_from_bulk = simulate_from_bulk
        self.bulk_cooler_dir = bulk_cooler_dir
        self.simulate_n = simulate_n
        self.simulate_depth = simulate_depth
        self.real_n = real_n  # number of real cells in combined real/simulated datasets
        self.depth_norm = depth_norm  # option to normalize by read depth
        self.distance_norm = distance_norm
        self.shuffle = shuffle  # shuffle the order for generating matrices after each epoch
        # option to use either observed (raw) reads or bias-correction ratio values
        self.use_raw_data = use_raw_data
        self.no_viz = no_viz  # option to skip visualizations
        # option to use rotated matrix representation instead of compressed band
        self.rotated_cells = rotated_cells
        # height at which there are no more values, used at height of rotated matrix
        self.limit2Mb = limit2Mb
        self.rotated_offset = rotated_offset
        # shape of model input layer
        self.input_shape = (
            self.limit2Mb, self.matrix_len + self.matrix_pad, 1)

        # file mapping cluster names to RGB values (only for visualization)
        self.color_config = color_config

        self.example_cell = None

        self.epoch = 0

    def __len__(self):
        """Denotes the number of batches per epoch"""
        if self.batch_size == -1:
            return 1
        else:
            return int(self.n_cells / self.batch_size) + 1

    def get_cell_by_index(self, index, downsample=False, preprocessing=None):
        """Generate one batch of data"""
        if index >= len(self.cell_list):
            index = index % len(self.cell_list)
        cell_name = self.cell_list[index]
        return self.__data_generation(cell_name, downsample=downsample, autoencoder_gen=False, preprocessing=preprocessing)

    def __data_generation(self, cell_name, downsample=False, autoencoder_gen=False, preprocessing=None):
        """Generates data containing batch_size samples"""
        cluster = str(self.reference.loc[cell_name, 'cluster'])
        depth = float(self.reference.loc[cell_name, 'depth'])
        try:
            batch = int(self.reference.loc[cell_name, 'batch'])
        except KeyError:
            batch = 1
        label = np.argwhere(self.classes == cluster)[0]
        cell = self.get_compressed_band_cell(cell_name, preprocessing=preprocessing)
        if cell is None:  # cell has no reads
            # it will be skipped when loading (need better solution when using generator)
            return cell, label, depth, batch
        else:
            cell = cell.A
            cell = np.expand_dims(cell, -1)  # add channel and batch dimension
            if downsample:
                new_x = self.downsample_mat(cell)
                cell_downsample = np.expand_dims(new_x, -1)
            else:
                cell_downsample = cell
            if self.binarize:
                cell = cell > 0
                cell = np.array(cell)
            if self.depth_norm:
                depth = int(self.reference.loc[cell_name, 'depth'])
                cell /= depth
            if self.standardize:
                cell = (cell - cell.mean()) / cell.std()
            if self.normalize:
                cell /= cell.max()
            if self.example_cell is None:
                self.example_cell = np.expand_dims(cell, 0)
            #cell = np.nan_to_num(cell)
            if autoencoder_gen:
                if downsample:
                    return cell_downsample, cell
                else:
                    return cell, cell
            else:
                return cell, label, depth, batch, cell_name

    def __getitem__(self, index):
        """Generate one batch of data"""
        batch_i = index
        x_batch = []
        y_batch = []
        if self.batch_size > 0:
            iter = range(batch_i * self.batch_size, batch_i *
                         self.batch_size + self.batch_size)
        else:
            iter = range(0, len(self.cell_list))
        for i in iter:
            if i >= len(self.cell_list):
                i = i % self.batch_size
            cell_name = self.cell_list[i]
            x, y = self.__data_generation(
                cell_name, downsample=self.downsample, autoencoder_gen=True)
            x_batch.append(x)
            y_batch.append(y)
        return np.array(x_batch), np.array(y_batch)

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        pass
        # if self.shuffle:  # randomize cell order for next epoch
        #     random.shuffle(self.cell_list)

    def get_simulated_pixels(self, cell_name, bulk_loops, chr_offsets, anchor_ref, downsample_percent):
        weights = bulk_loops['obs'].values / bulk_loops['obs'].sum()
        # sample with replacement (why we reset obs to 1)
        loops = bulk_loops.sample(
            frac=downsample_percent, weights=weights, replace=True, ignore_index=True)
        anchor_dict = anchor_ref['start'].to_dict()
        anchor_chr_dict = anchor_ref['chr'].to_dict()
        loops = self.bin_pixels(loops, anchor_dict, anchor_chr_dict,
                                chr_offsets, self.resolution, key='count', use_chr_offset=True)
        loops.rename(columns={'bin1': 'a1', 'bin2': 'a2',
                     'count': 'obs'}, inplace=True)
        return loops, cell_name

    def get_cell_pixels(self, cell_name, alt_dir=None, return_cellname=False):
        if self.data_dir is None and alt_dir is None:  # read from .scool
            try:
                c = cooler.Cooler(f"{self.scool_file}::/cells/{cell_name}")
            except Exception as e:
                try:
                    new_cellname = cell_name.replace(self.res_name, f"comb.{self.res_name}")
                    c = cooler.Cooler(f"{self.scool_file}::/cells/{new_cellname}")
                except Exception as e:
                    try:
                        new_cellname = cell_name.replace(self.res_name, f"3C.{self.res_name}")
                        c = cooler.Cooler(f"{self.scool_file}::/cells/{new_cellname}")
                    except Exception as e:
                        print(e)
                        pass
            loops = c.pixels()[:]
            loops.rename(
                columns={'bin1_id': 'a1', 'bin2_id': 'a2', 'count': 'obs'}, inplace=True)
        else:
            use_dir = self.data_dir if alt_dir is None else alt_dir
            try:
                if self.res_name == 'frag':
                    frag_file = 'frag_loop.' + \
                        cell_name.split('.')[0] + '.cis.filter'
                    loops = pd.read_csv(os.path.join(use_dir, frag_file), delimiter='\t',
                                        names=['a1', 'a2', 'obs', 'exp'], usecols=['a1', 'a2', 'obs'])
                else:
                    loops = pd.read_csv(os.path.join(use_dir, cell_name), delimiter='\t',
                                        names=['a1', 'a2', 'obs', 'exp'], usecols=['a1', 'a2', 'obs'])
            except Exception as e:
                try:
                    loops = pd.read_csv(os.path.join(use_dir, cell_name), delimiter='\t',
                                        names=['a1', 'a2', 'obs'])
                except Exception as e:
                    loops = pd.read_csv(os.path.join(use_dir, cell_name.replace(self.res_name, f"3C.{self.res_name}")), delimiter='\t',
                                        names=['a1', 'a2', 'obs'])
                                        
            loops.dropna(inplace=True)
        loops['a1'] = loops['a1'].astype(str)
        loops['a2'] = loops['a2'].astype(str)
        if return_cellname:
            return loops, cell_name
        else:
            return loops

    def update_from_scool(self, scool_file, keep_depth=False):
        # updates dataset based on existing cooler file
        # used for simulated data where we need to update the dataset to a new simulated version
        print('Setting dataset to', scool_file)
        self.scool_file = scool_file
        content_of_scool = cooler.fileops.list_coolers(scool_file)
        self.scool_contents = content_of_scool
        c = cooler.Cooler(f"{scool_file}::{content_of_scool[0]}")
        anchor_list = c.bins()[:]
        anchor_list = anchor_list[['chrom', 'start', 'end']]
        anchor_list['anchor'] = np.arange(len(anchor_list))
        anchor_list['anchor'] = anchor_list['anchor'].astype(str)
        anchor_list.rename(columns={'chrom': 'chr'}, inplace=True)
        # convert to anchor --> index dictionary
        anchor_dict = anchor_list_to_dict(anchor_list['anchor'].values)
        self.anchor_list = anchor_list
        self.anchor_dict = anchor_dict
        if not keep_depth:
            for cell_name in tqdm(self.cell_list):
                self.reference.loc[cell_name, 'depth'] = self.get_cell_pixels(cell_name)[
                    'obs'].sum()
        matrix_len = int(len(self.anchor_list) * self.resize_amount *
                         (math.sqrt(2) if self.rotated_cells else 1))
        # find next closest power of 2 so we can more easily define downsampling and upsampling
        # the matrix is padded with zeros to reach this new length
        next = matrix_len + (2 ** self.depth - matrix_len % (2 ** self.depth))
        # amount to pad matrix to reach new length
        self.matrix_pad = int(next - matrix_len)
        self.matrix_len = int(matrix_len)
    
    
    def write_scool(self, out_file, simulate=False, append_simulated=False, n_proc=1, downsample_frac=None):
        if simulate:
            coolers = []
            bulk_loops = []
            downsample_fracs = []
            cool_files = os.listdir(self.bulk_cooler_dir)
            for file in cool_files:
                if file.endswith('.mcool'):
                    c = cooler.Cooler(os.path.join(
                        self.bulk_cooler_dir, file + '::resolutions/10000'))
                else:
                    c = cooler.Cooler(os.path.join(self.bulk_cooler_dir, file))
                coolers.append(c)
                pixels = c.pixels()[:]
                pixels.rename(
                    columns={'bin1_id': 'a1', 'bin2_id': 'a2', 'count': 'obs'}, inplace=True)
                pixels.dropna(inplace=True)
                bulk_loops.append(pixels)
                downsample_fracs.append(
                    self.simulate_depth / pixels['obs'].sum())
        if simulate and not append_simulated:
            chr_offsets, uniform_bins = self.get_uniform_bins(self.resolution)
            uniform_bins.to_csv(
                f'bins_{self.dataset_name}.tsv', sep='\t', index=False)
            bins = pd.DataFrame()
            bins['chrom'] = uniform_bins['chr']
            bins['start'] = uniform_bins['start']
            bins['end'] = uniform_bins['end']
            bins['weight'] = 1

            anchor_ref = self.anchor_list.set_index(
                'anchor').dropna().reset_index(drop=True)
        else:
            bins = pd.DataFrame()
            bins['chrom'] = self.anchor_list['chr']
            bins['start'] = self.anchor_list['start']
            bins['end'] = self.anchor_list['end']
            bins['weight'] = 1
            if append_simulated:
                chr_offsets = {}
                prev_offset = 0
                for chr_name in sorted_nicely(bins['chrom'].unique()):
                    chr_offsets[chr_name] = prev_offset
                    prev_offset += int(
                        self.anchor_list.loc[self.anchor_list['chr'] == chr_name, 'end'].max() / self.resolution)
                print(chr_offsets)
                anchor_ref = self.anchor_list.set_index(
                    'anchor').dropna().reset_index(drop=True)

        loop_list = []
        cell_list = []
        simulated_i = 0
        def generate_pixels(chunksize=1000):
            name_pixel_dict = {}
            get_row_indices = np.vectorize(anchor_to_locus(self.anchor_dict))
            for cell_i, cell_name in tqdm(enumerate(sorted(self.cell_list)), total=len(self.cell_list)):
                if simulate and self.reference.loc[cell_name, 'type'] == 'simulated':
                    bulk_loops_idx = int(simulated_i / self.simulate_n)
                    simulated_i += 1
                    loops, _ = self.get_simulated_pixels(
                        cell_name, bulk_loops[bulk_loops_idx], chr_offsets, anchor_ref, downsample_fracs[bulk_loops_idx])
                else:
                    loops = self.get_cell_pixels(cell_name)
                if downsample_frac is not None:  # downsample each cell by some percentage
                    weights = loops['obs']
                    weights[weights <= 0] = 0
                    total = loops['obs'].sum()
                    loops['new_count'] = 1
                    loops = loops.sample(
                        n=int(total * downsample_frac), replace=True, weights=weights)
                    loops = loops[['a1', 'a2', 'new_count']]
                    loops = loops.groupby(['a1', 'a2']).sum().reset_index()
                    loops.rename(columns={'new_count': 'obs'}, inplace=True)

                #loop_list.append(loops)
                #cell_list.append(cell_name)

            #for cell_name, loops in tqdm(sorted(zip(cell_list, loop_list), key=lambda x: x[0])):
                if simulate and not append_simulated:
                    self.anchor_dict = anchor_list_to_dict(
                        uniform_bins['anchor'].values)
                    a1_mask = loops['a1'].isin(uniform_bins['anchor'])
                    a2_mask = loops['a2'].isin(uniform_bins['anchor'])
                    loops = loops[a1_mask & a2_mask]
                else:
                    a1_mask = loops['a1'].isin(self.anchor_list['anchor'])
                    a2_mask = loops['a2'].isin(self.anchor_list['anchor'])
                    loops = loops[a1_mask & a2_mask]
                if len(loops) == 0:  # cell has no reads
                    if self.verbose:
                        print('No reads, skipping...')
                        continue
                rows = get_row_indices(
                    loops['a1'].values)  # convert anchor names to row indices
                cols = get_row_indices(
                    loops['a2'].values)  # convert anchor names to column indices
                pixels = pd.DataFrame()
                pixels['bin1_id'] = rows
                pixels['bin2_id'] = cols
                # if 'pfc' in self.dataset_name:
                #     pixels['bin2_id'] = pixels['bin2_id'] - 1
                bad_loops_mask = pixels.apply(
                    lambda row: row['bin1_id'] > row['bin2_id'], axis=1)
                bad_pixels = pixels[bad_loops_mask].to_numpy()
                pixels.loc[bad_loops_mask, 'bin1_id'] = bad_pixels[..., 1]
                pixels.loc[bad_loops_mask, 'bin2_id'] = bad_pixels[..., 0]
                pixels['count'] = loops['obs']
                pixels = pixels.groupby(['bin1_id', 'bin2_id'])['count'].max().reset_index()
                try:
                    # remove any unnecessary zeros
                    pixels = pixels[pixels['count'] > 0].reset_index()
                except KeyError:
                    print(pixels)
                name_pixel_dict[cell_name] = pixels
                if len(name_pixel_dict) >= chunksize:
                    yield name_pixel_dict
                    name_pixel_dict = {}
            yield name_pixel_dict
        if downsample_frac is not None:
            out_file = out_file.replace(
                f'{self.res_name}.scool', f'{downsample_frac:.2f}_{self.res_name}.scool')
        try:
            os.remove(out_file)
        except FileNotFoundError:
            pass
        for i, name_pixel_dict in enumerate(generate_pixels()):
            print(i)
            cooler.create_scool(out_file, bins, name_pixel_dict, mode='a')

    def write_binned_scool(self, out_file, factor, new_res_name):
        bins = None 
        name_pixel_dict = {}
        for cool_content in tqdm(self.scool_contents):
            cellname = cool_content.split('/')[-1]
            base_uri = f"{self.scool_file}::{cool_content}"
            out_uri = f"tmp.cool"
            c = cooler.Cooler(base_uri)
            cooler.coarsen_cooler(base_uri, out_uri, factor=factor, chunksize=10000000)
            c = cooler.Cooler(out_uri)
            if bins is None:
                bins = c.bins()[:]
            name_pixel_dict[cellname.replace(self.res_name, new_res_name)] = c.pixels()[:]
        os.remove(out_uri)
        cooler.create_scool(out_file, bins, name_pixel_dict)

    def write_pseudo_bulk_coolers(self, out_dir='data/coolers'):
        os.makedirs(out_dir, exist_ok=True)
        for celltype in self.reference['cluster'].unique():
            print(celltype)
            cools = []
            for cell_name in tqdm(self.cell_list):
                if self.reference.loc[cell_name, 'cluster'] == celltype:
                    cools.append(f"{self.scool_file}::/cells/{cell_name}")
            cooler.merge_coolers(os.path.join(out_dir, f'{celltype}_{self.res_name}.cool'), cools, mergebuf=1000000)
            c = cooler.Cooler(os.path.join(out_dir, f'{celltype}_{self.res_name}.cool'))
            cooler.balance_cooler(c, chunksize=10000000, cis_only=True, store=True)

    def get_cell_anchor_vectors(self, load=True):
        current_dir = os.listdir('.')
        if load and 'cell_anchor_vectors.npy' in current_dir and 'cell_labels.npy' in current_dir and 'cell_depths.npy' in current_dir:
            return np.load('cell_anchor_vectors.npy'), np.load('cell_labels.npy'), np.load('cell_depths.npy')
        else:
            self.cell_anchor_vectors = []
            depths = []
            batches = []
            labels = []
            for cell_i, cell_name in enumerate(sorted(self.cell_list)):
                cluster = str(self.reference.loc[cell_name, 'cluster'])
                batch = int(self.reference.loc[cell_name, 'batch'])
                depth = float(self.reference.loc[cell_name, 'depth'])
                label = np.argwhere(self.classes == cluster)[0]
                if load:
                    loops = self.get_cell_pixels(cell_name)
                    self.cell_anchor_vectors.append(loops['obs'].values)
                depths.append(depth)
                batches.append(batch)
                labels.append(label)
            return np.array(self.cell_anchor_vectors), np.array(labels), np.array(depths), np.array(batches)

    def filter_cells(self, min_depth=5000, saved_bad_cells='inadequate_cells.npy', ignoreXY=True, load=True, ignore_chr_filter=False):
        """scHiCluster recommends filtering cells with less than 5k contacts and cells where any chromosome
           does not have x reads for a chromosome length of x (Mb)"""
        if 'downsample' in self.dataset_name:
            remove_cells = []
            valid_cells = os.listdir(self.data_dir)
            for i, (cell_name, row) in enumerate(self.reference.iterrows()):
                if cell_name not in valid_cells:
                    remove_cells.append(cell_name)
            # drop all inadequate cells
            self.reference.drop(remove_cells, inplace=True, errors='ignore')
            if self.verbose:
                print('Cells after filtering: %d' % len(self.reference))
            return
        os.makedirs('data/inadequate_cells', exist_ok=True)
        if self.verbose:
            print('Cells before filtering: %d' % len(self.reference))
        saved_bad_cells = saved_bad_cells.replace(
            '.npy', '_%s.npy' % self.dataset_name)
        self.reference = self.reference[self.reference['depth'] >= min_depth].copy()
        self.full_reference.loc[self.full_reference['depth'] < min_depth, 'filtered_reason'] = 'reference depth < min_depth'
        if ignore_chr_filter:
            return
        if self.verbose:
            print('Cells before filtering by chr: %d' % len(self.reference))
        if saved_bad_cells in os.listdir('data/inadequate_cells') and not ignore_chr_filter:
            remove_cells = np.load(os.path.join(
                'data/inadequate_cells', saved_bad_cells))
            if remove_cells.size == 0:
                return
            suffix = remove_cells[0][remove_cells[0].rfind('.') + 1:]
            remove_cells = [s.replace(suffix, self.res_name)
                            for s in remove_cells]
            np.save(os.path.join('data/inadequate_cells',
                    saved_bad_cells), np.array(remove_cells))
        else:
            chr_list = list(pd.unique(self.anchor_list['chr']))
            chr_anchor_dict = {}
            chr_length_dict = {}
            remove_cells = []  # stores names (indices) of cells to be removed
            if self.data_dir is None:
                content_of_scool = cooler.fileops.list_coolers(self.scool_file)
            for i, (cell_name, row) in tqdm(enumerate(self.reference.iterrows()), total=len(self.reference), desc='Filtering cells'):
                # if using a cooler, check if cell is present
                if self.data_dir is None:
                    if '/cells/' + cell_name not in content_of_scool:
                        print('Cannot find cell %s in .scool, filtering out...' %
                          cell_name)
                        remove_cells.append(cell_name)
                        continue
                try:
                    anchor_to_anchor = self.get_cell_pixels(cell_name)
                except FileNotFoundError as e:
                    print("Could not load", cell_name)
                    remove_cells.append(cell_name)
                    continue
                except UnboundLocalError as e:
                    print("Could not load", cell_name)
                    remove_cells.append(cell_name)
                    continue
                except KeyError as e:  # cannot find in .scool
                    print('Cannot find cell %s in .scool, filtering out...' %
                          cell_name)
                    remove_cells.append(cell_name)
                    continue
                if not ignore_chr_filter:
                    for chr_name in reversed(chr_list):  # reverse to maybe catch empty chroms early
                        if ignoreXY and ('chrX' in chr_name or 'chrY' in chr_name):
                            continue
                        if chr_name in chr_anchor_dict.keys() and chr_name in chr_length_dict.keys():  # reload chr data to save time
                            chr_anchors = chr_anchor_dict[chr_name]
                            chr_length = chr_length_dict[chr_name]
                        else:
                            chr_anchors = self.anchor_list[self.anchor_list['chr'] == chr_name]
                            chr_length = int(
                                (chr_anchors['end'].max() - chr_anchors['start'].min()) / 1e6)
                            chr_anchor_dict[chr_name] = chr_anchors
                            chr_length_dict[chr_name] = chr_length
                        a1_mask = anchor_to_anchor['a1'].isin(chr_anchors['anchor'])
                        a2_mask = anchor_to_anchor['a2'].isin(chr_anchors['anchor'])
                        chr_reads = int(anchor_to_anchor.loc[a1_mask & a2_mask, 'obs'].sum())
                        if chr_reads < chr_length:
                            if self.verbose:
                                print(len(remove_cells), 'Dropping', i,
                                      chr_name, chr_length, chr_reads, cell_name)
                            remove_cells.append(cell_name)
                            break
            np.save(os.path.join('data/inadequate_cells',
                    saved_bad_cells), np.array(remove_cells))
        # drop all inadequate cells
        self.reference.drop(remove_cells, inplace=True, errors='ignore')
        self.full_reference.loc[self.full_reference['cell'].isin(remove_cells), 'filtered_reason'] = 'chr_reads < chr_length'
        self.full_reference.to_csv(f'data/{self.dataset_name}_filtered_ref', index=False, sep='\t')
        if self.verbose:
            filter_reasons = self.full_reference['filtered_reason'].unique()
            for reason in filter_reasons:
                print(reason, ':', self.full_reference['filtered_reason'].value_counts()[reason])
            print(f'Reference file with filtering criteria saved to data/{self.dataset_name}_filtered_ref')

    def downsample_mat(self, mat, p=0.5, minp=0.1, maxp=0.99):
        if random.random() >= p:
            # array to store new downsampled batch
            new_x = np.zeros(mat[..., 0].shape)
            # uniformly sample a downsampling percent
            downsample_percent = np.random.uniform(minp, maxp, size=1)
            for i, s in enumerate(mat[..., 0]):
                _, bins = self.downsample_strata(i, s, downsample_percent)
                new_x[i] = bins
            return new_x
        else:
            return mat[..., 0]

    def get_compressed_band_cell(self, cell, preprocessing=None):
        if cell in self.sparse_matrices.keys():
            compressed_sparse_matrix = self.sparse_matrices[cell]
        else:
            anchor_to_anchor = self.get_cell_pixels(cell)
            if len(anchor_to_anchor) == 0:  # cell has no reads
                return None
            anchor_to_anchor = anchor_to_anchor[anchor_to_anchor['a1'].isin(
                self.anchor_list['anchor'])]
            anchor_to_anchor = anchor_to_anchor[anchor_to_anchor['a2'].isin(
                self.anchor_list['anchor'])]
            rows = np.vectorize(anchor_to_locus(self.anchor_dict))(
                anchor_to_anchor['a1'].values)  # convert anchor names to row indices
            cols = np.vectorize(anchor_to_locus(self.anchor_dict))(
                anchor_to_anchor['a2'].values)  # convert anchor names to column indices

            if self.use_raw_data:
                matrix = csr_matrix((anchor_to_anchor['obs'], (rows, cols)),
                                    shape=(len(self.anchor_list), len(self.anchor_list)))
            else:
                anchor_to_anchor['ratio'] = (
                    anchor_to_anchor['obs'] + 5) / (anchor_to_anchor['exp'] + 5)
                matrix = csr_matrix((anchor_to_anchor['ratio'], (rows, cols)),
                                    shape=(len(self.anchor_list), len(self.anchor_list)))
            if preprocessing is not None:
                tmp_mat = matrix.A
                for op in preprocessing:
                    if op.lower() == 'convolution':
                        tmp_mat = convolution(tmp_mat)
                    elif op.lower() == 'random_walk':
                        tmp_mat = random_walk(tmp_mat)
                    elif op.lower() == 'vc_sqrt_norm':
                        tmp_mat = VC_SQRT_norm(tmp_mat)
                    elif op.lower() == 'google':
                        tmp_mat = graph_google(tmp_mat)
                matrix = csr_matrix(tmp_mat)

            compressed_matrix = np.zeros(
                (self.limit2Mb, self.matrix_len + self.matrix_pad))
            for i in range(self.limit2Mb):
                diagonal = matrix.diagonal(k=i + self.rotated_offset)
                compressed_matrix[i, i:i + len(diagonal)] = diagonal

            compressed_sparse_matrix = csr_matrix(compressed_matrix)
            self.sparse_matrices[cell] = compressed_sparse_matrix
        return compressed_sparse_matrix

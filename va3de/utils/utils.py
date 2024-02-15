import re
import os 
import numpy as np

from tqdm import tqdm


chromosome_labels = {'chr1': 0, 'chr2': 1, 'chr3': 2, 'chr4': 3, 'chr5': 4, 'chr6': 5, 'chr7': 6, 'chr8': 7, 'chr9': 8,
                     'chr10': 9, 'chr11': 10, 'chr12': 11, 'chr13': 12, 'chr14': 13, 'chr15': 14, 'chr16': 15, 'chr17': 16, 'chr18': 17,
                     'chr19': 18, 'chr20': 19, 'chr21': 20, 'chr22': 21, 'chrX': 22, 'chrY': 23}

def resolution_to_name(res):
    if res >= 1e6:
        return f"{int(res / 1e6)}M"
    else:
        return f"{int(res / 1e3)}kb"


def res_name_to_int(res_name):
    resolution = 0
    if 'M' in res_name:
        resolution = int(res_name[:-1]) * 1000000
    elif 'mb' in res_name.lower():
        resolution = int(res_name[:-2]) * 1000000
    elif 'kb' in res_name:
        resolution = int(res_name[:-2]) * 1000 
    return resolution


#def name_to_params(name):


def matrix_to_interaction(inputfolder, mapbed, resolution, outputdir):
	print('Converting to locus pair interaction list...')
	mapbedfh = open(mapbed)

	bin_strs = False  # need to include 'bin' in token id
	anchor_strs = False
	if 'synthetic' in inputfolder or 'islet' in inputfolder or 'simulated' in inputfolder or 'hippocampus' in inputfolder or 'human_brain':
		bin_strs = True
	if 'pfc' in inputfolder or 'kim' in inputfolder or 'ramani' in inputfolder or 'li2019' in inputfolder:
		anchor_strs = True
	os.makedirs(outputdir, exist_ok=True)

	bintocoord = {}
	chr_offset = 0
	prev_bin = 0
	chr_name = None
	for line in mapbedfh :
		tokens = line.split()
		if bin_strs:
			bin = int(tokens[3].replace('bin', '').replace('_', '')) - 1

			if chr_name is None:
				chr_name = tokens[0]
			# elif chr_name != tokens[0]:
			# 	chr_name = tokens[0]
			# 	chr_offset += prev_bin
			bintocoord[bin] = (tokens[0],str(int(tokens[1])+resolution/2))
			prev_bin = bin
		elif anchor_strs:
			bin = int(tokens[3].replace('A_', ''))
			bintocoord[bin] = (tokens[0],str(int(int(tokens[1]) + resolution/2)))
		else:
			bintocoord[int(tokens[3])] = (tokens[0],str(int(tokens[1])+resolution/2))


	list = os.popen('ls '+ inputfolder + '/*matrix').read()
	filenames = list.split('\n')[:-1]

	for matrixfilename in tqdm(filenames) :
		if matrixfilename != '' :
			matrixfilefh = open(matrixfilename)
			outfilefh = open(os.path.join(outputdir, os.path.split(matrixfilename)[1][:-7]+'.int.bed'),'w')
			#print(matrixfilename)
			for line in matrixfilefh :
				tokens = line.split()
				bin1_idx = int(tokens[0])
				bin2_idx = int(tokens[1])
				
				#print(bintocoord.keys())
				#print(tokens[0], tokens[1])
				try:
					firstcoor = bintocoord[bin1_idx]
					secondcoor = bintocoord[bin2_idx]
					outtokens = [firstcoor[0], str(int(firstcoor[1].split('.')[0])), secondcoor[0], str(int(secondcoor[1].split('.')[0])), str(int(tokens[2].split('.')[0])), tokens[3]]
					outtokens = [str(t) for t in outtokens]
					outline = '\t'.join(outtokens)
					print(outline, file=outfilefh)
				except KeyError as e:
					pass


			outfilefh.close()



def anchor_list_to_dict(anchors):
    """
    Converts the array of anchor names to a dictionary mapping each anchor to its chromosomal index

    Args:
        anchors (:obj:`numpy.array`) : array of anchor name values

    Returns:
        `dict` : dictionary mapping each anchor to its index from the array
    """
    anchor_dict = {}
    for i, anchor in enumerate(anchors):
        anchor_dict[anchor] = i
    return anchor_dict


def anchor_to_locus(anchor_dict):
    """
    Function to convert an anchor name to its genomic locus which can be easily vectorized

    Args:
        anchor_dict (:obj:`dict`) : dictionary mapping each anchor to its chromosomal index

    Returns:
        `function` : function which returns the locus of an anchor name
    """
    def f(anchor):
        return anchor_dict[anchor]
    return f


def sorted_nicely(l):
    """
    Sorts an iterable object according to file system defaults
    Args:
        l (:obj:`iterable`) : iterable object containing items which can be interpreted as text

    Returns:
        `iterable` : sorted iterable
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def draw_heatmap(matrix, color_scale, ax=None, return_image=False):
    """
    Display ratio heatmap containing only strong signals (values > 1 or 0.98th quantile)

    Args:
        matrix (:obj:`numpy.array`) : ratio matrix to be displayed
        color_scale (:obj:`int`) : max ratio value to be considered strongest by color mapping
        ax (:obj:`matplotlib.axes.Axes`) : axes which will contain the heatmap.  If None, new axes are created
        return_image (:obj:`bool`) : set to True to return the image obtained from drawing the heatmap with the generated color map

    Returns:
        ``numpy.array`` : if ``return_image`` is set to True, return the heatmap as an array
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors
    import matplotlib.cm
    if color_scale != 0:
        breaks = np.append(np.arange(1.001, color_scale, (color_scale - 1.001) / 18), np.max(matrix))
    elif np.max(matrix) < 2:
        breaks = np.arange(1.001, np.max(matrix), (np.max(matrix) - 1.001) / 19)
    else:
        step = (np.quantile(matrix, q=0.98) - 1) / 18
        up = np.quantile(matrix, q=0.98) + 0.011
        if up < 2:
            up = 2
            step = 0.999 / 18
        breaks = np.append(np.arange(1.001, up, step), np.max(matrix))

    n_bin = 20  # Discretizes the interpolation into bins
    colors = ["#FFFFFF", "#FFE4E4", "#FFD7D7", "#FFC9C9", "#FFBCBC", "#FFAEAE", "#FFA1A1", "#FF9494", "#FF8686",
              "#FF7979", "#FF6B6B", "#FF5E5E", "#FF5151", "#FF4343", "#FF3636", "#FF2828", "#FF1B1B", "#FF0D0D",
              "#FF0000"]
    cmap_name = 'my_list'
    # Create the colormap
    cm = matplotlib.colors.LinearSegmentedColormap.from_list(
        cmap_name, colors, N=n_bin)
    norm = matplotlib.colors.BoundaryNorm(breaks, 20)
    # Fewer bins will result in "coarser" colomap interpolation
    if ax is None:
        _, ax = plt.subplots()
    img = ax.imshow(matrix, cmap=cm, norm=norm, interpolation='nearest')
    if return_image:
        plt.close()
        return img.get_array()


def draw_heatmap_quantile(matrix, ax=None, return_image=False):
    n_bin = 20  # Discretizes the interpolation into bins
    breaks = np.linspace(np.quantile(matrix, q=0.9), np.max(matrix), n_bin)
    colors = ["#FFFFFF", "#FFE4E4", "#FFD7D7", "#FFC9C9", "#FFBCBC", "#FFAEAE", "#FFA1A1", "#FF9494", "#FF8686",
              "#FF7979", "#FF6B6B", "#FF5E5E", "#FF5151", "#FF4343", "#FF3636", "#FF2828", "#FF1B1B", "#FF0D0D",
              "#FF0000"]
    cmap_name = 'my_list'
    # Create the colormap
    cm = matplotlib.colors.LinearSegmentedColormap.from_list(
        cmap_name, colors, N=n_bin)
    norm = matplotlib.colors.BoundaryNorm(breaks, n_bin)
    # Fewer bins will result in "coarser" colomap interpolation
    if ax is None:
        _, ax = plt.subplots()
    img = ax.imshow(matrix, cmap=cm, norm=norm, interpolation='nearest')
    if return_image:
        plt.close()
        return img.get_array()


def get_heatmap(matrix, color_scale):
    if color_scale != 0:
        breaks = np.append(np.arange(1.001, color_scale, (color_scale - 1.001) / 18), np.max(matrix))
    elif np.max(matrix) < 2:
        breaks = np.arange(1.001, np.max(matrix), (np.max(matrix) - 1.001) / 19)
    else:
        step = (np.quantile(matrix, q=0.98) - 1) / 18
        up = np.quantile(matrix, q=0.98) + 0.011
        if up < 2:
            up = 2
            step = 0.999 / 18
        breaks = np.append(np.arange(1.001, up, step), np.max(matrix))

    n_bin = 20  # Discretizes the interpolation into bins
    colors = ["#FFFFFF", "#FFE4E4", "#FFD7D7", "#FFC9C9", "#FFBCBC", "#FFAEAE", "#FFA1A1", "#FF9494", "#FF8686",
              "#FF7979", "#FF6B6B", "#FF5E5E", "#FF5151", "#FF4343", "#FF3636", "#FF2828", "#FF1B1B", "#FF0D0D",
              "#FF0000"]
    cmap_name = 'my_list'
    # Create the colormap
    cm = matplotlib.colors.LinearSegmentedColormap.from_list(
        cmap_name, colors, N=n_bin)
    norm = matplotlib.colors.BoundaryNorm(breaks, 20)
    # Fewer bins will result in "coarser" colomap interpolation
    m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cm)
    heatmap = m.to_rgba(matrix)
    mask = matrix > 1.2
    heatmap[..., -1] = np.ones_like(mask) * mask
    return heatmap
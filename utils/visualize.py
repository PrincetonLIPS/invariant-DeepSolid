import sys, os, pickle

sys.path.append( os.getcwd() + '/submodules/DeepSolid')
sys.path.append( os.getcwd() + '/submodules/space-groups')

import pandas as pd
import matplotlib.pyplot as plt
from space_groups.group import SymmetryGroup
from utils.loader import load_module, load_cuda_manually
from matplotlib import colors


import pyvista as pv
import numpy as np
pv.start_xvfb()

pv.global_theme.font.family = 'times'
pv.global_theme.font.label_size = 10
pv.global_theme.font.title_size = 40

DEFAULT_COLORS_10 = plt.get_cmap('tab10').colors
DEFAULT_COLORS_20 = plt.get_cmap('tab20').colors
ATOM_SIZE_DICT = {
    'H': 3,
    'Li': 5,
    'C': 6,
}
LARGEST_ATOM_SIZE = 50.0
ELECTRON_SIZE = 3.5
ELECTRON_COLOR = DEFAULT_COLORS_10[0]
ATOM_COLOR_LIST = [DEFAULT_COLORS_10[3], DEFAULT_COLORS_10[1]]
OBJ_COLOR = DEFAULT_COLORS_10[4]

class Object(object):
    pass

class SpaceGroupVisualizer():
    '''
        Visualizer class for pyvista plots. Native methods of this class do not require gpu
    '''
    def __init__(
            self,
            base_group: SymmetryGroup = None,
            cell = None,
            pyvista_cfg = None,
            base_cfg_path: str = None,
            cfg_path: str = None,
            libcu_lib_path: str = None,
        ):
        #  fix to accommodate old version of jax in DeepSolid
        if libcu_lib_path is not None:
            load_cuda_manually(libcu_lib_path)
        # either group and cell need to be supplied, or base_cfg_path and cfg_path need to be supplied
        if (base_group is not None) and (cell is not None) and (pyvista_cfg is not None):
            self.base_group = base_group
            self.cell = cell
            self.pyvista_cfg = pyvista_cfg
        elif (base_cfg_path is not None) and (cfg_path is not None):
            base_cfg_module = load_module('base_cfg', base_cfg_path)
            base_cfg = base_cfg_module.default()
            cfg_module = load_module('cfg', cfg_path)
            cfg = cfg_module.get_config(base_cfg)
            
            self.base_group = cfg.symmetria.gpave.group

            self.cell = cfg.system.pyscf_cell
            self.pyvista_cfg = cfg.pyvista
        else:
            raise ValueError('Either group, cell and pyvista_cfg need to be supplied, or base_cfg_path and cfg_path need to be supplied.')

    def pv_plotter_setup(
            self,
            subplot: bool = False,
            subplot_shape = None,
            plotter = None, 
            interactive: bool = False,
        ):
        if plotter is None:
            if subplot:
                plotter = pv.Plotter(shape=subplot_shape, border=False)
            else:
                plotter = pv.Plotter()
        if interactive:
            pv.set_jupyter_backend('html') 
        else:
            pv.set_jupyter_backend('static')
        return plotter

    def pv_plotter_wrapup(
            self,
            plotter, 
            title: str = None,
            show: bool = True,
            savepath: str = None,
            show_grid: bool = True,
        ):
        if title is not None:
            plotter.add_title(title)
        if show_grid is True:
            plotter.show_grid(
                xtitle='',
                ytitle='',
                ztitle='',
                font_size=20,
            )
        plotter.set_position(self.pyvista_cfg.camera['position'])
        plotter.set_viewup(self.pyvista_cfg.camera['viewup'])
        if 'extra_lines' in self.pyvista_cfg.camera:
            for line in self.pyvista_cfg.camera['extra_lines']:
                line_mesh = pv.Line(line[0], line[1])
                plotter.add_mesh(line_mesh, color='black')
        if 'zoom' in self.pyvista_cfg.camera:
            plotter.camera.zoom(self.pyvista_cfg.camera['zoom'])
        if show:
            plotter.show()
        if savepath is not None:
            plotter.save_graphic(savepath)
        return plotter

    def get_cell_grid(self, btmrgt_corner, cell):
        cell_indices = np.array([8] + list(range(8)))
        cell_type = np.array(pv.CellType.HEXAHEDRON)
        if self.base_group.dims == 2:
            z_vec = np.zeros(3)
        else:
            z_vec = cell[2]
        cell_points = np.array([
            btmrgt_corner,
            btmrgt_corner + cell[0],
            btmrgt_corner + cell[0] + cell[1],
            btmrgt_corner + cell[1],
            btmrgt_corner + z_vec,
            btmrgt_corner + cell[0] + z_vec,
            btmrgt_corner + cell[0] + cell[1] + z_vec,
            btmrgt_corner + cell[1] + z_vec,
        ])
        return pv.UnstructuredGrid(cell_indices, cell_type, cell_points)

    def plot_supercell(
            self,
            plotter = None,
            interactive: bool = False,
            title: str = None,
            show: bool = True,
            savepath: str = None,
            show_grid: bool = True,
        ):
        '''
            Args:
            - plotter: pyvista.Plotter() object. Generate new Plotter() if not supplied
            - interactive: bool, set jupyter backend to 'html' if True and 'static' if False
            - title: str, plot title
            - show: bool, whether to show the plot
            - savepath: str, if not None, call plotter.save_graphic(savepath)

            Return:
            - plotter: plotter object. 
            
            Call plotter.show() to show in jupyter notebook environment.
            Call plotter.save_graphic() to save pdf file.
        '''
        plotter = self.pv_plotter_setup(
                    plotter=plotter, 
                    interactive=interactive,
                )
        # plot supercell
        supcell = self.get_cell_grid( np.zeros(3), self.cell.a )
        plotter.add_mesh(supcell, show_edges=True, opacity=0.1)

        # plot unit cells
        S = self.base_group.S
        unit_cell = self.base_group.unit_cell
        for indices in np.ndindex(int(S[0]), int(S[1]), int(S[2])):
            cell = self.get_cell_grid( indices @ unit_cell, unit_cell )
            plotter.add_mesh(cell, show_edges=True, opacity=0.1)

        plotter = self.pv_plotter_wrapup(
                    plotter=plotter, 
                    title=title,
                    show=show,
                    savepath=savepath,
                    show_grid=show_grid
                )
        return plotter

    def plot_supercell_atom(
            self,
            plotter = None,
            interactive: bool = False,
            title: str = None,
            atom_colors: list = None,
            show: bool = True,
            savepath: str = None,
            show_grid: bool = True,
        ):
        '''
            Args:
            - plotter: pyvista.Plotter() object. Generate new Plotter() if not supplied
            - interactive: bool, set jupyter backend to 'html' if True and 'static' if False
            - title: str, plot title
            - atom_colors: list of values to be supplied as colors to pyvista for atoms
            - show: bool, whether to show the plot
            - savepath: str, if not None, call plotter.save_graphic(savepath)

            Return:
            - plotter: plotter object. 
            
            Call plotter.show() to show in jupyter notebook environment.
            Call plotter.save_graphic() to save pdf file.
        '''
        plotter = self.plot_supercell(
                        plotter = plotter,
                        interactive = interactive,
                        show=False,
                        show_grid=show_grid
                    )
        # obtain configs for atoms of different types
        all_atoms = self.cell.atom + self.pyvista_cfg.extra_atoms
        atom_names = list(set([atom[0] for atom in all_atoms]))
        atom_sizes = np.array([ATOM_SIZE_DICT[name] for name in atom_names])
        atom_sizes = atom_sizes / np.max(atom_sizes) * LARGEST_ATOM_SIZE
        if atom_colors is None:
            atom_colors = ATOM_COLOR_LIST[:len(atom_names)]
        else:
            atom_colors = atom_colors[:len(atom_names)] 
        
        # loop through atoms. Extra atoms for visualization are more transparent
        for atoms, opacity in [(self.cell.atom, 0.7), (self.pyvista_cfg.extra_atoms, 0.2)]:
            atom_coords_list = [np.array([atom[1] for atom in atoms if atom[0] == name]) for name in atom_names]

            for atom_coords, atom_name, atom_size, color in zip(atom_coords_list, atom_names, atom_sizes, atom_colors):
                if len(atom_coords) > 0:
                    atom_points = pv.PolyData(atom_coords)
                    plotter.add_mesh(atom_points, 
                                    color=color, 
                                    opacity=opacity, 
                                    point_size=atom_size, 
                                    render_points_as_spheres=True, 
                    )

        plotter = self.pv_plotter_wrapup(
                    plotter=plotter, 
                    title=title,
                    show=show,
                    savepath=savepath,
                    show_grid=show_grid
                )
        return plotter

    def plot_supercell_atom_symmetry( 
            self,
            mode: str,
            group: SymmetryGroup = None,
            plotter = None,
            interactive: bool = False,
            title: str = None,
            atom_colors: list = None,
            obj_color = None,
            show: bool = True,
            savepath: str = None,
        ):
        '''
            Args:
            - mode: 'gpave', 'orbifold', 'orbifold-before', 'orbifold-after'
            - group: SymmetryGroup to be used. If None is supplied, self.base_group is used
            - plotter: pyvista.Plotter() object. Generate new Plotter() if not supplied
            - interactive: bool, set jupyter backend to 'html' if True and 'static' if False
            - title: str, plot title
            - savepath: str, save file if supplied 
            - atom_colors: list of values to be supplied as colors to pyvista for atoms
            - obj_color: value to be supplied as a color to pyvista for the rendered object
            - show: bool, whether to show the plot
            - savepath: str, if not None, call plotter.save_graphic(savepath)

            Return:
            - plotter: plotter object. 
            
            Call plotter.show() to show in jupyter notebook environment.
            Call plotter.save_graphic() to save pdf file.
        '''
        assert mode in ['orbifold', 'gpave', 'orbifold-before', 'orbifold-after']
        from utils.gpave import make_group_ops, make_orbifold_ops
        import jax
        import jax.numpy as jnp

        if group is None:
            group = self.base_group 
        
        # only show normals of asus if mode is orbifold
        if mode == 'orbifold':
            plotter = self.plot_supercell_group_asu(
                            plotter=plotter,
                            interactive=interactive,
                            atom_colors=atom_colors,
                            show_normals=(mode in ['orbifold', 'orbifold-before', 'orbifold-after']),
                            show=False
                        )
        else:
            plotter = self.plot_supercell_atom(
                            plotter=plotter,
                            interactive=interactive,
                            atom_colors=atom_colors,
                            show=False
                        )


        if obj_color is None:
            obj_color = OBJ_COLOR
        obj = pv.read(self.pyvista_cfg.symmetria.obj_path)
        
        # adjust object to fit the scale according to the supercell 
        obj.rotate_vector(
                self.pyvista_cfg.symmetria.rotate_vec, 
                angle=self.pyvista_cfg.symmetria.rotate_angle, 
                inplace=True
        )
        obj.scale(
            self.pyvista_cfg.symmetria.scale, 
            inplace=True
        )
        obj.translate( 
            - obj.points.min(axis=0) + self.pyvista_cfg.symmetria.translate, 
            inplace=True
        )

        # perform group operations
        subsample = Object()
        subsample.on = False
        if mode == 'gpave' or mode == 'orbifold-before':
            op_objs = make_group_ops(group, subsample=subsample)(obj.points.flatten(), 0)
        else:
            asu_objs = make_group_ops(group, subsample=subsample)(obj.points.flatten(), 0)
            op_objs = jax.vmap(
                        lambda obj: make_orbifold_ops(group, proj_index=0)(obj.flatten(), 0),
            )(asu_objs)
            if mode == 'orbifold':
                # show both before and after
                op_objs = jnp.vstack((asu_objs, op_objs))

        for op_obj in op_objs:
            op_mesh = pv.PolyData(np.array(op_obj), faces=obj.faces)

            plotter.add_mesh(op_mesh, color=obj_color, show_edges=True, opacity=.2)

        plotter = self.pv_plotter_wrapup(
                    plotter=plotter, 
                    title=title,
                    show=show,
                    savepath=savepath,
                )
        return plotter

    def plot_supercell_group_asu(
            self,
            group: SymmetryGroup = None,
            plotter = None,
            interactive: bool = False,
            title: str = None,
            atom_colors: list = None,
            show_normals: bool = True,
            show: bool = True,
            savepath: str = None,
            show_grid: bool = True,
        ):
        '''
            Args:
            - group: SymmetryGroup to be used. If None is supplied, self.base_group is used
            - plotter: pyvista.Plotter() object. Generate new Plotter() if not supplied
            - interactive: bool, set jupyter backend to 'html' if True and 'static' if False
            - title: str, plot title
            - savepath: str, save file if supplied 
            - atom_colors: list of values to be supplied as colors to pyvista for atoms
            - show_normals: bool, whether to show the normals of each asu
            - show: bool, whether to show the plot
            - savepath: str, if not None, call plotter.save_graphic(savepath)

            Return:
            - plotter: plotter object. 
            
            Call plotter.show() to show in jupyter notebook environment.
            Call plotter.save_graphic() to save pdf file.
        '''

        if group is None:
            group = self.base_group 

        plotter = self.plot_supercell_atom(
                        plotter=plotter,
                        interactive=interactive,
                        atom_colors=atom_colors,
                        show=False,
                        show_grid=show_grid
                    )
        
        for asu in group.base_asu_list:
            # add asu
            max_index = np.max([np.max(a) for a in asu['face_list']])
            cell_indices = np.array([max_index+1] + list(range(max_index+1)))
            cell_type = np.array(pv.CellType.TETRA)
            
            z_factor = np.ones(asu['vertices'].shape)
            if group.dims == 2:
                z_factor[:,2] = 0

            cell_points = asu['vertices'] * z_factor
            
            asu_grid = pv.UnstructuredGrid(cell_indices, cell_type, cell_points)
            asu_edges = asu_grid.extract_all_edges()
            plotter.add_mesh(asu_grid, show_edges=True, opacity=0.1)
            plotter.add_mesh(asu_edges, line_width=1, color='k')
            
            if show_normals:
                # add asu center
                if group.dims == 3:
                    center = asu['center']
                else:
                    center = np.concatenate([asu['center'][:2],[0.]])
                plotter.add_mesh(pv.PolyData([center]), 
                                color=DEFAULT_COLORS_10[5], 
                                opacity=1, 
                                point_size=0.5 * LARGEST_ATOM_SIZE, 
                                render_points_as_spheres=True, 
                )

                # add asu normal lines
                for asu_normal in asu['normal_list']:
                    if group.dims == 3:
                        normal = asu_normal
                    else:
                        normal = np.concatenate([asu_normal[:2],[0.]])
                    plotter.add_lines(
                        np.array([center, center + normal]),
                        color=DEFAULT_COLORS_10[5], 
                        width=5,
                    )

    
        plotter = self.pv_plotter_wrapup(
                    plotter=plotter, 
                    title=title,
                    show=show,
                    savepath=savepath,
                    show_grid=show_grid
                )
        
        return plotter

    def plot_supercell_symmscan_cfg(
            self,
            symscan_fname,
            plotter = None,
            interactive: bool = False,
            title: str = None,
            atom_colors: list = None,
            show_asu: bool = False,
            show: bool = True,
            savepath: str = None,
            show_grid: bool = True,
        ):
        '''
            Args:
            - symscan_fname: name of config file in symmscan_config folder
            - plotter: pyvista.Plotter() object. Generate new Plotter() if not supplied
            - interactive: bool, set jupyter backend to 'html' if True and 'static' if False
            - title: str, plot title
            - atom_colors: list of values to be supplied as colors to pyvista for atoms
            - show_asu: bool, whether to show asus
            - show: bool, whether to show the plot
            - savepath: str, if not None, call plotter.save_graphic(savepath)

            Return:
            - plotter: plotter object. 
            
            Call plotter.show() to show in jupyter notebook environment.
            Call plotter.save_graphic() to save pdf file.
        '''
        if show_asu:
            plotter = self.plot_supercell_group_asu(
                            plotter=plotter,
                            interactive=interactive,
                            atom_colors=atom_colors,
                            show_normals=False,
                            show=False,
                            show_grid=show_grid
                        )
        else:
            plotter = self.plot_supercell_atom(
                            plotter=plotter,
                            interactive=interactive,
                            atom_colors=atom_colors,
                            show=False,
                            show_grid=show_grid
                        )
        
        # load symm cfg
        symm_module = load_module('symmscan_cfg', f'symmscan_config/{symscan_fname}')
        elecs, symm_cam = symm_module.symm_cfg(L_Bohr=self.cell.L_Bohr)
        elec_points = pv.PolyData(elecs)
        
        # determine electron size relative to the largest atom size
        all_atoms = self.cell.atom + self.pyvista_cfg.extra_atoms
        atom_names = list(set([atom[0] for atom in all_atoms]))
        atom_sizes = np.array([ATOM_SIZE_DICT[name] for name in atom_names])
        electron_size = ELECTRON_SIZE / np.max(atom_sizes) * LARGEST_ATOM_SIZE

        plotter.add_mesh(elec_points, 
                         color=ELECTRON_COLOR, 
                         point_size=electron_size, 
                         render_points_as_spheres=True, 
        )
        
        # use camera parameters from symmscan_cfg
        camera_cache = self.pyvista_cfg.camera
        self.pyvista_cfg.camera = symm_cam
        plotter = self.pv_plotter_wrapup(
            plotter=plotter, 
            title=title,
            show=show,
            savepath=savepath,
            show_grid=show_grid
            )
        self.pyvista_cfg.camera = camera_cache 

        return plotter

class DeepSolidVisualizer():
    '''
        Visualizer for DeepSolid wavefunctions. Appropriate jax dependencies are set up upon initialization.
    '''
    def __init__(
        self,
        log_dir_list: list,
        color_list: list = [],
        label_list: list = [],
        libcu_lib_path: str = None,
    ):
        self.replace_log_dir_list(log_dir_list, color_list, label_list)
        self.networks = None
        self.symmscan = None
        #  fix to accommodate old version of jax in DeepSolid
        if libcu_lib_path is not None:
            load_cuda_manually(libcu_lib_path)
        self.libcu_lib_path = libcu_lib_path

    def replace_log_dir_list(
        self,
        log_dir_list: list,
        color_list: list = [],
        label_list: list = [],
    ):
        '''
            If no color_list supplied, use default colors if len(log_dir_list) <= 10.
            If no label_list supplied, use log_dir_list.
        '''
        self.metadata_list = []
        for log_dir in log_dir_list:
            if not os.path.isdir(log_dir):
                raise ValueError(f'Invalid directory {log_dir}.')
            with open(log_dir + '__metadata.pk', 'rb+') as f:
                metadata = pickle.load(f)
            self.metadata_list.append(metadata)   
        
        if len(color_list) > 0 and len(color_list) != len(log_dir_list):
            raise ValueError(f'Supplying {len(color_list)} colors for {len(log_dir_list)} directories.')
        if len(color_list) == 0: 
            if len(log_dir_list) > 20:
                raise ValueError(f'Need to supply custom colors for > 10 directories. {len(log_dir_list)} directories found.')
            elif len(log_dir_list) > 10:
                color_list = DEFAULT_COLORS_20[:len(log_dir_list)]
            else:
                color_list = DEFAULT_COLORS_10[:len(log_dir_list)]
        
        if len(label_list) > 0 and len(label_list) != len(log_dir_list):
            raise ValueError(f'Supplying {len(label_list)} labels for {len(log_dir_list)} directories.')
        if len(label_list) == 0: 
            label_list = log_dir_list
        
        self.log_dir_list = log_dir_list
        self.color_list = color_list
        self.label_list = label_list
    
    def load_train_stats(self):
        '''
            Load train_stats.csv from self.log_dir_list
        '''
        # check if train_stats.csv exists
        for log_dir in self.log_dir_list:
            fpath = log_dir + 'train_stats.csv'
            if not os.path.exists(fpath):
                raise ValueError(f'{fpath} does not exist.')
        # load dataframes
        self.train_stats_list = []
        for log_dir in self.log_dir_list:
            self.train_stats_list.append(
                pd.read_csv(log_dir + 'train_stats.csv')
            )

    def plot_train_stats(
            self,
            field,
            indices = None,
            t_range = (0, None),
            figsize = (10, 6),
            xlim = None,
            ylim = None,
            title = None,
            savepath = None,
            ma_window: int = None,
        ):
        '''
            Plot data loaded in self.train_stats_list.

            If ma_window is not None, moving average is used for plotting the stat
        '''
        plt.figure(figsize=figsize)
        if indices is None:
            indices = list(range(0, len(self.train_stats_list)))

        for i, (stats, color, label) in enumerate(zip(self.train_stats_list, self.color_list, self.label_list)):
            if i not in indices:
                continue
            if t_range[0] == None and t_range[1] == None:
                data = stats[field]
            elif t_range[0] == None:
                data = stats[stats['step'] < t_range[1]][['step',field]]
            elif t_range[1] == None:
                data = stats[stats['step'] >= t_range[0]][['step',field]]
            else:
                data = stats[stats['step'].isin(range(*t_range))][['step',field]]
            
            if ma_window is None:
                plt.plot(
                    data['step'], data[field], color=color, label=label
                )
            else:
                ts = pd.Series(data[field], index=data['step']).rolling(window=ma_window)
                rolling_mean = ts.mean()[ma_window-1:]
                rolling_std = ts.std(ddof=0)[ma_window-1:] / np.sqrt(ma_window)
                rolling_index = np.array(rolling_mean.index)
                plt.plot(rolling_index, rolling_mean, color=color, label=label)
                
                fill_color = np.array(colors.to_rgba(color))
                fill_color[3] *= 0.3
                plt.fill_between(rolling_index,
                                rolling_mean - rolling_std,
                                rolling_mean + rolling_std, 
                                facecolor=fill_color)
            
        if title is not None:
            plt.title(title)
        plt.tight_layout()
        plt.legend()
        plt.xlim(xlim)
        plt.ylim(ylim)
        
        fig = plt.gcf()
        plt.show()
        if savepath is not None:
            fig.savefig(savepath)

    def load_space_visualizers(self):
        '''
            Load self.space_visualizers
        '''
        self.space_visualizers = []
        for log_dir, metadata in zip(self.log_dir_list, self.metadata_list):
            base_cfg_str = metadata['base_cfg_str']
            cfg_str = metadata['cfg_str']            
            self.space_visualizers.append(
                SpaceGroupVisualizer(
                    base_cfg_path=log_dir + base_cfg_str,
                    cfg_path=log_dir + cfg_str
                )
            )
    
    def plot_supercell_atom(
            self,
            interactive: bool = False,
            show_list: list = None,
            title_list: list = None
        ):
        '''
            Visualize supercell with atoms.
            self.load_space_visualizers must be run before this

            Args:
            - interactive: bool, whether the plot should be interactive
            - show_list: list of bools to specify which plots to show. If not specified, default to all True
            - title_list: list of strings used as titles. If not specified, use self.label_list
        '''
        if title_list is None:
            title_list = ['Supercell ' + label for label in self.label_list]
        elif len(title_list) != len(self.metadata_list):
            raise ValueError(str(len(self.metadata_list)) + ' titles expected but ' +
                             str(len(title_list)) + ' titles received. ')
        if show_list is None:
            show_list = [True] * len(self.metadata_list)
        elif len(show_list) != len(self.metadata_list):
            raise ValueError('Length-' + str(len(self.metadata_list)) + ' show_list expected but ' +
                             'length-' + str(len(title_list)) + ' show_list received. ')

        plotter_list = []
        for symm_visualizer, log_dir, metadata, title, show \
                in zip(self.space_visualizers, self.log_dir_list, self.metadata_list, 
                        title_list, show_list):    
            
            plotter = symm_visualizer.plot_supercell_atom(
                interactive=interactive,
                title=title,
                show=show,
                savepath= log_dir + metadata['cfg_str'].split('.py')[0] + '_supercell.pdf',
            )
            plotter_list.append(plotter)
        return plotter_list
                
    def plot_supercell_atom_symmetry(
            self,
            group_name: str = 'average',
            interactive: bool = False,
            show_list: list = None,
            title_list: list = None
        ):
        '''
            Visualize symmetries in supercell
            self.load_space_visualizers must be run before this

            Args:
            - group_name: name of SymmetryGroup to be used. Either 'average' or 'augment' or 'measure'
            - interactive: bool, whether the plot should be interactive
            - show_list: list of bools to specify which plots to show. If not specified, default to all True
            - title_list: list of strings used as titles. If not specified, use self.label_list
        '''
        if title_list is None:
            title_list = ['Symmetry ' + label for label in self.label_list]
        elif len(title_list) != len(self.metadata_list):
            raise ValueError(str(len(self.metadata_list)) + ' titles expected but ' +
                             str(len(title_list)) + ' titles received. ')
        if show_list is None:
            show_list = [True] * len(self.metadata_list)
        elif len(show_list) != len(self.metadata_list):
            raise ValueError('Length-' + str(len(self.metadata_list)) + ' show_list expected but ' +
                             'length-' + str(len(title_list)) + ' show_list received. ')

        group_plotter_list = []
        for symm_visualizer, log_dir, metadata, title, show \
                in zip(self.space_visualizers, self.log_dir_list, self.metadata_list, 
                        title_list, show_list):    
            
            group_plotter = symm_visualizer.plot_supercell_atom_symmetry(
                group_name=group_name,
                interactive=interactive,
                title=title,
                show=show,
                savepath=log_dir + metadata['cfg_str'].split('.py')[0] + '_' + group_name + '_symmetry.pdf',
            )
            group_plotter_list.append(group_plotter)
        return group_plotter_list

    def load_samples(
            self,
            samplefname_list: list):
        if len(samplefname_list) != len(self.metadata_list):
            raise ValueError(str(len(self.metadata_list)) + ' sample files expected but ' +
                             str(len(samplefname_list)) + ' sample files received. ')
        
        self.samples_list = []
        for log_dir, fname in zip(self.log_dir_list, samplefname_list): 
            if os.path.isfile(log_dir + fname):
                with open(log_dir + fname, 'rb') as file:
                    samples = pickle.load(file) 
                self.samples_list.append(samples)
            else:
                raise ValueError(log_dir + fname + ' not found.')
            
    def plot_samples_marginals(
            self,
            num_elec_to_view: int,
            color = None,
            opacity: float = 0.1,
            point_size: float = 2.0,
            constraint_fn = None,
            interactive: bool = False,
            show_list: list = None,
            savepath_list: list = None,
        ):
        '''
            Visualize marginal density of num_elec_to_view electrons
            self.load_space_visualizers and self.load_samples must be run before this

            Args:
            - interactive: bool, whether the plot should be interactive
            - num_elec_to_view: number of electrons to visualize at the same time
            - color: electron color. If none, set to ELECTRON_COLOR
            - opacity: float
            - point_size: float
            - constraint_fn: function that takes in a numpy array of shape [UNKNOWN, num_elec_to_view, 3] and returns a boolean array of shape [UNKNOWN], for constraining the sets on which the marginal density is visualized
            - show_list: list of bools to specify which plots to show. If not specified, default to all True
            - savepath_list: list of strings used as savepaths. If not specified, use log_dir + metadata['cfg_str'].split('.py')[0] + '_' + str(num_elec_to_view) + '-marginals.pdf'
        '''
        if show_list is None:
            show_list = [True] * len(self.metadata_list)
        elif len(show_list) != len(self.metadata_list):
            raise ValueError('Length-' + str(len(self.metadata_list)) + ' show_list expected but ' +
                             'length-' + str(len(show_list)) + ' show_list received. ')

        if savepath_list is None:
            savepath_list = [log_dir + metadata['cfg_str'].split('.py')[0] + '_' + str(num_elec_to_view) + '-marginals.pdf' for log_dir, metadata in zip(self.log_dir_list, self.metadata_list)]
        elif len(savepath_list) != len(self.metadata_list):
            raise ValueError('Length-' + str(len(self.metadata_list)) + ' savepath_list expected but ' +
                             'length-' + str(len(savepath_list)) + ' savepath_list received. ')

        if constraint_fn is None:
            constraint_fn = lambda elecset_samples: [True for elecset in elecset_samples]
        
        if color is None:
            color = ELECTRON_COLOR

        plotter_list = []

        for symm_visualizer, log_dir, metadata, show, samples_raw, savepath \
                in zip(self.space_visualizers, self.log_dir_list, self.metadata_list, 
                        show_list, self.samples_list, savepath_list):    
            
            num_elecs = np.sum(symm_visualizer.cell.nelec)

            # concatenate num_elec_to_view copies of samples with shifted indices
            samples_tmp = samples_raw.reshape([-1,num_elecs,3])
            if symm_visualizer.group.dims == 2:
                samples_tmp[:,:,2] = 0 # ignore z value
            samples_all = np.array([ np.roll(samples_tmp, shift, axis=1).reshape([-1,3]) for shift in range(num_elec_to_view)])
            samples_all = samples_all.swapaxes(0,1) # shape (num_samples, num_elec_to_view, 3)
            # filter array
            samples_all = samples_all[constraint_fn(samples_all),:,:]
            samples_all = samples_all.swapaxes(0,1) # shape (num_elec_to_view, num_filtered_samples, 3)
            
            if num_elec_to_view > 1:
                plotter = symm_visualizer.pv_plotter_setup(
                    subplot=True,
                    subplot_shape=(num_elec_to_view // 2 + 1, 2),
                    interactive=interactive,
                )
            else:
                plotter = symm_visualizer.pv_plotter_setup(
                    interactive=interactive,
                )

            for i, samples in enumerate(samples_all):
                if num_elec_to_view > 1:
                    plotter.subplot(i // 2, i % 2)

                plotter = symm_visualizer.plot_supercell_atom(
                                plotter=plotter,
                                show=False,
                        )
                samples_points = pv.PolyData(samples)
                plotter.add_mesh(
                    samples_points,
                    color=color,
                    opacity=opacity,
                    point_size=point_size,
                    render_points_as_spheres=True, 
                )

            plotter = symm_visualizer.pv_plotter_wrapup(
                plotter=plotter, 
                show=show,
                savepath=savepath,
            )
            plotter_list.append(plotter)
        
        return plotter_list
    

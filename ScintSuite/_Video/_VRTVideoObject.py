"""
Object to work with the VRT cameras.

"""

from ScintSuite._Video._BasicVideoObject import BVO
import xml.etree.ElementTree as et
import numpy as np
import tkinter as tk                       # To open UI windows
import ScintSuite._GUIs as ssGUI                   # For GUI elements
import matplotlib.pyplot as plt
import ScintSuite._TimeTrace as tt
import matplotlib.colors as colors
import ScintSuite._Plotting as ssplt
import ScintSuite._IO as ssio
import os
from ScintSuite._Machine import machine
import ScintSuite._VRT as vrt
import xarray as xr

# import numpy as np
# import ScintSuite._Mapping as ssmap
# import ScintSuite.LibData as ssdat
# import ScintSuite.SimulationCodes.FILDSIM as ssFILDSIM
# from ScintSuite.version_suite import version
# from scipy.io import netcdf                # To export remap data
# from tqdm import tqdm                      # For waitbars
import ScintSuite._Paths as p
pa = p.Path(machine)
del p

import logging
logger = logging.getLogger('ScintSuite.VRTVideo')
 
class VRTVideo(BVO):
    """
    Video class for the VRT cameras

    Heavily based on FILDVideoObject.py by Jose Rueda (jrrueda@us.es)

    Javier Hidalgo-Salaverri (jhsalaverri@us.es)

    Updated by Alex Reyner (areyner@us.es) to work with the shares

    """
    def __init__(self, camera: str, shot: int, quantity = 'raw'):
        """
        Initialise the class

        :param  camera: Camera name
        :param  shot: Shot number
        """

        self.shot = shot
        self.camera = camera

        folder= '/shares/experiments/aug-rawfiles/VRT/'+str(shot)[0:2]+'/S'+str(shot)
        self.path_video = None
        # The data format have changed many times. We will define diferent candidates
        # and go over them in order
        candidates = ['.npy', '.mp4']
        # Loop over candidates
        if os.path.isdir(folder):
            for ext in candidates:
                matches = []
                for root_dir, _, files in os.walk(folder):  # looks inside folders too
                    for fname in files:
                        if str(shot) in fname and camera in fname and fname.endswith(ext):
                            full_path = os.path.join(root_dir, fname)
                            if os.path.isfile(full_path):  # file found
                                matches.append(full_path)
                if matches:
                    if len(matches) > 1:
                        logger.warning('More than one coincidence, ' \
                        'selecting closer name')
                    self.path_video = min(matches, key=lambda p: (len(os.path.basename(p)), p))
                    break
        self.properties = {}

        if fname.endswith('.mp4'): #temporal, needs .mp4 upgrade
            # Get the time base and the corresponding frame number
            self.path_time = folder+'/Prot/FrameProt/'+camera+'_FrameProt.xml'
            root = et.parse(self.path_time).getroot()
            TS6 = int(root.attrib['ts6Time'],0)
            time = []
            nframe = []
            for r in root.findall('Entry'):
                time.append((int(r.attrib['time'],0)-TS6)*1e-9)
                nframe.append(int(r.attrib['frameNumber']))
            self.timebase = np.array(time, dtype=float)
            self.exp_dat['frames'] = self.exp_dat['frames'][::-1,:,:]
            self.exp_dat['tframes'] = np.array(time)
            self.exp_dat['nframes'] = np.array(nframe)

        BVO.__init__(self, file = self.path_video, shot = self.shot)
        self.exp_dat.coords['py'].attrs['long_name'] = 'Y pix'
        self.exp_dat.coords['px'].attrs['long_name'] = 'X pix'
        self.exp_dat.coords['t'].attrs['long_name'] = 'Time'
        self.exp_dat.coords['t'].attrs['units'] = 's'
        self.exp_dat['frames'].attrs['long_name'] = 'Counts'

        self.get_parameters()
        self.get_VRTrois(quantity = quantity)

    def get_parameters(self):
        fn = "/shares/experiments/aug-rawfiles/VRT/%02i/S%i/S%i_%s.meta.xml"\
            % (self.shot//1000, self.shot, self.shot, self.camera)
        metaroot = et.parse(fn).getroot()
        configXML = metaroot.find("config").find("AP_CONF[@type='config']")
        parxml=configXML.find('./*[@name="obj:Parameterizations"]')
        configXML = metaroot.find("config").find("AP_CONF[@type='config']")
        interlocksXML=metaroot.find("config").find("AP_CONF[@type='interlocks']")
        comparatorsXML=interlocksXML.findall('USAGE[@name="obj:Comparators"]/OBJECTLIST/OBJECT')
        messagesXML=metaroot.find("config").find("AP_CONF[@type='messages']")
        evaluatorsXML=configXML.findall('USAGE[@name="obj:Evaluators"]/OBJECTLIST/OBJECT')
        signalsXML=metaroot.findall("signals/signal")

        params = {}
        if parxml is not None:
            for p in parxml.findall(".//PARAMETER"):
                name = p.attrib.get("name", "")
                val = p.attrib.get("value", "")
                ptype = p.attrib.get("type", "")

                # Conversión de tipos básica si aplica
                parsed_val = val
                if ptype in ("uint32_t", "uint64_t", "int32_t", "int64_t"):
                    try:
                        parsed_val = int(val, 0)
                    except ValueError:
                        pass
                elif ptype in ("float", "double"):
                    try:
                        parsed_val = float(val)
                    except ValueError:
                        pass

                params[name] = parsed_val
        self.CameraData = params

    def get_VRTrois(self, quantity = 'temperature'):
        """
        This function gets the time traces for the ROIs defined in the camera.
        Makes use of the code developed by T.Lunt: tilmann.lunt@ipp.mpg.de

        Alex Reyner: areyner@us.es 

        :param  qunatity: units of the roi

        """

        class Args:
            """
            Arguments for the calibrated camera class
            """
            shot = 43178
            shottype = "s"
            quantity = "raw"
            pcal = "/shares/departments/AUG/users/vida/projects/camera_description/calibration/simple/"
            fncal = "01Eod_calibration.xml"

        class calibrated_camera:
            """
            T.Lunt: tilmann.lunt@ipp.mpg.de
            """
            def __init__(self, configXML, args):
                self.args=args       
                parxml=configXML.find('./*[@name="obj:Parameterizations"]')
                objxml=parxml.find('OBJECTLIST/OBJECT')
                self.camtype=camtype=objxml.attrib['name'].replace('Camera','')
                self.gain=gain=int(objxml.find('.//USAGE[@name="p:Gain"]/PARAMETER').attrib['value'])
                if camtype=='Jai':
                    shutter=int(objxml.find('.//USAGE[@name="p:ShutterPreset"]/PARAMETER').attrib['value'])
                    self.texp=texp={0:0.00833333333333,1:0.004,2:0.002,3:0.001,4:0.0005,5:0.00025,6:0.000125,7:0.0001,8:6.66666666667e-05,9:3.33333333333e-05}[shutter]
                else:
                    shutter=int(objxml.find('.//USAGE[@name="p:Shutter"]/PARAMETER').attrib['value'])
                    self.texp=texp=shutter*1e-6
                fnview = "/shares/departments/AUG/users/vida/projects/camera_description/views/view_%s.xml" % args.camera
                viewxml=et.parse(fnview).getroot()
                self.viewparameters=vp={}
                for m in viewxml.findall("modification"):
                    if int(m.attrib['pulse'])<args.shot:
                        for k,v in m.attrib.items():vp[k]=v
                fnobj = "/shares/departments/AUG/users/vida/projects/camera_description/objectives/objective_%s.xml" % vp['objectiveid']
                objxml=et.parse(fnobj).getroot()
                if  vp['aperture'].lower()!='none':
                    aperture_ring=float(vp['aperture']) 
                    art=self.aring_vs_atrans=np.array([[float(e.attrib['value']),float(e.attrib['transmission'])] for e in objxml.findall('aperture')])
                    self.a=np.interp(aperture_ring,art[:,0],art[:,1]) 
                else:
                    self.a=1.0
                    
                calroot = None
                if args.quantity in ['radiance','temperature']:
                    calroot = et.parse(args.pcal+'/'+args.fncal).getroot()
                    self.a_cal = float(calroot.attrib["a"])
                    self.LeCS = float(calroot.find("weighted_radiance_source").attrib["LeCS"])
                    coeffs = [c for c in calroot.findall("setting") if int(c.attrib["GA"])==gain][0]
                    self.c1w=float(coeffs.attrib["c1w"])
                    self.c1b=float(coeffs.attrib["c1b"])
                    self.c0w=float(coeffs.attrib["c0w"])
                    self.c0b=float(coeffs.attrib["c0b"])
                    self.T_vs_rad=np.array([[float(e.attrib['T']),float(e.attrib['LeW'])] for e in calroot.findall('radiance_table/entry')])
            
            def getlabel(self):
                return {'raw':['Raw signal', 'cts.'],
                        'radiance':['(weighted) radiance ', '[$Wm^{-2}sr^{-1}$]'],
                        'temperature':['Temp.', 'K']}[self.args.quantity]
            
            def raw2q(self,I):
                if self.args.quantity == 'raw':
                    return I
                else:
                    I_w = self.c1w*self.texp + self.c0w
                    I_b = self.c1b*self.texp + self.c0b          
                    L = (self.LeCS*self.a_cal/self.a) * (I-I_b)/(I_w-I_b)
                    if self.args.quantity == 'radiance':
                        return L
                    if self.args.quantity == 'temperature':
                        return np.interp(L,self.T_vs_rad[:,1],self.T_vs_rad[:,0])

        def _get_thresholds(sn):
            th = []
            for cXML in comparatorsXML:
                vec = cXML.find('USAGE[@name="p:FloatInputSignals"]/SIGNALVECTOR')
                if vec is not None and vec.attrib.get('members') == sn:
                    scaling = cXML.find('USAGE[@name="o:Scaling"]/SIGNAL')
                    nm = scaling.attrib.get('name', '') if scaling is not None else ''

                    message = ''
                    for mXML in messagesXML.findall('USAGE[@name="obj:Executors"]/OBJECTLIST/OBJECT'):
                        inp = mXML.find('USAGE[@name="p:InputSignal"]/SIGNAL')
                        if inp is not None and inp.attrib.get('name') == nm:
                            message = mXML.attrib.get('class', '')

                    param = cXML.find('USAGE[@name="p:ParameterValues"]/PARAMETER')
                    if param is not None and 'value' in param.attrib:
                        th.append([float(param.attrib['value']), message])
            return th

        def _get_signal(eval_name, eXML, k):
            for sXML in signalsXML:
                if sXML.attrib.get('name') == eval_name:
                    rows = [
                        ((int(t.attrib["timestamp"]) - ts6) * 1e-9, float(t.attrib[k]))
                        for t in sXML.findall('value')
                        if "timestamp" in t.attrib and k in t.attrib
                    ]
                    if rows:
                        tv = np.array(rows, dtype=np.float64)
                        return tv[:, 0], tv[:, 1]
            return None, None


        arguments = Args()    
        arguments.shot = self.shot
        arguments.camera = self.camera
        arguments.quantity = quantity
        filename = "/shares/experiments/aug-rawfiles/VRT/%02i/S%i/S%i_%s.meta.xml"\
                    % (self.shot//1000, self.shot, self.shot, self.camera)
    
        metaroot = et.parse(filename).getroot()
        ts6 = int(metaroot.attrib["ts6"], 0)
        configXML = metaroot.find("config").find("AP_CONF[@type='config']")
        interlocksXML=metaroot.find("config").find("AP_CONF[@type='interlocks']")
        comparatorsXML=interlocksXML.findall('USAGE[@name="obj:Comparators"]/OBJECTLIST/OBJECT')
        messagesXML=metaroot.find("config").find("AP_CONF[@type='messages']")
        evaluatorsXML=configXML.findall('USAGE[@name="obj:Evaluators"]/OBJECTLIST/OBJECT')
        signalsXML=metaroot.findall("signals/signal")
        c_cam = calibrated_camera(configXML, arguments)

        # Look for all evaluators
        evaluators_map = {}
        for eXML in evaluatorsXML:
            eval_param = eXML.find("USAGE[@name='p:EvaluationName']/PARAMETER")
            if eval_param is None or 'value' not in eval_param.attrib:
                continue
            eval_name = eval_param.attrib['value']
            c = eXML.attrib.get('class', '')
            if c == 'vrt::evaluator::Mean':
                sig_node = eXML.find("USAGE[@name='o:Mean']/SIGNAL")
                k = 'mean'
            elif c == 'vrt::evaluator::HotSpot':
                sig_node = eXML.find("USAGE[@name='o:Max']/SIGNAL")
                k = 'max'
            else:
                continue
            if sig_node is not None and 'name' in sig_node.attrib:
                evaluators_map[eval_name] = {
                    'eXML': eXML,
                    'signal_name': sig_node.attrib['name'],
                    'key': k,
                }
        eval_names = list(evaluators_map.keys())

        # One by one, get signal and thresholds of each ROI
        data = xr.Dataset()
        for name in eval_names:
            info = evaluators_map[name]
            t_vec, val_vec = _get_signal(name, info['eXML'], info['key'])
            ths = _get_thresholds(info['signal_name'])  # lista de [valor, mensaje]

            data[name] = (['t'], c_cam.raw2q(val_vec), 
                            {'signal': name, 
                             'long_name': f'{c_cam.getlabel()[0]}', 
                             'units': f'{c_cam.getlabel()[1]}', 
                             'thresholds': ths,}
            )
        data.coords['t'] = t_vec
        data.coords['t'].attrs['long_name'] = 'Time'
        data.coords['t'].attrs['units'] = 's'

        self.camera_rois = data

    def plot_frame(self, frame_number=None, ax=None, cmap='afmhot',
                   t: float = None, scale: str = 'linear',
                   **kwargs):
        """
        Plot a frame from the loaded frames

        :param  frame_number: Number of the frame to plot, relative to the video
            file, optional
        :param  ax: Axes where to plot, is none, just a new axes will be created
        :param  cmap: colormap to be used
        :param  t: if an specific time want to be used
        :param  scale: Scale for the plot: 'linear', 'sqrt', or 'log'

        :return fig, ax: the figure and axes where the frame has been drawn
        """

        # --- Check inputs:
        if (frame_number is not None) and (t is not None):
            raise Exception('Do not give frame number and time!')
        if (frame_number is None) and (t is None):
            raise Exception("Didn't you want to plot something?")

        # --- Load the frames
        # If we use the frame number explicitly
        if frame_number is not None:
            plot_frame = self.exp_dat.frames.isel(t=frame_number, method='nearest')
        elif t is not None:
            plot_frame = self.exp_dat.frames.sel(t=t, method='nearest')

        # --- Prepare the plot:
        default_kwargs = {
            # 'vmin': float(plot_frame.quantile(0.5).values),
            # 'vmax': float(plot_frame.quantile(0.99).values),
            'cbar_kwargs': {'fraction': 0.046, 'pad': 0.04,},
        }
        if scale == 'sqrt':
            default_kwargs['norm'] = colors.PowerNorm(0.5)
        elif scale == 'log':
            default_kwargs['norm'] = colors.LogNorm(0.5)
        plot_kwargs = {**default_kwargs, **kwargs}

        if ax is None:
            fig, ax = plt.subplots(constrained_layout=True)
        else:
            fig = ax.figure

        img = plot_frame.plot.imshow(ax=ax, cmap=cmap, **plot_kwargs)
        ax.set_aspect('equal', adjustable='box')

        return fig, ax

    def plot_rois(self):
        """
        Plots all roi in a camera
        """
        data = self.camera_rois
        rois = list(data.data_vars)

        fig, axs = plt.subplots(len(rois), 1, sharex=True)
        ax = np.atleast_1d(axs).ravel()

        for i, roi in enumerate(rois):
            da = data[roi]
            short_name = roi[len(self.camera):].lstrip("_") \
                if roi.startswith(self.camera) else roi
            lines = da.plot(ax=ax[i], label=short_name)

            th_pairs = da.attrs.get('thresholds', [])
            for th_val, msg in th_pairs:
                ls = '--' if 'EPE' in msg else ':'
                color = 'red' if 'EPE' in msg else 'limegreen'
                label = 'EPE' if 'EPE' in msg else 'INFO'
                ax[i].axhline(th_val, ls=ls, color=color, label=label)

            ax[i].legend()
            ax[i].set_xlabel('')

        ax[-1].set_xlabel('Time [s]')
        fig.suptitle(self.camera)

    # tt.create_roi does not exist
    # def create_roi(self, time: float = 0.0, save: bool = False,
    #                filename: str = ''):
    #     """
    #     Define a ROI to be used afterwards. It can be saved

    #     This is a wrapper for the create_roi function of the LibTimeTraces
    #     library by Jose Rueda (joserrueda@us.es)
    #     """
    #     # Plot the chosen time
    #     self.plot_frame(t = time)
    #     fig = plt.gcf()
    #     fig,roi = tt.create_roi(fig)

    #     if save:
    #         if filename == '':
    #             folder = os.path.join(pa.Results, 'VRT/masks/')
    #             filename = folder+'S'+str(self.shot)+'_'+self.camera
    #             file_id = 1
    #             exists = True
    #             while exists == True:
    #                 if os.path.isfile(filename+'_'+str(file_id)+'.nc'):
    #                     file_id += 1
    #                 else:
    #                     exists = False

    #             filename = filename+'_'+str(file_id)+'.nc'

    #         # Get the corresponding frame
    #         frame_index = np.argmin(abs(self.exp_dat['tframes'] - time))
    #         # tf = self.exp_dat['tframes'][frame_index]
    #         frame = self.exp_dat['frames'][:, :, frame_index].squeeze()
    #         mask = roi.get_mask(frame)
    #         ssio.save_mask(mask = mask,filename = filename, shot = self.shot,
    #                        frame = frame)
    #     return roi

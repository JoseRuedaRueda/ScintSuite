"""
Routines used to work with the VRT data

"""

# Use et to get info from the xml files
import xml.etree.ElementTree as et
import glob
import numpy as np
import os
try:
    from shapely.geometry import Point
    from shapely.geometry.polygon import Polygon
except ModuleNotFoundError:
    pass
from matplotlib import pyplot as plt


def get_cameras(shot):
    """
    Get the list of cameras, the ROIs and main properties for a certain shot

    Javier Hidalgo-Salaverri: jhsalaverri@us.es

    @param shot

    """
    # Get the camera xml configuration files
    conf_files = glob.glob('/afs/ipp/u/augd/rawfiles/VRT/'+str(shot)[0:2]
                           +'/S'+str(shot)+'/Conf/Guggi*.xml')
    conf_files += glob.glob('/afs/ipp/u/augd/rawfiles/VRT/'+str(shot)[0:2]+
                            '/S'+str(shot)+'/Conf/labrt.xml')

    cameras = {}
    for conf_file_path in conf_files:
        root = et.parse(conf_file_path).getroot()
        # Each camera configuration is under a "grabber" section
        for grabber in root.findall('Grabber'):
            rois = []
            lim = []
            for roi in grabber.findall('Area'):
                rois.append(roi.attrib['name'])
                lim.append(float(roi.attrib['limit3']))
            # Get the camera configuration
            enabled = True if grabber.attrib['enabled'] == 'true' else False
            # Get the camera configuration (SH, GA)
            SH = []; GA = []
            for cam_conf in grabber.findall('ConfInfo/Entry'):
                c = cam_conf.attrib['cmd']
                if '?' in c:
                    c = cam_conf.attrib['cmd'].split('?')[0]+'='
                    if '=' in cam_conf.attrib['reply']:
                        c += cam_conf.attrib['reply'].split('=')[1]
                    else:
                        c += cam_conf.attrib['reply']
                    #  I think this can be changed to c = cam_conf.attrib['reply']. Has this got any use?
                if 'SH' in c: SH = int(c.split('=')[1])
                if 'GA' in c: GA = int(c.split('=')[1])
            # The same camera may appear in two file and the info is updated
            if grabber.attrib['name'] in cameras:
                if not rois == []:
                    cameras[grabber.attrib['name']]['rois'] = rois
                if not lim == []:
                    cameras[grabber.attrib['name']]['limits'] = lim
                if not enabled == []:
                    cameras[grabber.attrib['name']]['enabled'] = enabled
                if not SH == []: cameras[grabber.attrib['name']]['SH'] = SH
                if not GA == []: cameras[grabber.attrib['name']]['GA'] = GA
            else:
                cameras[grabber.attrib['name']] = {'rois': rois,
                                                   'limits': lim,
                                                   'enabled': enabled,
                                                   'SH': SH,
                                                   'GA': GA}
            # print(cameras[grabber.attrib['name']])
            # if grabber.attrib['name'] == '07Eod1': print('SH = ' +str(SH))
            # if not SH == []: cameras[grabber.attrib['name']]['SH'] = SH
            # if not GA == []: cameras[grabber.attrib['name']]['GA'] = GA
    return cameras


def get_aperture(cam, shot):
    """
    Get the camera aperture for a certain shot

    Based on a script of Tilmann Lunt
    Javier Hidalgo-Salaverri: jhsalaverri@us.es

    @param cam: camera name
    @param shot
    """
    fn = '/afs/ipp/u/vida/vrt/config/camera-description/views/view_'+cam+'.xml'
    root = et.parse(fn).getroot()
    a_pos = np.nan
    for modif in root.findall('modification'):
        if 'aperture' in modif.attrib:
            if int(modif.attrib['pulse']) <= shot:
                a_pos = float(modif.attrib['aperture'])

    # Experimental aperture-transmission curve
    curve_path = '/afs/ipp/u/vida/vrt/config/camera-description/objectives/'\
        'objective_001.xml'
    tr = []
    objectiveroot = et.parse(curve_path).getroot()
    for e in objectiveroot.findall('aperture'):
        tr.append([float(e.attrib['value']),
                            float(e.attrib['transmission'])])
    tr = np.array(tr)
    ii = np.argsort(tr[:,0])
    tr = tr[ii]

    return np.interp(a_pos,tr[:,0],tr[:,1])


def get_calibration(cam, shot, GA, SH):
    """
    Get the camera calibration signal -> temperature for a certain shot

    Based on a script of Tilmann Lunt
    Javier Hidalgo-Salaverri: jhsalaverri@us.es

    @param cam: camera name
    @param shot
    """
    camrange = 1024
    cal_path = '/afs/ipp/u/vida/vrt/config/camera-description/'\
        'calibration/simple/'
    fn = cal_path+cam+'_calibration.xml'
    if not os.path.isfile(fn): return None
    calroot = et.parse(fn).getroot()
    if calroot.find('radiance_table') is None:return None

    a_cal=float(calroot.attrib['a']) #aperture during calibration

    a = get_aperture(cam, shot)

    LeCS = float(calroot.find('weighted_radiance_source').attrib['LeCS'])
    for setting in calroot.findall('setting'):
      if GA == int(setting.attrib['GA']):
         c0w = float(setting.attrib['c0w'])
         c1w = float(setting.attrib['c1w'])
         c0b = float(setting.attrib['c0b'])
         c1b = float(setting.attrib['c1b'])

    for sh in calroot.findall('parameterdescription/shutter'):
       if int(sh.attrib['SH'])==SH:
         texp = float(sh.attrib['t_exp'])

    rad = []
    for e in calroot.findall('radiance_table/entry'):
        rad.append((float(e.attrib['T']),float(e.attrib['LeW'])))
    rad = np.array(rad)
    ii = np.argsort(rad[:,0]);rad=rad[ii]

    T = np.zeros(camrange)
    for cnts in range(camrange):
       Ib=c1b*texp+c0b
       Iw=c1w*texp+c0w
       #Le=LeCS*a_cal/a*(cnts-Ib)/(Iw-Ib)
       Lebk=LeCS*a_cal/a*(cnts)/(Iw-Ib)

       T[cnts]=np.interp(Lebk,rad[:,1],rad[:,0])

    return T

def ROI2mask(path: str = '', nx: int = None, ny: int = None):
    """
    Convert a VRT ROI (or any other ROI defined with a .xml file) to a binary
    mask.

    Javier Hidalgo Salaverri: jhsalaverri@us.es

    @param path
    @param nx: x dimension of the frame
    @param ny: y dimension of the frame
    """
    # Load the ROI
    root = et.parse(path).getroot()
    y = []; x0 = []; x1 = [];
    for roi_point in root[0]:
        y.append(int(roi_point.attrib['y']))
        x0.append(int(roi_point.attrib['x0']))
        x1.append(int(roi_point.attrib['x1']))

    # The ROI is flipped in the x axis
    y = y[::-1]
    # Convert it to a single line
    x = np.concatenate((x1,x0[::-1]))
    y = np.concatenate((y,y[::-1]))

    # Create the binary mask
    mask = np.ones((nx,ny))
    roi = Polygon(np.column_stack((x,y)))
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            mask[i][j] = roi.contains(Point(i,j))
    mask = mask.astype(bool)
    return {'frame': None, 'mask': mask, 'nx': nx, 'ny': ny, 'shot': None}

def get_time_trace(shot: int = None, roiname: str = '',
                   calibrate: bool = False, plots: bool = True):
    """
    Get the time trace of a VRT camera in terms of signal/temperature
    @param roiname: ROI to plot. If empty, gets every ROI
    @param calibrate: get temperature calibration (if existing)
    @param plots

    @return tt: dictionary with time trace and threshold
    """
    calibration_path = '/afs/ipp/u/vida/vrt/config/camera-description/'
    calibration_path +='calibration/simple/'
    VRT_path = '/afs/ipp/u/augd/rawfiles/VRT/'+str(shot)[0:2]+'/S'+str(shot)
    camrange = 1024

    # Get the camera xml configuration files
    conf_files = glob.glob(VRT_path+'/Conf/Guggi*.xml')
    conf_files += glob.glob(VRT_path+'/Conf/labrt.xml')

    time_array = []
    temp_array = []
    temp_lim_array = []
    signal_array = []
    signal_lim_array = []
    camera_array = []

    for conf_file_path in conf_files:
        root = et.parse(conf_file_path).getroot()
        # Each camera configuration is under a "grabber" section
        for grabber in root.findall('Grabber'):
            # Get only the cameras that were enabled
            if grabber.attrib['enabled'].lower()=='true':
                cam = grabber.attrib['name']

                # Get the camera configuration (ID, SH, GA)
                for cam_conf in grabber.findall('ConfInfo/Entry'):
                    c = cam_conf.attrib['cmd']
                    if '?' in c:
                        c = cam_conf.attrib['cmd'].split('?')[0]+'='
                        if '=' in cam_conf.attrib['reply']:
                            c += cam_conf.attrib['reply'].split('=')[1]
                        else:
                            c += cam_conf.attrib['reply']

                    if 'SH' in c: SH=int(c.split('=')[1])
                    if 'GA' in c: GA=int(c.split('=')[1])

                for area in grabber.findall('Area'):

                    protocol = area.attrib['ProtocolFile'].replace('Prot/','')
                    file_path = VRT_path+'/Prot/%s' % protocol
                    root = et.parse(file_path).getroot()

                    TS6 = int(root.attrib['ts6'],0)
                    time = []
                    for r in root.find('Values').findall('Val'):
                        time.append((int(r.attrib['time'],0)-TS6)*1e-9)
                    time = np.array(time)
                    val = []
                    for r in root.find('Values'). findall('Val'):
                        val.append(float(r.text))
                    val = np.array(val)
                    lim = float(area.attrib['limit3'])
                    signal_lim = val*0+lim

                    lablim = ''
                    limcol = 'blue'
                    if area.attrib['doVpe'].lower() == 'true':
                        limcol = 'red' if np.any(val>=lim) else 'green'

                    # Get only the requested ROI. All of them by default
                    if roiname.lower() in protocol.lower():
                        Tlim = []
                        temp = []
                        if calibrate:
                            T = get_calibration(cam, shot, GA, SH)
                            if T is not None:
                                temp = np.interp(val*camrange,
                                                 np.arange(camrange),T)
                                Tlim=np.interp(val*0+lim*camrange,
                                               np.arange(camrange),T)
                            else:
                                print('No calibration for '+roiname)
                        else:
                            T = None

                        # Plot the results
                        if plots:
                            fig,ax=plt.subplots()
                            fig.suptitle('%i, %s (%s)' %(shot,cam,protocol),
                                         fontsize=22,fontweight='bold')
                            ax.text(0.03,0.85,lablim,fontsize=15,
                                    transform=ax.transAxes)

                            if T is None:
                                ax.plot(time,val,color='blue')
                                ax.plot(time,signal_lim,color=limcol)
                                ax.set_ylim(0.0,lim*1.1)
                                ax.set_ylabel('counts')
                            else:
                                ax.plot(time,temp,color='blue')
                                ax.plot(time,Tlim,color=limcol)
                                ax.set_ylim(800.0,np.max(Tlim)*1.1)
                                ax.set_ylabel('T [K]')

                            ax.set_xlim(0,None)
                            ax.set_xlabel('time [s]')

                        camera_array.append(area.attrib['name'])
                        time_array.append(time)
                        signal_array.append(val)
                        signal_lim_array.append(signal_lim)
                        temp_array.append(temp)
                        temp_lim_array.append(Tlim)
    plt.show()
    output = {
        'camera_ID': camera_array,
        'time': time_array,
        'signal': signal_array,
        'signal_lim': signal_lim_array,
        'temp': temp_array,
        'temp_lim': temp_lim_array}
    return output

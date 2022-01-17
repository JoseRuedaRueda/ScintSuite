""" 
Routines used to work with the VRT data

"""

# Use et to get info from the xml files
import xml.etree.ElementTree as et
import glob
import numpy as np
import os

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
            for roi in grabber.findall('Area'):
                rois.append(roi.attrib['name']) 
            # Get the camera configuration
            enabled = True if grabber.attrib['enabled'] == 'true' else False 
            # Get the camera configuration (SH, GA)
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
            cameras[grabber.attrib['name']] = {'rois': rois,
                                               'enabled': enabled,
                                               'SH': SH, 
                                               'GA': GA}
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
    
    rad=np.array([[float(e.attrib['T']),float(e.attrib['LeW'])] for e in calroot.findall('radiance_table/entry')])
    ii = np.argsort(rad[:,0]);rad=rad[ii]
     
    T = np.zeros(camrange) 
    for cnts in range(camrange):
       Ib=c1b*texp+c0b
       Iw=c1w*texp+c0w     
       #Le=LeCS*a_cal/a*(cnts-Ib)/(Iw-Ib)
       Lebk=LeCS*a_cal/a*(cnts)/(Iw-Ib)
       
       T[cnts]=np.interp(Lebk,rad[:,1],rad[:,0])
       
    return T 
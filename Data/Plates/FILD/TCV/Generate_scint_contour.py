
'''
Routine to generate scintillator contour for TCV scintillators to overlay in videos

Since we work with STLs in the machine geometry we need to rotate the scintillator plate to be in the Z plane

'''
import numpy as np
import os
import matplotlib.pylab as plt
import ScintSuite as ss
import ScintSuite._Mapping as ssmap


if __name__ == '__main__':


    ###
    geomID = 'scint_2023_'
    Geometry = ss.simcom.Geometry(GeomID=geomID)
    Geometry.plot2Dfilled(view = 'Scint', element_to_plot = [0,2], plot_pinhole = False)
    plt.show()
    Geometry.apply_movement()   
    Geometry.elements_to_stl(file_name_save= geomID + '_rotated_mm', units = 'mm', viewScint = True)
    #Geometry.writeGeometry(geomID + '_rotated_mm', viewScint = False)
    ###

    plate = 'Data/Plates/FILD/TCV/TCV2023.txt'
    scintillator = ssmap.Scintillator(file = plate)
    scintillator.code = 'fildsim'
    coord_rearanged = {'x1': np.array([scintillator._coord_real['x1'][0]]), 
                       'x2': np.array([scintillator._coord_real['x2'][0]]), 
                       'x3': np.array([scintillator._coord_real['x3'][0] ])}
    
    idx = 0
    while len(scintillator._coord_real['x1'])>=2:
        p1_x = scintillator._coord_real['x1'][idx]
        p1_y = scintillator._coord_real['x2'][idx]
        p1_z = scintillator._coord_real['x3'][idx]  

        scintillator._coord_real['x1'] = np.concatenate([scintillator._coord_real['x1'][:idx],scintillator._coord_real['x1'][idx+1:]])
        scintillator._coord_real['x2'] = np.concatenate([scintillator._coord_real['x2'][:idx],scintillator._coord_real['x2'][idx+1:]])
        scintillator._coord_real['x3'] = np.concatenate([scintillator._coord_real['x3'][:idx],scintillator._coord_real['x3'][idx+1:]])

        distances = np.sqrt( (scintillator._coord_real['x1'] - p1_x)**2 
                    + (scintillator._coord_real['x2'] - p1_y)**2 
                    #+ (scintillator._coord_real['x3'] - p1_z)**2 
                    
        )
        idx = np.argmin(distances)


        coord_rearanged['x1'] = np.concatenate([coord_rearanged['x1'],np.array([scintillator._coord_real['x1'][idx]])])
        coord_rearanged['x2'] = np.concatenate([coord_rearanged['x2'],np.array([scintillator._coord_real['x2'][idx]])])
        coord_rearanged['x3'] = np.concatenate([coord_rearanged['x3'],np.array([scintillator._coord_real['x3'][idx]])])

    scintillator._coord_real['x1'] = coord_rearanged['x1']
    scintillator._coord_real['x2'] = coord_rearanged['x2']
    scintillator._coord_real['x3'] = coord_rearanged['x3']



    with open(plate, 'w') as f:
        f.writelines([scintillator.name[0] + '\n'])
        f.writelines([scintillator.description[0] + '\n',
                        scintillator.description[1] + '\n'])
        f.writelines([str(2) + '\n',
                        str(len(scintillator._coord_real['x1'])) + '\n'])
        for it in range(len(scintillator._coord_real['x1']) ):

            f.writelines([str(scintillator._coord_real['x3'][it]) + ' ',
                          str(scintillator._coord_real['x1'][it]) + ' ',
                          str(scintillator._coord_real['x2'][it]) + '\n'])


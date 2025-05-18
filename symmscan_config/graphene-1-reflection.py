import numpy as np

def symm_cfg(L_Bohr):
    # compute the adj, opp and hyp lengths of a degree 15 triangle, whose vertex is used to lay out 6 coplanar electrons 
    ang = np.pi / 6

    adj_30 = 1/2. * L_Bohr
    opp_30 = adj_30 * np.tan(ang)
    hyp_30 = adj_30 / np.cos(ang) 

    rot_center_one = np.array([adj_30, opp_30, 0.])
    rot_center_two = np.array([0., hyp_30, 0.])
    dist = opp_30 / 2
    dist2 = dist * 1.8
    

    # note that the remaining 6 electrons are lined up along two vertical lines that pass through the atom positions
    elecs = np.array([
                      rot_center_two + np.array([dist * np.sin(ang),  dist * np.cos(ang), 0]),
                      rot_center_two + np.array([- dist * np.sin(ang),  dist * np.cos(ang), 0]),
                      rot_center_two + np.array([dist * np.sin(ang),  - dist * np.cos(ang), 0]),
                      rot_center_two + np.array([- dist * np.sin(ang),  - dist * np.cos(ang), 0]),
                      rot_center_two + np.array([dist,  0, 0]),
                      rot_center_two - np.array([dist,  0, 0]),
                      rot_center_two + np.array([dist2 * np.sin(ang),  dist2 * np.cos(ang), 0]),
                      rot_center_two + np.array([- dist2 * np.sin(ang),  dist2 * np.cos(ang), 0]),
                      rot_center_two + np.array([dist2 * np.sin(ang),  - dist2 * np.cos(ang), 0]),
                      rot_center_two + np.array([- dist2 * np.sin(ang),  - dist2 * np.cos(ang), 0]),
                      rot_center_two + np.array([dist2,  0, 0]),
                      rot_center_two - np.array([dist2,  0, 0]),
        ])
    camera = {
        'position': [0,0,1000],
        'viewup': [0,1,0],
        # 'zoom': 0.5,
    }
    return elecs, camera
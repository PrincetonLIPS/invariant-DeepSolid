import numpy as np

def symm_cfg(L_Bohr):
    # compute the adj, opp and hyp lengths of a degree 15 triangle, whose vertex is used to lay out 6 coplanar electrons 
    adj_15 = 1/2. * L_Bohr
    opp_15 = adj_15 * np.tan(np.pi/12)
    hyp_15 = adj_15 / np.cos(np.pi/12) 

    # note that the remaining 6 electrons are lined up along two vertical lines that pass through the atom positions
    elecs = np.array([ 
                      [adj_15, opp_15, 0.],
                      [hyp_15 / np.sqrt(2), hyp_15 / np.sqrt(2), 0.],
                      [L_Bohr - hyp_15 / np.sqrt(2), hyp_15 / np.sqrt(2), 0.],
                      [1/2. * L_Bohr, np.cos(np.pi/6)/3 * L_Bohr, .8],
                      [1/2. * L_Bohr, np.cos(np.pi/6)/3 * L_Bohr, 0],
                      [1/2. * L_Bohr, np.cos(np.pi/6)/3 * L_Bohr, -.8],
                      [hyp_15 * np.cos(np.pi * 5/12), hyp_15 * np.sin(np.pi * 5/12), 0.],
                      [-hyp_15 * np.cos(np.pi * 5/12), hyp_15 * np.sin(np.pi * 5/12), 0.],
                      [0., np.cos(np.pi/6) * L_Bohr - opp_15, 0.],
                      [0., np.cos(np.pi/6)*2/3 * L_Bohr, .8],
                      [0., np.cos(np.pi/6)*2/3 * L_Bohr, 0],
                      [0., np.cos(np.pi/6)*2/3 * L_Bohr, -.8]
        ])
    camera = {
        'position': [0,-10,40],
        'viewup': [0,1.7,1],
        # 'zoom': 0.5,
    }
    return elecs, camera
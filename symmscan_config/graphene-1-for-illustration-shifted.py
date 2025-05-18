import numpy as np

def symm_cfg(L_Bohr):
    # compute the adj, opp and hyp lengths of a degree 15 triangle, whose vertex is used to lay out 6 coplanar electrons 
    adj_15 = 1/2. * L_Bohr
    opp_15 = adj_15 * np.tan(np.pi/12)
    hyp_15 = adj_15 / np.cos(np.pi/12) 

    opp_30 = np.cos(np.pi/6)/3 * L_Bohr


    # note that the remaining 6 electrons are lined up along two vertical lines that pass through the atom positions
    elecs = np.array([ 
                      [adj_15 + adj_15, opp_15 - opp_30, 0.],
                      [hyp_15 / np.sqrt(2) + adj_15, hyp_15 / np.sqrt(2) - opp_30, 0.],
                      [L_Bohr - hyp_15 / np.sqrt(2) + adj_15, hyp_15 / np.sqrt(2) - opp_30, 0.],
                      [1/2. * L_Bohr + adj_15, np.cos(np.pi/6)/3 * L_Bohr - opp_30, .8],
                      [1/2. * L_Bohr + adj_15, np.cos(np.pi/6)/3 * L_Bohr - opp_30, 0],
                      [1/2. * L_Bohr + adj_15, np.cos(np.pi/6)/3 * L_Bohr - opp_30, -.8],
                      [hyp_15 * np.cos(np.pi * 5/12) + adj_15, hyp_15 * np.sin(np.pi * 5/12) - opp_30, 0.],
                      [-hyp_15 * np.cos(np.pi * 5/12) + adj_15, hyp_15 * np.sin(np.pi * 5/12) - opp_30, 0.],
                      [0. + adj_15, opp_30*3 - opp_15 - opp_30, 0.],
                      [0. + adj_15, opp_30*2 - opp_30, .8],
                      [0. + adj_15, opp_30*2 - opp_30, 0],
                      [0. + adj_15, opp_30*2 - opp_30, -.8]
        ])
    camera = {
        'position': [0,-10,40],
        'viewup': [0,1.7,1],
        # 'zoom': 0.5,
    }
    return elecs, camera
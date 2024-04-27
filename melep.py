import torch
torch.manual_seed(0)
import numpy as np

def compute_melep(
        model, 
        target_data, 
        source_labels,
    ):
    
    Y = target_data.CLASSES
    N = len(target_data)
    
    target_weights = target_data.y.sum(axis=0)/len(target_data)
    assert len(target_weights) == len(Y)
    
    ### Step 1
    loader = torch.utils.data.DataLoader(
            target_data, batch_size=128, shuffle=False
        )
    model.to(device=torch.device('cpu'))
    model.eval()

    dummy_probas = torch.tensor([], requires_grad=False)
    true_labels = torch.tensor([], requires_grad=False)

    with torch.no_grad():
        for i, (data, labels) in enumerate(loader):
            probas = torch.sigmoid(model(data))

            dummy_probas = torch.cat([dummy_probas, probas])
            true_labels = torch.cat([true_labels, labels])

    assert dummy_probas.shape[1] == len(source_labels)
    
    ### Step 2 & 3
    melep = []

    for y in range(len(Y)):
        melep_this_y = []
        
        for z in range(len(source_labels)):
            # Empirical joint distribution
            P_yz = torch.zeros((2, 2))
            z_eq_zero_dist = 1 - dummy_probas[:, z]
            P_yz[0][0] = z_eq_zero_dist[torch.argwhere(true_labels[:, y]==0)].sum()/N
            P_yz[1][0] = z_eq_zero_dist[torch.argwhere(true_labels[:, y]==1)].sum()/N
            P_yz[0][1] = dummy_probas[:, z][torch.argwhere(true_labels[:, y]==0)].sum()/N
            P_yz[1][1] = dummy_probas[:, z][torch.argwhere(true_labels[:, y]==1)].sum()/N

            # Empirical Marginal distribution
            P_z = P_yz.sum(axis=0)
            # Empirical conditional distribution
            P_y_given_z = P_yz / P_z

            theta_i_z0 = 1-dummy_probas[:, z]
            theta_i_z1 = dummy_probas[:, z]
            y_i = true_labels[:, y]

            melep_this_pairs = torch.mean(torch.log(
                P_y_given_z[y_i.long(),0] * theta_i_z0 + P_y_given_z[y_i.long(),1] * theta_i_z1
            ))

            melep_this_y.append(melep_this_pairs)
        
        melep.append(-np.average(melep_this_y)) 

    return np.average(melep, weights=target_weights)

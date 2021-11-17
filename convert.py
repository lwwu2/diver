""" convert checkpoint to training weight or serialized data for real-time deployment """
import torch
from pathlib import Path

from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    
    # add PROGRAM level args
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--serialize', type=int) # whether to extract serialized model
    
    parser.set_defaults(resume=False)
    args = parser.parse_args()

    file = Path(args.checkpoint_path)
    folder = file.parent

    weight_file = folder / 'weight.pth'

    # extract weight file
    if not weight_file.exists():
        state = torch.load(file, map_location='cpu')['state_dict']
        voxel_mask = torch.load(folder/'alpha_map.pt',map_location='cpu') > 1e-2 # load occupancy mask
        weight = {}
        for k,v in state.items():
            if 'mlp_voxel' in k: # drop implicit mlp
                continue
            if 'model.' in k:
                weight[k.replace('model.', '')] = v
        del state
        weight['voxel_mask'] = voxel_mask

        # find all occupied voxel
        ii,jj,kk = torch.where(voxel_mask)
        xx,yy,zz = torch.meshgrid(*[torch.arange(2)]*3)
        xx,yy,zz = xx.reshape(-1),yy.reshape(-1),zz.reshape(-1)
        ii,jj,kk = (ii[None]+xx[:,None]),(jj[None]+yy[:,None]),(kk[None]+zz[:,None])
        ii,jj,kk = ii.reshape(-1),jj.reshape(-1),kk.reshape(-1)

        idx = torch.stack([ii,jj,kk], dim=-1)
        idx = idx.unique(dim=0)
        ii,jj,kk = idx.T

        voxels = weight['voxels']
        voxel_num = voxels.shape[0]
        voxel_dim = voxels.shape[-1]

        sparse_voxels = torch.sparse_coo_tensor(
            idx.T, voxels[ii,jj,kk],
            (voxel_num,voxel_num,voxel_num,voxel_dim))
        
        weight['voxels'] = sparse_voxels
        torch.save(weight, weight_file)
    else:
        weight = torch.load(weight_file, map_location='cpu')
    
    # extract serialized model from the weight file
    if args.serialize == 1:
        voxel_dim = weight['voxels'].shape[-1]
        
        if voxel_dim == 64:
            linear1,bias1 = weight['mlp1.0.weight'],weight['mlp1.0.bias']
            linear21,bias21 = weight['mlp1.2.weight'],weight['mlp1.2.bias']

            linear22,bias22 = weight['mlp2.0.weight'],weight['mlp2.0.bias']
            linear3,bias3 = weight['mlp2.2.weight'],weight['mlp2.2.bias']

            linear40 = linear21[:1]
            bias40 = bias21[:1]
            # composite layers without activation
            linear41 = linear22[:,:64]@linear21[1:]
            bias41 = linear22[:,:64]@bias21[1:]+bias22
            linear42 = linear22[:,64:]
            linear4 = torch.cat([linear40,linear41],0)
            bias4 = torch.cat([bias40,bias41],0)

            new_weight = {'voxels': weight['voxels'],'voxel_mask': weight['voxel_mask']}
            new_weight['weight0'] = torch.cat([bias1[:,None],linear1],1)
            new_weight['weight1'] = torch.cat([bias4[0,None],linear4[0]],0)
            new_weight['weight2'] = torch.cat([bias4[1:,None],linear4[1:],linear42],1)
            new_weight['weight3'] = torch.cat([weight['mlp2.2.bias'][:,None],weight['mlp2.2.weight']],dim=-1)

            w2 = new_weight['weight2']
            w21 = w2[:,:-27]
            w22 = w2[:,-27:]
            # recorder positional encoding in a way easy to be processed by the cuda code
            w2 = torch.cat([w21,w22[:,torch.tensor([
                0,3,9,15,21,6,12,18,24,
                1,4,10,16,22,7,13,19,25,
                2,5,11,17,23,8,14,20,26])]],-1)

            params = torch.cat([
                new_weight['weight0'].T.reshape(-1),
                new_weight['weight1'].reshape(-1),
                w2.T.reshape(-1),
                new_weight['weight3'].T.reshape(-1),
                torch.zeros(28),
            ],0).contiguous() # padding to fit a warp in cuda

        elif voxel_dim == 32:
            linear01,bias01 = weight['mlp1.0.weight'],weight['mlp1.0.bias']
            linear11,bias11 = weight['mlp1.2.weight'],weight['mlp1.2.bias']
            linear21,bias21 = weight['mlp1.4.weight'],weight['mlp1.4.bias']

            linear22,bias22 = weight['mlp2.0.weight'],weight['mlp2.0.bias']
            linear3,bias3 = weight['mlp2.2.weight'],weight['mlp2.2.bias']

            linear40 = linear21[:1]
            bias40 = bias21[:1]
            linear41 = linear22[:,:32]@linear21[1:]
            bias41 = linear22[:,:32]@bias21[1:]+bias22
            linear42 = linear22[:,32:]
            linear4 = torch.cat([linear40,linear41],0)
            bias4 = torch.cat([bias40,bias41],0)

            new_weight = {'voxels': weight['voxels'],'voxel_mask': weight['voxel_mask']}
            new_weight['weightin'] = torch.cat([bias01[:,None],linear01],1)
            new_weight['weight0'] = torch.cat([bias11[:,None],linear11],1)
            new_weight['weight1'] = torch.cat([bias4[0,None],linear4[0]],0)
            new_weight['weight2'] = torch.cat([bias4[1:,None],linear4[1:],linear42],1)
            new_weight['weight3'] = torch.cat([weight['mlp2.2.bias'][:,None],weight['mlp2.2.weight']],dim=-1)
        
            w2 = new_weight['weight2']
            w21 = w2[:,:-27]
            w22 = w2[:,-27:]
            w2 = torch.cat([w21,w22[:,torch.tensor([
                0,3,9,15,21,6,12,18,24,
                1,4,10,16,22,7,13,19,25,
                2,5,11,17,23,8,14,20,26])]],-1)

            params = torch.cat([
                new_weight['weightin'][:,0],
                new_weight['weight0'].T.reshape(-1),
                new_weight['weight1'].reshape(-1),
                w2.T.reshape(-1),
                new_weight['weight3'].T.reshape(-1),
                torch.zeros(28),
            ],0).contiguous() 

        # store feature vectors in an 1d array
        voxels = new_weight['voxels'].coalesce()
        N = voxels.shape[0]-1
        idxs = voxels.indices()
        features = voxels.values()

        # pre-multiply first linear layer to feature vector for diver32 model
        if voxel_dim == 32:
            features = (features@new_weight['weightin'][:,1:].T).contiguous()

        valid = (idxs<N).all(0)
        # indices of occupancy mask is a subset of indices of feature vectors
        is_corner = new_weight['voxel_mask'][idxs[0][valid],idxs[1][valid],idxs[2][valid]]
        mask_mask = torch.zeros_like(idxs[0]).bool()
        mask_mask[valid] = is_corner
        # store indices as flattened
        inds = idxs[0]*(N+1)*(N+1)+idxs[1]*(N+1)+idxs[2]
        serialize = {'f':features,'i':inds.int(),'m': mask_mask,'p':params}

        torch.save(serialize, folder/'serialize.pth')
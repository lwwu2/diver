import torch

def integrate_mlp(voxel_mlp, N, C, coord_i, coord_o):
    """ integrte features along the ray with implicit model
    Args:
        voxel_mlp: implicit voxel grid mlp
        N: voxel grid size
        C: voxel feature dim
        coord_i: Bx3 voxel entry point
        coord_o: Bx3 voxel exit point
    Return:
        feature: BxC integrated features
    """

    # accurate voxel corner calculation
    pmin = torch.min((coord_i+1e-4).long(),(coord_o+1e-4).long())
    
    # get eight vertices of voxels
    xmin,ymin,zmin = pmin.split(1,dim=-1)
    xmax,ymax,zmax = (pmin+1).clamp_max(N-1).split(1,dim=-1)
    corners = torch.stack([
        zmin,ymin,xmin, zmin,ymin,xmax,
        zmin,ymax,xmin, zmin,ymax,xmax,
        zmax,ymin,xmin, zmax,ymin,xmax,
        zmax,ymax,xmin, zmax,ymax,xmax
    ],dim=-1).reshape(-1,3)
    
    # filter out repeated vertices
    corners_u, corners_rev = corners.unique(dim=0, return_inverse=True)
    
    # query MLP, 2x2x2xBxC features
    F_u = voxel_mlp(corners_u.float()/(N*0.5)-1.0)
    F = F_u[corners_rev].reshape(zmin.shape[0],2,2,2,C).permute(1,2,3,0,4)
    
    # local coordinates of eight vertices
    P = torch.stack([coord_i-pmin, 
                     coord_o-pmin],0).permute(0,2,1)

    # calculate integrated features
    A = P.sum(0).prod(0)/12+(P.prod(1).sum(0))/6
    X = F[1] - F[0]
    X = X[0] - X[1]
    X = X[0] - X[1]
    feature = A[:,None]*X
    
    P_roll = P[:,[1,2,0]]
    # 3xBxM
    B = (P*P_roll).sum(0)/3+(P*P_roll[[1,0]]).sum(0)/6
    # 3xBxMxC
    Y = F[0,0,0,None]-F[[0,0,1],[1,1,0],0]\
       -F[[0,1,0],0,[1,0,1]]+F[[0,1,1],[1,1,0],[1,0,1]]
    feature = feature + torch.einsum('dk,dkc->kc',B,Y)
    
    # 3xBxM
    C = P.sum(0)/2
    # 3xBxMxC
    Z = F[[0,0,1],[0,1,0],[1,0,0]]-F[0,0,0,None]
    feature = feature + torch.einsum('dk,dkc->kc',C,Z)
    feature = feature + F[0,0,0]
    return feature

def integrate(voxels, coord_i, coord_o):
    """ integrte features along the ray with explicit model
    Args:
        voxels: NxNxNxC dense voxel grid of feature vectors
        coord_i: Bx3 voxel entry point
        coord_o: Bx3 voxel exit point
        mask: BxM bool tensor of missing intersection indicator
    Return:
        feature: BxC integrated features
    """
    N,C = voxels.shape[-2:]
    
    # accurate voxel corner calculation
    pmin = torch.min((coord_i+1e-4).long(),(coord_o+1e-4).long())
    
    # explicit voxel grid query
    xmin,ymin,zmin = pmin.split(1,dim=-1)
    xmax,ymax,zmax = (pmin+1).clamp_max(N-1).split(1,dim=-1)
    F = torch.stack([
        voxels[zmin,ymin,xmin],voxels[zmin,ymin,xmax],
        voxels[zmin,ymax,xmin],voxels[zmin,ymax,xmax],
        voxels[zmax,ymin,xmin],voxels[zmax,ymin,xmax],
        voxels[zmax,ymax,xmin],voxels[zmax,ymax,xmax],
    ],0).reshape(2,2,2,-1,C)

    # local coordinates of eight vertices
    P = torch.stack([coord_i-pmin, 
                     coord_o-pmin],0).permute(0,2,1)

    # get integrated features
    A = P.sum(0).prod(0)/12+(P.prod(1).sum(0))/6
    X = F[1] - F[0]
    X = X[0] - X[1]
    X = X[0] - X[1]
    feature = A[:,None]*X
    
    P_roll = P[:,[1,2,0]]
    # 3xBxM
    B = (P*P_roll).sum(0)/3+(P*P_roll[[1,0]]).sum(0)/6
    # 3xBxMxC
    Y = F[0,0,0,None]-F[[0,0,1],[1,1,0],0]\
       -F[[0,1,0],0,[1,0,1]]+F[[0,1,1],[1,1,0],[1,0,1]]
    feature = feature + torch.einsum('dk,dkc->kc',B,Y)
    
    # 3xBxM
    C = P.sum(0)/2
    # 3xBxMxC
    Z = F[[0,0,1],[0,1,0],[1,0,0]]-F[0,0,0,None]
    feature = feature + torch.einsum('dk,dkc->kc',C,Z)
    feature = feature + F[0,0,0]
    
    return feature
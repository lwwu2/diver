default_options = {
    # dataset config
    'im_shape':{
        'type': int,
        'nargs': 2,
        'default': [image_height,image_width]
    },
    'batch_size':{
        'type': int,
        'default': batch_size
    },
    'dataset': {
        'type': str,
        'nargs': 2,
        'default': [{'blender', 'tanks'}, '{DATASET_PATH}/{SCENE}']
    },
    'num_workers': {
        'type': int,
        'default': dataloader_worker_thread_num
    },


    # coarse to fine config
    'fine':{
        'type': int,
        'default': {0,1} # 0 for coarse stage 1 for fine stage
    },
    'coarse_path':{
        'type': str,
        'default': '{CHECKPOINT_FOLDER}'
    },
    'mask_scale':{
        'type': float,
        'default': releative_scale_to_voxel_grid
    },
    
    
    # training strategy config
    'implicit':{
        'type': bool,
        'default': whehter_to_use_implicit_model
    },
    'thresh_a':{
        'type': float,
        'default': 1e-2
    },
    

    # optimizer config
    'optimizer': {
        'type': str,
        'choices': ['SGD', 'Ranger', 'Adam'],
        'default': 'Adam'
    },
    'learning_rate': {
        'type': float,
        'default': learning_rate
    },
    'weight_decay': {
        'type': float,
        'default': 0
    },

    'scheduler_rate':{
        'type': float,
        'default': 0.5
    },
    'milestones':{
        'type': int,
        'nargs': '*',
        'default': [1000] # never used
    },
    
    
    # voxel grid config
    'grid_size':{
        'type': float,
        'default': 2.8
    },
    'voxel_num':{
        'type': int,
        'default': number_of_voxels_along_each_axis
    },
    'voxel_dim':{
        'type': int,
        'default': 32
    },
    
    
    # regularization loss config
    'l_s':{
        'type': float,
        'default': 1e-5
    },
    
    
    # implicit model config
    'implicit_network_depth':{
        'type': int,
        'default': 8
    },
    'implicit_channels':{
        'type': int,
        'default': 512
    },
    'implicit_skips':{
        'type': int,
        'nargs':'*',
        'default': [4]
    },
    'implicit_point_encode':{
        'type': int,
        'default': 10
    },
    
    
    # decoer mlp config
    'mlp_point': {
        'type': int,
        'nargs': 3,
        'default': [hidden_dimension,network_depth,output_dimension]
    },
    'mlp_view': {
        'type': int,
        'nargs': 2,
        'default': [hidden_dimension,network_depth]
    },
    'dir_encode':{
        'type': int,
        'default': 4
    },
    
    
    # rendering config
    'white_back': {
        'type': bool,
        'default': True # if background color is white.
    }
}

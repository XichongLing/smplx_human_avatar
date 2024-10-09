import numpy as np
import torch
import torch.nn as nn
import tinycudann as tcnn
from omegaconf import OmegaConf


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d
        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, input_dims=3):
    if multires == 0:
        return lambda x: x, input_dims
    assert multires > 0

    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }
    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim


class HannwEmbedder:
    def __init__(self, cfg, **kwargs):
        self.cfg = cfg
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)

        # get hann window weights
        if self.cfg.full_band_iter <= 0 or self.cfg.kick_in_iter >= self.cfg.full_band_iter:
            alpha = torch.tensor(N_freqs, dtype=torch.float32)
        else:
            kick_in_iter = torch.tensor(self.cfg.kick_in_iter,
                                        dtype=torch.float32)
            t = torch.clamp(self.kwargs['iter_val'] - kick_in_iter, min=0.)
            N = self.cfg.full_band_iter - kick_in_iter
            m = N_freqs
            alpha = m * t / N

        for freq_idx, freq in enumerate(freq_bands):
            w = (1. - torch.cos(np.pi * torch.clamp(alpha - freq_idx,
                                                    min=0., max=1.))) / 2.
            # print("freq_idx: ", freq_idx, "weight: ", w, "iteration: ", self.kwargs['iter_val'])
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq, w=w: w * p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_hannw_embedder(cfg, multires, iter_val,):
    embed_kwargs = {
        'include_input': False,
        'input_dims': 3,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'periodic_fns': [torch.sin, torch.cos],
        'iter_val': iter_val
    }

    embedder_obj = HannwEmbedder(cfg, **embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim

# need to chaneg here for smplx
class HierarchicalPoseEncoder(nn.Module):
    '''Hierarchical encoder from LEAP.'''

    def __init__(self, num_joints=24, rel_joints=False, dim_per_joint=6, out_dim=-1, **kwargs):
        super().__init__()

        self.num_joints = num_joints
        self.rel_joints = rel_joints
        # cfg.kintree_table
        if (num_joints == 24):
            self.ktree_parents = np.array([-1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,
            9,  9,  9, 12, 13, 14, 16, 17, 18, 19, 20, 21], dtype=np.int32)
        else:
            self.ktree_parents = np.array([-1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  9,  9, 12,
                13, 14, 16, 17, 18, 19, 15, 15, 15, 20, 25, 26, 20, 28, 29, 20,
                31, 32, 20, 34, 35, 20, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46,
                47, 21, 49, 50, 21, 52, 53], dtype = np.int32)


        self.layer_0 = nn.Linear(9*num_joints + 3*num_joints, dim_per_joint)
        dim_feat = 13 + dim_per_joint

        layers = []
        for idx in range(num_joints):
            layer = nn.Sequential(nn.Linear(dim_feat, dim_feat), nn.ReLU(), nn.Linear(dim_feat, dim_per_joint))

            layers.append(layer)

        self.layers = nn.ModuleList(layers)

        if out_dim <= 0:
            self.out_layer = nn.Identity()
            self.n_output_dims = num_joints * dim_per_joint
        else:
            self.out_layer = nn.Linear(num_joints * dim_per_joint, out_dim)
            self.n_output_dims = out_dim

    def forward(self, rots, Jtrs, skinning_weight=None):
        batch_size = rots.size(0)

        if self.rel_joints:
            with torch.no_grad():
                Jtrs_rel = Jtrs.clone()
                Jtrs_rel[:, 1:, :] = Jtrs_rel[:, 1:, :] - Jtrs_rel[:, self.ktree_parents[1:], :]
                Jtrs = Jtrs_rel.clone()

        global_feat = torch.cat([rots.view(batch_size, -1), Jtrs.view(batch_size, -1)], dim=-1)
        global_feat = self.layer_0(global_feat)
        # global_feat = (self.layer_0.weight@global_feat[0]+self.layer_0.bias)[None]
        out = [None] * self.num_joints
        for j_idx in range(self.num_joints):
            rot = rots[:, j_idx, :]
            Jtr = Jtrs[:, j_idx, :]
            parent = self.ktree_parents[j_idx]
            if parent == -1:
                bone_l = torch.norm(Jtr, dim=-1, keepdim=True)
                in_feat = torch.cat([rot, Jtr, bone_l, global_feat], dim=-1)
                out[j_idx] = self.layers[j_idx](in_feat)
            else:
                parent_feat = out[parent]
                bone_l = torch.norm(Jtr if self.rel_joints else Jtr - Jtrs[:, parent, :], dim=-1, keepdim=True)
                in_feat = torch.cat([rot, Jtr, bone_l, parent_feat], dim=-1)
                out[j_idx] = self.layers[j_idx](in_feat)

        out = torch.cat(out, dim=-1)
        out = self.out_layer(out)
        return out

class VanillaCondMLP(nn.Module):
    def __init__(self, dim_in, dim_cond, dim_out, config, dim_coord=3):
        super(VanillaCondMLP, self).__init__()

        self.n_input_dims = dim_in
        self.n_output_dims = dim_out

        self.n_neurons, self.n_hidden_layers = config.n_neurons, config.n_hidden_layers

        self.config = config
        dims = [dim_in] + [self.n_neurons for _ in range(self.n_hidden_layers)] + [dim_out]

        self.embed_fn = None
        if config.multires > 0:
            embed_fn, input_ch = get_embedder(config.multires, input_dims=dim_in)
            self.embed_fn = embed_fn
            dims[0] = input_ch

        self.last_layer_init = config.get('last_layer_init', False)

        self.num_layers = len(dims)
        for l in range(0, self.num_layers - 1):
            if l + 1 in config.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            if l in config.cond_in:
                lin = nn.Linear(dims[l] + dim_cond, out_dim)
            else:
                lin = nn.Linear(dims[l], out_dim)

            if self.last_layer_init and l == self.num_layers - 2:
                torch.nn.init.normal_(lin.weight, mean=0., std=1e-5)
                torch.nn.init.constant_(lin.bias, val=0.)


            setattr(self, "lin" + str(l), lin)

        self.activation = nn.LeakyReLU()

    def forward(self, coords, cond=None):
        if cond is not None:
            cond = cond.expand(coords.shape[0], -1)

        if self.embed_fn is not None:
            coords_embedded = self.embed_fn(coords)
        else:
            coords_embedded = coords

        x = coords_embedded
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.config.cond_in:
                x = torch.cat([x, cond], 1)

            if l in self.config.skip_in:
                x = torch.cat([x, coords_embedded], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)

        return x

def get_skinning_mlp(n_input_dims, n_output_dims, config):
    if config.otype == 'VanillaMLP':
        network = VanillaCondMLP(n_input_dims, 0, n_output_dims, config)
    else:
        raise ValueError

    return network

def get_deformation_mlp(n_input_dims, n_cond_dims, n_output_dims, config):
    network = VanillaCondMLP(n_input_dims, n_cond_dims, n_output_dims, config)
    return network

def get_bone_encoder(config):
    network = BoneTransformEncoder(config.input_dim, config.output_dim)
    return network

def get_ImplicitNet(config):
    return ImplicitNet(config)

class HannwCondMLP(nn.Module):
    def __init__(self, dim_in, dim_cond, dim_out, config, dim_coord=3):
        super(HannwCondMLP, self).__init__()

        self.n_input_dims = dim_in
        self.n_output_dims = dim_out

        self.n_neurons, self.n_hidden_layers = config.n_neurons, config.n_hidden_layers

        self.config = config
        dims = [dim_in] + [self.n_neurons for _ in range(self.n_hidden_layers)] + [dim_out]

        self.embed_fn = None
        if config.multires > 0:
            _, input_ch = get_hannw_embedder(config.embedder, config.multires, 0)
            dims[0] = input_ch

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            if l + 1 in config.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            if l in config.cond_in:
                lin = nn.Linear(dims[l] + dim_cond, out_dim)
            else:
                lin = nn.Linear(dims[l], out_dim)

            if l in config.cond_in:
                # Conditional input layer initialization
                torch.nn.init.constant_(lin.weight[:, -dim_cond:], 0.0)
            torch.nn.init.constant_(lin.bias, 0.0)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.ReLU()

    def forward(self, coords, iteration, cond=None):
        if cond is not None:
            cond = cond.expand(coords.shape[0], -1)

        if self.config.multires > 0:
            embed_fn, _ = get_hannw_embedder(self.config.embedder, self.config.multires, iteration)
            coords_embedded = embed_fn(coords)
        else:
            coords_embedded = coords

        x = coords_embedded
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.config.cond_in:
                x = torch.cat([x, cond], 1)

            if l in self.config.skip_in:
                x = torch.cat([x, coords_embedded], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)

        return x

def config_to_primitive(config, resolve=True):
    return OmegaConf.to_container(config, resolve=resolve)

class HashGrid(nn.Module):
    def __init__(self, config):
        super().__init__()
        xL = config.get('max_resolution', -1)
        if xL > 0:
            L = config.n_levels
            x0 = config.base_resolution
            config.per_level_scale = float(np.exp(np.log(xL / x0) / (L - 1)))
        self.encoding = tcnn.Encoding(3, config_to_primitive(config))
        self.n_output_dims = self.encoding.n_output_dims
        self.n_input_dims = self.encoding.n_input_dims

    def forward(self, x):
        x = (x + 1.) * 0.5 # [-1, 1] => [0, 1]

        return self.encoding(x)
    

class ImplicitNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        dims = [config.d_in] + list(
            config.dims) + [config.d_out + config.feature_vector_size]

        self.num_layers = len(dims)
        self.skip_in = config.skip_in
        self.embed_fn = None
        self.config = config

        self.dim_in = config.d_in
        self.dim_out = config.d_out + config.feature_vector_size


        if config.multires > 0:
            embed_fn, input_ch = get_embedder(config.multires, input_dims=config.d_in)
            self.embed_fn = embed_fn
            dims[0] = input_ch
        self.cond = config.cond   
        if self.cond == 'smpl':
            self.cond_layer = [0]
            self.cond_dim = 73      # 64 (smpl encoding) + 9 (time encoding)
        elif self.cond == 'frame':
            self.cond_layer = [0]
            self.cond_dim = config.dim_frame_encoding
        elif self.cond == "semantic" and config.dim_semantic_encoding > 0:
            self.cond_layer = [0]
            self.lin_p0 = torch.nn.Linear(256, config.dim_semantic_encoding)
            self.cond_dim = config.dim_semantic_encoding
        elif self.cond == 'smpl+time':
            self.cond_layer = [0]
            self.cond_dim = 69 + 9
        self.dim_pose_embed = 0
        if self.dim_pose_embed > 0:
            self.lin_p0 = torch.nn.Linear(self.cond_dim, self.dim_pose_embed)
            self.cond_dim = self.dim_pose_embed
        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]
            
            if self.cond != 'none' and l in self.cond_layer:
                lin = torch.nn.Linear(dims[l] + self.cond_dim, out_dim)
            else:
                lin = torch.nn.Linear(dims[l], out_dim)
            if config.init == 'geometry':
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight,
                                          mean=np.sqrt(np.pi) /
                                          np.sqrt(dims[l]),
                                          std=0.0001)
                    torch.nn.init.constant_(lin.bias, -config.bias)
                elif config.multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0,
                                          np.sqrt(2) / np.sqrt(out_dim))
                elif config.multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0,
                                          np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):],
                                            0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0,
                                          np.sqrt(2) / np.sqrt(out_dim))
            if config.init == 'zero':
                init_val = 1e-5
                if l == self.num_layers - 2:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.uniform_(lin.weight, -init_val, init_val)
            if config.weight_norm:
                lin = torch.nn.utils.weight_norm(lin)
            setattr(self, "lin" + str(l), lin)
        self.softplus = torch.nn.Softplus(beta=100)

    def forward(self, input, cond, time_enc=None):
        if input.ndim == 2: input = input.unsqueeze(0)

        num_batch, num_point, num_dim = input.shape
        out_dim = self.config.d_out + self.config.feature_vector_size

        if num_batch * num_point == 0:
            return torch.zeros(num_batch, num_point, out_dim, device=input.device)

        input = input.reshape(num_batch * num_point, num_dim)
        if self.cond != 'none':
            if self.cond == "semantic":
                input_cond = cond[self.cond].squeeze(0)
                input_cond = self.lin_p0(input_cond)
            elif self.cond == 'smpl+time':
                input_cond = cond['smpl'].shape[1]
                input_cond = cond['smpl'].unsqueeze(1).expand(num_batch, num_point, input_cond)
                input_cond = input_cond.reshape(num_batch * num_point, input_cond.shape[2])
            else:
                num_cond = cond[self.cond].shape[1]
                input_cond = cond[self.cond].unsqueeze(1).expand(num_batch, num_point, num_cond)
                input_cond = input_cond.reshape(num_batch * num_point, num_cond)
            if self.dim_pose_embed:
                input_cond = self.lin_p0(input_cond)

        if self.embed_fn is not None:
            input = self.embed_fn(input)
        if time_enc is not None:
            # time_enc = time_enc.unsqueeze(1).expand(num_batch, num_point, -1)
            time_enc = time_enc.expand(num_batch, num_point, -1)
            time_enc = time_enc.reshape(num_batch * num_point, -1).cuda()
            input_cond = torch.cat([input_cond, time_enc], dim=-1)
        x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            if self.cond != 'none' and l in self.cond_layer:
                x = torch.cat([x, input_cond], dim=-1)
            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.softplus(x)
        # import ipdb; ipdb.set_trace()
        # x = x.reshape(num_batch, num_point, out_dim)
        return x

    def gradient(self, x, cond):
        x.requires_grad_(True)
        y = self.forward(x, cond)[:, :1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(outputs=y,
                                        inputs=x,
                                        grad_outputs=d_output,
                                        create_graph=True,
                                        retain_graph=True,
                                        only_inputs=True)[0]
        return gradients.unsqueeze(1)
    
class BoneTransformEncoder(nn.Module):
    def __init__(self, input_dim=384, output_dim=55):
        super(BoneTransformEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)  # No activation on output, could be learned features
        return x
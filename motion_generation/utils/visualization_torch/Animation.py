import operator
import torch 
import numpy as np
# import numpy.core.umath_tests as ut

from utils.visualization_torch.Quaternions import Quaternions


class Animation:
    """
    Animation is a numpy-like wrapper for animation data

    Animation data consists of several arrays consisting
    of F frames and J joints.

    The animation is specified by

        rotations : (F, J) Quaternions | Joint Rotations
        positions : (F, J, 3) ndarray  | Joint Positions

    The base pose is specified by

        orients   : (J) Quaternions    | Joint Orientations
        offsets   : (J, 3) ndarray     | Joint Offsets

    And the skeletal structure is specified by

        parents   : (J) ndarray        | Joint Parents
    """

    def __init__(self, rotations, positions, orients, offsets, parents, names, frametime):

        self.rotations = rotations
        self.positions = positions
        self.orients = orients
        self.offsets = offsets
        self.parents = parents
        self.names = names
        self.frametime = frametime

    def __op__(self, op, other):
        return Animation(
            op(self.rotations, other.rotations),
            op(self.positions, other.positions),
            op(self.orients, other.orients),
            op(self.offsets, other.offsets),
            op(self.parents, other.parents))

    def __iop__(self, op, other):
        self.rotations = op(self.roations, other.rotations)
        self.positions = op(self.roations, other.positions)
        self.orients = op(self.orients, other.orients)
        self.offsets = op(self.offsets, other.offsets)
        self.parents = op(self.parents, other.parents)
        return self

    def __sop__(self, op):
        return Animation(
            op(self.rotations),
            op(self.positions),
            op(self.orients),
            op(self.offsets),
            op(self.parents))

    def __add__(self, other):
        return self.__op__(operator.add, other)

    def __sub__(self, other):
        return self.__op__(operator.sub, other)

    def __mul__(self, other):
        return self.__op__(operator.mul, other)

    def __div__(self, other):
        return self.__op__(operator.div, other)

    def __abs__(self):
        return self.__sop__(operator.abs)

    def __neg__(self):
        return self.__sop__(operator.neg)

    def __iadd__(self, other):
        return self.__iop__(operator.iadd, other)

    def __isub__(self, other):
        return self.__iop__(operator.isub, other)

    def __imul__(self, other):
        return self.__iop__(operator.imul, other)

    def __idiv__(self, other):
        return self.__iop__(operator.idiv, other)

    def __len__(self):
        return len(self.rotations)

    def __getitem__(self, k):
        if isinstance(k, tuple):
            return Animation(
                self.rotations[k],
                self.positions[k],
                self.orients[k[1:]],
                self.offsets[k[1:]],
                self.parents[k[1:]],
                self.names[k[1:]],
                self.frametime[k[1:]])
        else:
            return Animation(
                self.rotations[k],
                self.positions[k],
                self.orients,
                self.offsets,
                self.parents,
                self.names,
                self.frametime)

    def __setitem__(self, k, v):
        if isinstance(k, tuple):
            self.rotations.__setitem__(k, v.rotations)
            self.positions.__setitem__(k, v.positions)
            self.orients.__setitem__(k[1:], v.orients)
            self.offsets.__setitem__(k[1:], v.offsets)
            self.parents.__setitem__(k[1:], v.parents)
        else:
            self.rotations.__setitem__(k, v.rotations)
            self.positions.__setitem__(k, v.positions)
            self.orients.__setitem__(k, v.orients)
            self.offsets.__setitem__(k, v.offsets)
            self.parents.__setitem__(k, v.parents)

    @property
    def shape(self):
        return (self.rotations.shape[0], self.rotations.shape[1])

    def clone(self):
        # print("self.rotations:", type(self.rotations))
        # print("self.positions:", type(self.positions))
        # print("self.orients:", type(self.orients))
        # print("self.offsets:", type(self.offsets))
        return Animation(
            self.rotations.clone(), self.positions.clone(),
            self.orients.clone(), self.offsets.detach().clone(),
            self.parents.clone(), self.names,
            self.frametime)
    
    def to(self, device):
        self.rotations = self.rotations.to(device)
        self.positions = self.positions.to(device)
        self.orients = self.orients.to(device)
        self.offsets = self.offsets.to(device)
        return self 

    def repeat(self, *args, **kw):
        return Animation(
            self.rotations.repeat(*args, **kw),
            self.positions.repeat(*args, **kw),
            self.orients, self.offsets, self.parents, self.frametime, self.names)

    def ravel(self):
        return torch.cat([
            torch.log(self.rotations).flatten(),
            self.positions.flatten(),
            torch.log(self.orients).flatten(),
            self.offsets.flatten()])
    
    # def ravel(self):
    #     return np.hstack([
    #         self.rotations.log().ravel(),
    #         self.positions.ravel(),
    #         self.orients.log().ravel(),
    #         self.offsets.ravel()])

    @classmethod
    def unravel(cls, anim, shape, parents):
        nf, nj = shape
        rotations = anim[nf * nj * 0:nf * nj * 3]
        positions = anim[nf * nj * 3:nf * nj * 6]
        orients = anim[nf * nj * 6 + nj * 0:nf * nj * 6 + nj * 3]
        offsets = anim[nf * nj * 6 + nj * 3:nf * nj * 6 + nj * 6]
        return cls(
            Quaternions.exp(rotations), positions,
            Quaternions.exp(orients), offsets,
            parents.copy())


# local transformation matrices
def transforms_local(anim, ):
    """
    Computes Animation Local Transforms

    As well as a number of other uses this can
    be used to compute global joint transforms,
    which in turn can be used to compete global
    joint positions

    Parameters
    ----------

    anim : Animation
        Input animation

    Returns
    -------

    transforms : (F, J, 4, 4) ndarray

        For each frame F, joint local
        transforms for each joint J
    """
    
    transforms = anim.rotations.transforms()
    # print("transforms:", transforms.shape)
    transforms[:, :, 0:3, 3] = anim.positions
    transforms[:, :, 3:4, 3] = 1.0
    return transforms


# def transforms_inv(ts):
#     fts = ts.reshape(-1, 4, 4)
#     fts = np.array(list(map(lambda x: np.linalg.inv(x), fts)))
#     return fts.reshape(ts.shape)

def transforms_inv(ts):
    fts = ts.reshape(-1, 4, 4)
    fts = torch.inverse(fts)  # 批量计算矩阵逆
    return fts.reshape(ts.shape)


def transforms_blank(anim, device, dtype):
    """
    Blank Transforms

    Parameters
    ----------

    anim : Animation
        Input animation

    Returns
    -------

    transforms : (F, J, 4, 4) ndarray
        Array of identity transforms for
        each frame F and joint J
    """

    ts = torch.zeros(anim.shape + (4, 4), device = device, dtype = dtype)
    ts[:, :, 0, 0] = 1.0;
    ts[:, :, 1, 1] = 1.0;
    ts[:, :, 2, 2] = 1.0;
    ts[:, :, 3, 3] = 1.0;
    return ts

def transforms_multiply(t0s, t1s):
    """
    Transforms Multiply

    Multiplies two arrays of animation transforms

    Parameters
    ----------

    t0s, t1s : (F, J, 4, 4) ndarray
        Two arrays of transforms
        for each frame F and each
        joint J

    Returns
    -------

    transforms : (F, J, 4, 4) ndarray
        Array of transforms for each
        frame F and joint J multiplied
        together
    """
    # return ut.matrix_multiply(t0s, t1s)
    return torch.matmul(t0s, t1s)


# 使用 torch.compile 编译整个函数或代码块
# @torch.compile
def compute_globals(globals, locals, anim):
    for i in range(1, anim.shape[1]):
        globals[:, i] = torch.matmul(globals[:, anim.parents[i]], locals[:, i])
    return globals

# global transformation matrices
def transforms_global(anim):
    """
    Global Animation Transforms

    This relies on joint ordering
    being incremental. That means a joint
    J1 must not be a ancestor of J0 if
    J0 appears before J1 in the joint
    ordering.

    Parameters
    ----------
    
    anim : Animation
        Input animation
        
    Returns
    ------
    
    transforms : (F, J, 4, 4) ndarray
        Array of global transforms for
        each frame F and joint J
    """
    locals = transforms_local(anim)
    globals = transforms_blank(anim, device = locals.device, dtype = locals.dtype)
    globals[:, 0] = locals[:, 0]

    globals = compute_globals(globals, locals, anim)
    # for i in range(1, anim.shape[1]):
    #     globals[:, i] = torch.matmul(globals[:, anim.parents[i]], locals[:, i])
    return globals

# !!! useful!
def positions_global(anim):
    """
    Global Joint Positions

    Given an animation compute the global joint
    positions at at every frame

    Parameters
    ----------

    anim : Animation
        Input animation

    Returns
    -------

    positions : (F, J, 3) ndarray
        Positions for every frame F
        and joint position J
    """

    # get the last column -- corresponding to the coordinates
    positions = transforms_global(anim)[:, :, :, 3]
    return positions[:, :, :3] / positions[:, :, 3, None]

""" Rotations """


def rotations_global(anim):
    """
    Global Animation Rotations

    This relies on joint ordering
    being incremental. That means a joint
    J1 must not be a ancestor of J0 if
    J0 appears before J1 in the joint
    ordering.

    Parameters
    ----------

    anim : Animation
        Input animation

    Returns
    -------

    points : (F, J) Quaternions
        global rotations for every frame F
        and joint J
    """
    locals = anim.rotations
    globals = Quaternions.id(anim.shape)

    globals[:, 0] = locals[:, 0]

    # 最耗时的部分，根据global
    for i in range(1, anim.shape[1]):
        globals[:, i] = globals[:, anim.parents[i]] * locals[:, i]

    return globals


def rotations_parents_global(anim):
    rotations = rotations_global(anim)
    rotations = rotations[:, anim.parents]
    rotations[:, 0] = Quaternions.id(len(anim))
    return rotations

""" Offsets & Orients """


def orients_global(anim):
    locals = anim.orients
    globals = Quaternions.id(anim.shape[1])

    globals[:, 0] = locals[:, 0]

    for i in range(1, anim.shape[1]):
        globals[:, i] = globals[:, anim.parents[i]] * locals[:, i]

    return globals


# def offsets_transforms_local(anim):
#     transforms = anim.orients[np.newaxis].transforms()
#     transforms = np.concatenate([transforms, np.zeros(transforms.shape[:2] + (3, 1))], axis=-1)
#     transforms = np.concatenate([transforms, np.zeros(transforms.shape[:2] + (1, 4))], axis=-2)
#     transforms[:, :, 0:3, 3] = anim.offsets[np.newaxis]
#     transforms[:, :, 3:4, 3] = 1.0
#     return transforms


def offsets_transforms_local(anim):
    transforms = anim.orients.unsqueeze(0).transforms()
    transforms = torch.cat([transforms, torch.zeros(transforms.shape[:2] + (3, 1), device=transforms.device, dtype=transforms.dtype )], dim=-1)
    transforms = torch.cat([transforms, torch.zeros(transforms.shape[:2] + (1, 4), device=transforms.device, dtype=transforms.dtype)], dim=-2)
    transforms[:, :, 0:3, 3] = anim.offsets.unsqueeze(0)
    transforms[:, :, 3:4, 3] = 1.0
    return transforms


def offsets_transforms_global(anim):
    locals = offsets_transforms_local(anim)
    globals = transforms_blank(anim)

    globals[:, 0] = locals[:, 0]

    for i in range(1, anim.shape[1]):
        globals[:, i] = transforms_multiply(globals[:, anim.parents[i]], locals[:, i])

    return globals


def offsets_global(anim):
    offsets = offsets_transforms_global(anim)[:, :, :, 3]
    return offsets[0, :, :3] / offsets[0, :, 3, None]


""" Lengths """


# def offset_lengths(anim):
#     return np.sum(anim.offsets[1:] ** 2.0, axis=1) ** 0.5

def offset_lengths(anim):
    return torch.sum(anim.offsets[1:] ** 2.0, dim=1) ** 0.5

def position_lengths(anim):
    return torch.sum(anim.positions[:, 1:] ** 2.0, dim=2) ** 0.5


""" Skinning """


def skin(anim, rest, weights, mesh, maxjoints=4):
    full_transforms = transforms_multiply(
        transforms_global(anim),
        transforms_inv(transforms_global(rest[0:1])))

    weightids = torch.argsort(-weights, dim=1)[:, :maxjoints]
    
    weightvls = torch.gather(weights, 1, weightids)
    weightvls = weightvls / weightvls.sum(dim=1, keepdim=True)
    

    # 使用torch.cat替代np.hstack，添加齐次坐标
    verts = torch.cat([mesh, torch.ones((len(mesh), 1))], dim=1)
    
    # 使用unsqueeze替代np.newaxis进行维度扩展
    verts = verts.unsqueeze(0).unsqueeze(2).unsqueeze(4)
    
    # 矩阵乘法运算（假设transforms_multiply已适配PyTorch）
    verts = transforms_multiply(full_transforms[:, weightids], verts)
    
    verts = (verts[:, :, :, :3] / verts[:, :, :, 3:4])[:, :, :, :, 0]

    return torch.sum(weightvls.unsqueeze(0).unsqueeze(-1) * verts, dim=2)


# def skin(anim, rest, weights, mesh, maxjoints=4):
#     full_transforms = transforms_multiply(
#         transforms_global(anim),
#         transforms_inv(transforms_global(rest[0:1])))

#     weightids = np.argsort(-weights, axis=1)[:, :maxjoints]
#     weightvls = np.array(list(map(lambda w, i: w[i], weights, weightids)))
#     weightvls = weightvls / weightvls.sum(axis=1)[..., np.newaxis]

#     verts = np.hstack([mesh, np.ones((len(mesh), 1))])
#     verts = verts[np.newaxis, :, np.newaxis, :, np.newaxis]
#     verts = transforms_multiply(full_transforms[:, weightids], verts)
#     verts = (verts[:, :, :, :3] / verts[:, :, :, 3:4])[:, :, :, :, 0]

#     return np.sum(weightvls[np.newaxis, :, :, np.newaxis] * verts, axis=2)
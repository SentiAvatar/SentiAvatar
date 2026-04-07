import torch
import torch.nn.functional as F
from typing import Union, Tuple
import numpy as np 
import time 

@torch.compile
def inner_from_transforms(ts):
    d0, d1, d2 = ts[..., 0, 0], ts[..., 1, 1], ts[..., 2, 2]

    q0 = (d0 + d1 + d2 + 1.0) / 4.0
    q1 = (d0 - d1 - d2 + 1.0) / 4.0
    q2 = (-d0 + d1 - d2 + 1.0) / 4.0
    q3 = (-d0 - d1 + d2 + 1.0) / 4.0

    q0 = torch.sqrt(torch.clamp(q0, min=0))
    q1 = torch.sqrt(torch.clamp(q1, min=0))
    q2 = torch.sqrt(torch.clamp(q2, min=0))
    q3 = torch.sqrt(torch.clamp(q3, min=0))
    # print(f"After calculation - q0: {q0.shape}, q1: {q1.shape}, q2: {q2.shape}, q3: {q3.shape}") # 检查计算后的形状
    
    c0 = (q0 >= q1) & (q0 >= q2) & (q0 >= q3)
    indc0 = torch.where(c0)
    
    c1 = (q1 >= q0) & (q1 >= q2) & (q1 >= q3)
    indc1 = torch.where(c1)
    
    c2 = (q2 >= q0) & (q2 >= q1) & (q2 >= q3)
    indc2 = torch.where(c2)
    
    c3 = (q3 >= q0) & (q3 >= q1) & (q3 >= q2)
    indc3 = torch.where(c3)
    
    q1[indc0[0], indc0[1]] *= torch.sign(ts[indc0[0], indc0[1], 2, 1] - ts[indc0[0], indc0[1], 1, 2])
    q2[indc0[0], indc0[1]] *= torch.sign(ts[indc0[0], indc0[1], 0, 2] - ts[indc0[0], indc0[1], 2, 0])
    q3[indc0[0], indc0[1]] *= torch.sign(ts[indc0[0], indc0[1], 1, 0] - ts[indc0[0], indc0[1], 0, 1])

    q0[indc1[0], indc1[1],] *= torch.sign(ts[indc1[0], indc1[1], 2, 1] - ts[indc1[0], indc1[1], 1, 2])
    q2[indc1[0], indc1[1],] *= torch.sign(ts[indc1[0], indc1[1], 1, 0] + ts[indc1[0], indc1[1], 0, 1])
    q3[indc1[0], indc1[1],] *= torch.sign(ts[indc1[0], indc1[1], 0, 2] + ts[indc1[0], indc1[1], 2, 0])

    q0[indc2[0], indc2[1]] *= torch.sign(ts[indc2[0], indc2[1], 0, 2] - ts[indc2[0], indc2[1], 2, 0])
    q1[indc2[0], indc2[1]] *= torch.sign(ts[indc2[0], indc2[1], 1, 0] + ts[indc2[0], indc2[1], 0, 1])
    q3[indc2[0], indc2[1]] *= torch.sign(ts[indc2[0], indc2[1], 2, 1] + ts[indc2[0], indc2[1], 1, 2])

    q0[indc3[0], indc3[1]] *= torch.sign(ts[indc3[0], indc3[1], 1, 0] - ts[indc3[0], indc3[1], 0, 1])
    q1[indc3[0], indc3[1]] *= torch.sign(ts[indc3[0], indc3[1], 2, 0] + ts[indc3[0], indc3[1], 0, 2])
    q2[indc3[0], indc3[1]] *= torch.sign(ts[indc3[0], indc3[1], 2, 1] + ts[indc3[0], indc3[1], 1, 2])
    return q0, q1, q2, q3 


class Quaternions:
    """
    Quaternions is a wrapper around a numpy ndarray
    that allows it to act as if it were an narray of
    a quater data type.

    Therefore addition, subtraction, multiplication,
    division, negation, absolute, are all defined
    in terms of quater operations such as quater
    multiplication.

    This allows for much neater code and many routines
    which conceptually do the same thing to be written
    in the same way for point data and for rotation data.

    The Quaternions class has been desgined such that it
    should support broadcasting and slicing in all of the
    usual ways.
    """

    def __init__(self, qs):
        if isinstance(qs, np.ndarray):
            qs = torch.tensor(qs)
            
        if isinstance(qs, torch.Tensor):
            if len(qs.shape) == 1:
                qs = qs.unsqueeze(0)
            self.qs = qs
            self.device = self.qs.device 
            self.m = torch.zeros(self.shape + (4, 4), device=self.qs.device, dtype=self.qs.dtype)
            return

        if isinstance(qs, Quaternions):
            self.qs = qs
            self.device = self.qs.device 
            self.m = torch.zeros(self.shape + (4, 4), device=self.qs.device, dtype=self.qs.dtype)
            return
        
        raise TypeError('Quaternions must be constructed from iterable, numpy array, or Quaternions, not %s' % type(qs))

    def to(self, device):
        self.qs = self.qs.to(device)
        self.device = self.qs.device 
        self.m = self.m.to(device)
        return self 
        
    def __str__(self):
        return "Quaternions(" + str(self.qs) + ")"

    def __repr__(self):
        return "Quaternions(" + repr(self.qs) + ")"

    """ Helper Methods for Broadcasting and Data extraction """

    @classmethod
    def _broadcast(cls, sqs, oqs, scalar=False):
        if isinstance(oqs, float): 
            return sqs, oqs * torch.ones(sqs.shape[:-1], device=sqs.device, dtype=sqs.dtype)

        ss = torch.tensor(sqs.shape) if not scalar else torch.tensor(sqs.shape[:-1])
        os = torch.tensor(oqs.shape)

        if len(ss) != len(os):
            raise TypeError('Quaternions cannot broadcast together shapes %s and %s' % (sqs.shape, oqs.shape))

        if torch.all(ss == os): 
            return sqs, oqs

        if not torch.all((ss == os) | (os == torch.ones(len(os))) | (ss == torch.ones(len(ss)))):
            raise TypeError('Quaternions cannot broadcast together shapes %s and %s' % (sqs.shape, oqs.shape))

        sqsn, oqsn = sqs.clone(), oqs.clone()

        for a in torch.where(ss == 1)[0]: 
            sqsn = sqsn.repeat(*[os[a].item() if i == a else 1 for i in range(len(ss))])
        for a in torch.where(os == 1)[0]: 
            oqsn = oqsn.repeat(*[ss[a].item() if i == a else 1 for i in range(len(os))])

        return sqsn, oqsn

    '''
    def _broadcast(cls, sqs, oqs, scalar=False):
        if isinstance(oqs, float): return sqs, oqs * np.ones(sqs.shape[:-1])

        ss = np.array(sqs.shape) if not scalar else np.array(sqs.shape[:-1])
        os = np.array(oqs.shape)

        if len(ss) != len(os):
            raise TypeError('Quaternions cannot broadcast together shapes %s and %s' % (sqs.shape, oqs.shape))

        if np.all(ss == os): return sqs, oqs

        if not np.all((ss == os) | (os == np.ones(len(os))) | (ss == np.ones(len(ss)))):
            raise TypeError('Quaternions cannot broadcast together shapes %s and %s' % (sqs.shape, oqs.shape))

        sqsn, oqsn = sqs.copy(), oqs.copy()

        for a in np.where(ss == 1)[0]: sqsn = sqsn.repeat(os[a], axis=a)
        for a in np.where(os == 1)[0]: oqsn = oqsn.repeat(ss[a], axis=a)

        return sqsn, oqsn
    
    '''
    """ Adding Quaterions is just Defined as Multiplication """

    def __add__(self, other):
        return self * other

    def __sub__(self, other):
        return self / other

    """ Quaterion Multiplication """
    
    def __inner_mul__(self, sqs, oqs, qs):
        q0 = sqs[..., 0];
        q1 = sqs[..., 1];
        q2 = sqs[..., 2];
        q3 = sqs[..., 3];
        r0 = oqs[..., 0];
        r1 = oqs[..., 1];
        r2 = oqs[..., 2];
        r3 = oqs[..., 3];
        
        qs[..., 0] = r0 * q0 - r1 * q1 - r2 * q2 - r3 * q3
        qs[..., 1] = r0 * q1 + r1 * q0 - r2 * q3 + r3 * q2
        qs[..., 2] = r0 * q2 + r1 * q3 + r2 * q0 - r3 * q1
        qs[..., 3] = r0 * q3 - r1 * q2 + r2 * q1 + r3 * q0
        
        # qs[..., 0] = oqs[..., 0] * sqs[..., 0] - oqs[..., 1] * sqs[..., 1] - oqs[..., 2] * sqs[..., 2] - oqs[..., 3] * sqs[..., 3]
        # qs[..., 1] = oqs[..., 0] * sqs[..., 1] + oqs[..., 1] * sqs[..., 0] - oqs[..., 2] * sqs[..., 3] + oqs[..., 3] * sqs[..., 2]
        # qs[..., 2] = oqs[..., 0] * sqs[..., 2] + oqs[..., 1] * sqs[..., 3] + oqs[..., 2] * sqs[..., 0] - oqs[..., 3] * sqs[..., 1]
        # qs[..., 3] = oqs[..., 0] * sqs[..., 3] - oqs[..., 1] * sqs[..., 2] + oqs[..., 2] * sqs[..., 1] + oqs[..., 3] * sqs[..., 0]
    
        return qs
    
    # @torch.compile
    def __inner_mul_compile__(self, sqs, oqs, qs):
        q0 = sqs[..., 0];
        q1 = sqs[..., 1];
        q2 = sqs[..., 2];
        q3 = sqs[..., 3];
        r0 = oqs[..., 0];
        r1 = oqs[..., 1];
        r2 = oqs[..., 2];
        r3 = oqs[..., 3];
        
        qs[..., 0] = r0 * q0 - r1 * q1 - r2 * q2 - r3 * q3
        qs[..., 1] = r0 * q1 + r1 * q0 - r2 * q3 + r3 * q2
        qs[..., 2] = r0 * q2 + r1 * q3 + r2 * q0 - r3 * q1
        qs[..., 3] = r0 * q3 - r1 * q2 + r2 * q1 + r3 * q0
        
        # qs[..., 0] = oqs[..., 0] * sqs[..., 0] - oqs[..., 1] * sqs[..., 1] - oqs[..., 2] * sqs[..., 2] - oqs[..., 3] * sqs[..., 3]
        # qs[..., 1] = oqs[..., 0] * sqs[..., 1] + oqs[..., 1] * sqs[..., 0] - oqs[..., 2] * sqs[..., 3] + oqs[..., 3] * sqs[..., 2]
        # qs[..., 2] = oqs[..., 0] * sqs[..., 2] + oqs[..., 1] * sqs[..., 3] + oqs[..., 2] * sqs[..., 0] - oqs[..., 3] * sqs[..., 1]
        # qs[..., 3] = oqs[..., 0] * sqs[..., 3] - oqs[..., 1] * sqs[..., 2] + oqs[..., 2] * sqs[..., 1] + oqs[..., 3] * sqs[..., 0]
    
        return qs
    
    def __mul__(self, other):
        """
        Quaternion multiplication has three main methods.

        When multiplying a Quaternions array by Quaternions
        normal quater multiplication is performed.

        When multiplying a Quaternions array by a vector
        array of the same shape, where the last axis is 3,
        it is assumed to be a Quaternion by 3D-Vector
        multiplication and the 3D-Vectors are rotated
        in space by the Quaternions.

        When multipplying a Quaternions array by a scalar
        or vector of different shape it is assumed to be
        a Quaternions by Scalars multiplication and the
        Quaternions are scaled using Slerp and the identity
        quaternions.
        """

        """ If Quaternions type do Quaternions * Quaternions """
        if isinstance(other, Quaternions):
            
            sqs, oqs = Quaternions._broadcast(self.qs, other.qs)
            time_1 = time.time()
            qs = torch.zeros(sqs.shape, device=sqs.device, dtype=sqs.dtype)
            if len(sqs.shape) == 3 and sqs.shape[1] == 1 and sqs.shape[2] == 4:
                qs = self.__inner_mul_compile__(sqs, oqs, qs)
            else:
                qs = self.__inner_mul__(sqs, oqs, qs)
            qs = torch.where(torch.isnan(qs), torch.full_like(qs, 0),  qs)
            '''
            q0 = sqs[..., 0];
            q1 = sqs[..., 1];
            q2 = sqs[..., 2];
            q3 = sqs[..., 3];
            r0 = oqs[..., 0];
            r1 = oqs[..., 1];
            r2 = oqs[..., 2];
            r3 = oqs[..., 3];
            qs = torch.empty(sqs.shape, device=sqs.device, dtype=sqs.dtype)            
            qs[..., 0] = r0 * q0 - r1 * q1 - r2 * q2 - r3 * q3
            qs[..., 1] = r0 * q1 + r1 * q0 - r2 * q3 + r3 * q2
            qs[..., 2] = r0 * q2 + r1 * q3 + r2 * q0 - r3 * q1
            qs[..., 3] = r0 * q3 - r1 * q2 + r2 * q1 + r3 * q0
            '''
            
            time_2 = time.time()
            # print((time_2 - time_1) * 100)
            return Quaternions(qs)

        """ If array type do Quaternions * Vectors """
        if isinstance(other, torch.Tensor) and other.shape[-1] == 3:
            vs = Quaternions(torch.cat([torch.zeros(other.shape[:-1] + (1,), device=other.device, dtype=other.dtype), other], dim=-1))

            return (self * (vs * -self)).imaginaries

                
        """ If float do Quaternions * Scalars """
        if isinstance(other, torch.Tensor) or isinstance(other, float):
            return Quaternions.slerp(Quaternions.id_like(self), self, other)

        raise TypeError('Cannot multiply/add Quaternions with type %s' % str(type(other)))

    def __div__(self, other):
        """
        When a Quaternion type is supplied, division is defined
        as multiplication by the inverse of that Quaternion.

        When a scalar or vector is supplied it is defined
        as multiplicaion of one over the supplied value.
        Essentially a scaling.
        """

        if isinstance(other, Quaternions): return self * (-other)
        if isinstance(other, torch.Tensor): return self * (1.0 / other)
        if isinstance(other, float): return self * (1.0 / other)
        raise TypeError('Cannot divide/subtract Quaternions with type %s' + str(type(other)))

    def __eq__(self, other):
        return torch.all(self.qs == other.qs)

    def __ne__(self, other):
        return not torch.all(self.qs == other.qs)

    def __neg__(self):
        """ Invert Quaternions """
        return Quaternions(self.qs * torch.tensor([1, -1, -1, -1], device=self.qs.device, dtype=self.qs.dtype))

    def __abs__(self):
        """ Unify Quaternions To Single Pole """
        qabs = self.normalized().clone()
        top = torch.sum((qabs.qs) * torch.tensor([1, 0, 0, 0], device=self.qs.device, dtype=self.qs.dtype), dim=-1)
        bot = torch.sum((-qabs.qs) * torch.tensor([1, 0, 0, 0], device=self.qs.device, dtype=self.qs.dtype), dim=-1)
        qabs.qs[top < bot] = -qabs.qs[top < bot]
        return qabs

    def __iter__(self):
        return iter(self.qs)

    def __len__(self):
        return len(self.qs)

    def __getitem__(self, k):
        return Quaternions(self.qs[k])

    def __setitem__(self, k, v):
        self.qs[k] = v.qs

    @property
    def lengths(self):
        return torch.sum(self.qs ** 2.0, dim=-1) ** 0.5

    @property
    def reals(self):
        return self.qs[..., 0]

    @property
    def imaginaries(self):
        return self.qs[..., 1:4]

    @property
    def shape(self):
        return self.qs.shape[:-1]

    def repeat(self, n, **kwargs):
        return Quaternions(self.qs.repeat(n, **kwargs))

    def normalized(self):
        lengths = self.lengths
        # Avoid division by zero
        lengths = torch.where(lengths == 0, torch.tensor(1e-10, device=lengths.device, dtype=lengths.dtype), lengths)
        return Quaternions(self.qs / lengths.unsqueeze(-1))


    def log(self):
        norm = abs(self.normalized())
        imgs = norm.imaginaries
        lens = torch.sqrt(torch.sum(imgs ** 2, dim=-1))
        lens = torch.atan2(lens, norm.reals) / (lens + 1e-10)
        return imgs * lens.unsqueeze(-1)
    
    def constrained(self, axis):

        rl = self.reals
        im = torch.sum(axis * self.imaginaries, dim=-1)

        t1 = -2 * torch.atan2(rl, im) + torch.pi
        t2 = -2 * torch.atan2(rl, im) - torch.pi

        top = Quaternions.exp(axis.unsqueeze(0) * (t1.unsqueeze(-1) / 2.0))
        bot = Quaternions.exp(axis.unsqueeze(0) * (t2.unsqueeze(-1) / 2.0))
        img = self.dot(top) > self.dot(bot)

        ret = top.copy()
        ret[img] = top[img]
        ret[~img] = bot[~img]
        return ret

    def constrained_x(self):
        return self.constrained(torch.tensor([1, 0, 0], device=self.qs.device, dtype=self.qs.dtype))

    def constrained_y(self):
        return self.constrained(torch.tensor([0, 1, 0], device=self.qs.device, dtype=self.qs.dtype))

    def constrained_z(self):
        return self.constrained(torch.tensor([0, 0, 1], device=self.qs.device, dtype=self.qs.dtype))

    def dot(self, q):
        return torch.sum(self.qs * q.qs, dim=-1)

    def clone(self):
        return Quaternions(self.qs.clone())

    def reshape(self, s):
        self.qs = self.qs.reshape(s)
        return self

    def interpolate(self, ws):
        return Quaternions.exp(torch.average(abs(self).log, dim=0, weights=ws))

    def euler(self, order='xyz'):  # fix the wrong convert, this should convert to world euler by default.

        q = self.normalized().qs
        q0 = q[..., 0]
        q1 = q[..., 1]
        q2 = q[..., 2]
        q3 = q[..., 3]
        es = torch.zeros(self.shape + (3,), device=self.qs.device, dtype=self.qs.dtype)

        if order == 'xyz':
            es[..., 0] = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
            es[..., 1] = torch.asin((2 * (q0 * q2 - q3 * q1)).clip(-1, 1))
            es[..., 2] = torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
        elif order == 'yzx':
            es[..., 0] = torch.atan2(2 * (q1 * q0 - q2 * q3), -q1 * q1 + q2 * q2 - q3 * q3 + q0 * q0)
            es[..., 1] = torch.atan2(2 * (q2 * q0 - q1 * q3), q1 * q1 - q2 * q2 - q3 * q3 + q0 * q0)
            es[..., 2] = torch.asin((2 * (q1 * q2 + q3 * q0)).clip(-1, 1))
        else:
            raise NotImplementedError('Cannot convert from ordering %s' % order)

        """

        # These conversion don't appear to work correctly for Maya.
        # http://bediyap.com/programming/convert-quaternion-to-euler-rotations/

        if   order == 'xyz':
            es[fa + (0,)] = np.arctan2(2 * (q0 * q3 - q1 * q2), q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3)
            es[fa + (1,)] = np.arcsin((2 * (q1 * q3 + q0 * q2)).clip(-1,1))
            es[fa + (2,)] = np.arctan2(2 * (q0 * q1 - q2 * q3), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3)
        elif order == 'yzx':
            es[fa + (0,)] = np.arctan2(2 * (q0 * q1 - q2 * q3), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3)
            es[fa + (1,)] = np.arcsin((2 * (q1 * q2 + q0 * q3)).clip(-1,1))
            es[fa + (2,)] = np.arctan2(2 * (q0 * q2 - q1 * q3), q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3)
        elif order == 'zxy':
            es[fa + (0,)] = np.arctan2(2 * (q0 * q2 - q1 * q3), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3)
            es[fa + (1,)] = np.arcsin((2 * (q0 * q1 + q2 * q3)).clip(-1,1))
            es[fa + (2,)] = np.arctan2(2 * (q0 * q3 - q1 * q2), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3) 
        elif order == 'xzy':
            es[fa + (0,)] = np.arctan2(2 * (q0 * q2 + q1 * q3), q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3)
            es[fa + (1,)] = np.arcsin((2 * (q0 * q3 - q1 * q2)).clip(-1,1))
            es[fa + (2,)] = np.arctan2(2 * (q0 * q1 + q2 * q3), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3)
        elif order == 'yxz':
            es[fa + (0,)] = np.arctan2(2 * (q1 * q2 + q0 * q3), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3)
            es[fa + (1,)] = np.arcsin((2 * (q0 * q1 - q2 * q3)).clip(-1,1))
            es[fa + (2,)] = np.arctan2(2 * (q1 * q3 + q0 * q2), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3)
        elif order == 'zyx':
            es[fa + (0,)] = np.arctan2(2 * (q0 * q1 + q2 * q3), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3)
            es[fa + (1,)] = np.arcsin((2 * (q0 * q2 - q1 * q3)).clip(-1,1))
            es[fa + (2,)] = np.arctan2(2 * (q0 * q3 + q1 * q2), q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3)
        else:
            raise KeyError('Unknown ordering %s' % order)

        """

        # https://github.com/ehsan/ogre/blob/master/OgreMain/src/OgreMatrix3.cpp
        # Use this class and convert from matrix

        return es

    def average(self):

        if len(self.shape) == 1:

            system = torch.einsum('bi,bj->ij', self.qs, self.qs)
            w, v = torch.linalg.eigh(system)
            qiT_dot_qref = torch.einsum('bi,ij->bj', self.qs, v)
            return Quaternions(v[:, torch.argmin((1. - qiT_dot_qref ** 2).sum(dim=0))])

        else:

            raise NotImplementedError('Cannot average multi-dimensionsal Quaternions')

    def angle_axis(self):

        norm = self.normalized()
        s = torch.sqrt(1 - (norm.reals ** 2.0))
        s[s == 0] = 0.001

        angles = 2.0 * torch.acos(norm.reals)
        axis = norm.imaginaries / s.unsqueeze(-1)

        return angles, axis

    # @torch.compile
    def transforms(self):

        qw = self.qs[..., 0]
        qx = self.qs[..., 1]
        qy = self.qs[..., 2]
        qz = self.qs[..., 3]

        x2 = qx + qx;
        y2 = qy + qy;
        z2 = qz + qz;
        xx = qx * x2;
        yy = qy * y2;
        wx = qw * x2;
        xy = qx * y2;
        yz = qy * z2;
        wy = qw * y2;
        xz = qx * z2;
        zz = qz * z2;
        wz = qw * z2;
        
        self.m[..., 0, 0] = 1.0 - (yy + zz)
        self.m[..., 0, 1] = xy - wz
        self.m[..., 0, 2] = xz + wy
        self.m[..., 1, 0] = xy + wz
        self.m[..., 1, 1] = 1.0 - (xx + zz)
        self.m[..., 1, 2] = yz - wx
        self.m[..., 2, 0] = xz - wy
        self.m[..., 2, 1] = yz + wx
        self.m[..., 2, 2] = 1.0 - (xx + yy)
        # print("m device:", m.device)
        return self.m.clone()
    
    # @torch.compile
    def transforms_by_joint(self, joint):

        qw = self.qs[..., joint: joint + 1, 0]
        qx = self.qs[..., joint: joint + 1, 1]
        qy = self.qs[..., joint: joint + 1, 2]
        qz = self.qs[..., joint: joint + 1, 3]

        x2 = qx + qx;
        y2 = qy + qy;
        z2 = qz + qz;
        xx = qx * x2;
        yy = qy * y2;
        wx = qw * x2;
        xy = qx * y2;
        yz = qy * z2;
        wy = qw * y2;
        xz = qx * z2;
        zz = qz * z2;
        wz = qw * z2;
        
        self.m[..., joint: joint + 1, 0, 0] = 1.0 - (yy + zz)
        self.m[..., joint: joint + 1, 0, 1] = xy - wz
        self.m[..., joint: joint + 1, 0, 2] = xz + wy
        self.m[..., joint: joint + 1, 1, 0] = xy + wz
        self.m[..., joint: joint + 1, 1, 1] = 1.0 - (xx + zz)
        self.m[..., joint: joint + 1, 1, 2] = yz - wx
        self.m[..., joint: joint + 1, 2, 0] = xz - wy
        self.m[..., joint: joint + 1, 2, 1] = yz + wx
        self.m[..., joint: joint + 1, 2, 2] = 1.0 - (xx + yy)
        # print("m device:", m.device)
        return self.m.clone()
    
    def ravel(self):
        return self.qs.ravel()

    @classmethod
    def id(cls, n):

        if isinstance(n, tuple):
            qs = torch.zeros(n + (4,))
            qs[..., 0] = 1.0
            return Quaternions(qs)

        if isinstance(n, int):
            qs = torch.zeros((n, 4))
            qs[:, 0] = 1.0
            return Quaternions(qs)

        raise TypeError('Cannot Construct Quaternion from %s type' % str(type(n)))

    @classmethod
    def id_like(cls, a):
        qs = torch.zeros(a.shape + (4,),  device=a.qs.device, dtype=a.qs.dtype)
        qs[..., 0] = 1.0
        return Quaternions(qs)

    @classmethod
    def exp(cls, ws):

        ts = torch.sum(ws ** 2.0, axis=-1) ** 0.5
        ts[ts == 0] = 0.001
        ls = torch.sin(ts) / ts

        qs = torch.zeros(ws.shape[:-1] + (4,))
        qs[..., 0] = torch.cos(ts)
        qs[..., 1] = ws[..., 0] * ls
        qs[..., 2] = ws[..., 1] * ls
        qs[..., 3] = ws[..., 2] * ls

        return Quaternions(qs).normalized()

    @classmethod
    def slerp(cls, q0s, q1s, a):

        fst, snd = cls._broadcast(q0s.qs, q1s.qs)
        fst, a = cls._broadcast(fst, a, scalar=True)
        snd, a = cls._broadcast(snd, a, scalar=True)

        dot  = torch.sum(fst * snd, axis=-1)

        neg = dot  < 0.0
        dot [neg] = -dot [neg]
        snd[neg] = -snd[neg]

        amount0 = torch.zeros(a.shape)
        amount1 = torch.zeros(a.shape)

        linear = (1.0 - dot ) < 0.01
        omegas = torch.acos(dot[~linear])
        sinoms = torch.sin(omegas)

        amount0[linear] = 1.0 - a[linear]
        amount1[linear] = a[linear]
        amount0[~linear] = torch.sin((1.0 - a[~linear]) * omegas) / sinoms
        amount1[~linear] = torch.sin(a[~linear] * omegas) / sinoms

        return Quaternions(
            amount0.unsqueeze(-1) * fst +
            amount1.unsqueeze(-1) * snd)

    @classmethod
    def between(cls, v0s, v1s):
        a = torch.cross(v0s, v1s)
        w = torch.sqrt((v0s ** 2).sum(axis=-1) * (v1s ** 2).sum(axis=-1)) + (v0s * v1s).sum(axis=-1)
        return Quaternions(torch.cat([w.unsqueeze(-1), a], dim=-1)).normalized()

    @classmethod
    # @torch.compile
    def from_angle_axis(cls, angles, axis):
        axis = axis / (torch.sqrt(torch.sum(axis ** 2, dim=-1)) + 1e-10).unsqueeze(-1)
        sines = torch.sin(angles / 2.0).unsqueeze(-1)
        cosines = torch.cos(angles / 2.0).unsqueeze(-1)
        return Quaternions(torch.cat([cosines, axis * sines], dim=-1))

    @classmethod
    def from_euler(cls, es, order='xyz', world=False):

        axis = {
            'x': torch.tensor([1, 0, 0], device=es.device, dtype=es.dtype),
            'y': torch.tensor([0, 1, 0], device=es.device, dtype=es.dtype),
            'z': torch.tensor([0, 0, 1], device=es.device, dtype=es.dtype),
        }

        q0s = Quaternions.from_angle_axis(es[..., 0], axis[order[0]])
        q1s = Quaternions.from_angle_axis(es[..., 1], axis[order[1]])
        q2s = Quaternions.from_angle_axis(es[..., 2], axis[order[2]])

        return (q2s * (q1s * q0s)) if world else (q0s * (q1s * q2s))

    
        
    @classmethod
    def from_transforms(cls, ts):
        # print("cls:", cls.shape, "ts:", ts.shape)
        q0, q1, q2, q3 = inner_from_transforms(ts)
        qs = torch.zeros(ts.shape[:-2] + (4,), device=ts.device, dtype=ts.dtype)
        qs[..., 0] = q0
        qs[..., 1] = q1
        qs[..., 2] = q2
        qs[..., 3] = q3
    
        return cls(qs)
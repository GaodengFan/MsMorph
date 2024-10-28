import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal

class Encoder(nn.Module):
    def __init__(self, in_channel=1, first_out_channel=16,alpha=0.1):
        super(Encoder, self).__init__()

        c = first_out_channel
        self.conv0 = nn.Sequential(
            nn.Conv3d(in_channel,  c, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(c),
            nn.LeakyReLU(alpha),
        )

        self.conv1 = nn.Sequential(
            nn.AvgPool3d(2),
            nn.Conv3d(c, 2*c, kernel_size=3, stride=1, padding=1),#80
            nn.Conv3d(2*c, 2*c, kernel_size=3, stride=1,padding=1),
            nn.InstanceNorm3d(2*c),
            nn.LeakyReLU(0.1)
        )

        self.conv2 = nn.Sequential(
            nn.AvgPool3d(2),
            nn.Conv3d(2*c, 4*c, kernel_size=3, stride=1, padding=1),#40
            nn.Conv3d(4 * c, 4 * c, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(4 * c),
            nn.LeakyReLU(0.1)
        )

        self.conv3 = nn.Sequential(
            nn.AvgPool3d(2),
            nn.Conv3d(4*c, 8*c, kernel_size=3, stride=1, padding=1),#20
            nn.Conv3d(8 * c, 8 * c, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(8 * c),
            nn.LeakyReLU(0.1)
        )
    def forward(self, x):
        out0 = self.conv0(x)  # 1
        out1 = self.conv1(out0)  # 1/2
        out2 = self.conv2(out1)  # 1/4
        out3 = self.conv3(out2)  # 1/8
        return [out0, out1, out2, out3]
class MSBlock(nn.Module):
    def __init__(self,channels,kernel_size=3, stride=1, padding=1, alpha=0.1,k=2):
        super().__init__()
        self.simweight = nn.Sequential(
            nn.Conv3d(channels,channels,3,1,1),
            nn.Sigmoid(),
        )
        self.convx=nn.Sequential(
            nn.Conv3d(2*channels, channels, kernel_size, stride, padding),
            nn.InstanceNorm3d(channels),
            nn.LeakyReLU(alpha),
            nn.Conv3d(channels, channels, kernel_size, stride, padding),
            nn.InstanceNorm3d(channels),
            nn.LeakyReLU(alpha),
        )
        self.convy = nn.Sequential(
            nn.Conv3d(2 * channels, channels, kernel_size, stride, padding),
            nn.InstanceNorm3d(channels),
            nn.LeakyReLU(alpha),
            nn.Conv3d(channels, channels, kernel_size, stride, padding),
            nn.InstanceNorm3d(channels),
            nn.LeakyReLU(alpha),
        )
        self.convz = nn.Sequential(
            nn.Conv3d(2 * channels, channels, kernel_size, stride, padding),
            nn.InstanceNorm3d(channels),
            nn.LeakyReLU(alpha),
            nn.Conv3d(channels, channels, kernel_size, stride, padding),
            nn.InstanceNorm3d(channels),
            nn.LeakyReLU(alpha),
        )
        self.defconv=nn.Conv3d(channels,1,3,1,1)
        self.defconv.weight = nn.Parameter(Normal(0, 1e-5).sample(self.defconv.weight.shape))
        self.defconv.bias = nn.Parameter(torch.zeros(self.defconv.bias.shape))

    def forward(self,fixed,moving):
        f_dy ,f_dx,f_dz= torch.gradient(fixed,dim=2)[0],torch.gradient(fixed,dim=3)[0],torch.gradient(fixed,dim=4)[0]
        m_dy ,m_dx,m_dz= torch.gradient(moving,dim=2)[0],torch.gradient(moving,dim=3)[0],torch.gradient(moving,dim=4)[0]
        y,x,z=self.simweight(f_dy-m_dy),self.simweight(f_dx-m_dx),self.simweight(f_dz-m_dz)
        y=self.defconv(self.convy(torch.cat([y*fixed,y*moving],dim=1)))
        x=self.defconv(self.convx(torch.cat([x*fixed,x*moving],dim=1)))
        z=self.defconv(self.convz(torch.cat([z*fixed,z*moving],dim=1)))
        flow=torch.cat([y,x,z],dim=1)
        return flow
class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        # grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)
class MSMorph(nn.Module):
    '''
    Siamese More Similar

    '''
    def __init__(self,imgsize=(160,192,160),in_channel=1, channels=16):
        super().__init__()
        self.encoder = Encoder(in_channel=in_channel, first_out_channel=channels)
        self.msblock1=MSBlock(channels*8)
        self.msblock2=MSBlock(channels*4)
        self.msblock3=MSBlock(channels*2)
        self.msblock4=MSBlock(channels)
        self.warp=nn.ModuleList()
        # self.diff=nn.ModuleList()
        for i in range(4):
            self.warp.append(SpatialTransformer([s // 2**(3-i) for s in imgsize]))
        self.up=nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

    def forward(self,moving,fixed):
        F4,F3,F2,F1=self.encoder(fixed)#[1, 16, 160, 192, 160],[1, 32, 80, 96, 80],[1, 64, 40, 48, 40],[1, 128, 20, 24, 20]
        M4,M3,M2,M1=self.encoder(moving)
        flow=self.msblock1(F1,M1)
        FM1=self.warp[0](M1,flow)
        q=self.msblock1(F1,FM1)
        flow=self.warp[0](flow,q)+q

        flow=self.up(2*flow)
        FM2=self.warp[1](M2,flow)
        q=self.msblock2(F2,FM2)
        flow=self.warp[1](flow,q)+q

        flow=self.up(2*flow)
        FM3=self.warp[2](M3,flow)
        q=self.msblock3(F3,FM3)
        flow=self.warp[2](flow,q)+q

        flow=self.up(2*flow)
        FM4=self.warp[3](M4,flow)
        q=self.msblock4(F4,FM4)
        flow=self.warp[3](flow,q)+q

        y_warped=self.warp[3](moving,flow)

        return y_warped,flow


if __name__ == '__main__':
    size = (1, 1, 160, 192, 160)
    model = MSMorph(size[2:])
    # print(str(model))
    A = torch.ones(size)
    B = torch.ones(size)
    out, flow = model(A, B)
    print(out.shape, flow.shape)

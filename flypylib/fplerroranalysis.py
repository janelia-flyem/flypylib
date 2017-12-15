from flypylib import fplobjdetect, fplsynapses
from scipy import ndimage
import matplotlib.pyplot as plt
import h5py
import numpy as np

class ObjPredVisualizeErrors:

    def __init__(self, im, pd_voxel, gt,
                 obj_min_dist=27, smoothing_sigma=5,
                 volume_offset=(0,0,0), buffer_sz=15, allow_mult=False,
                 print_offset=(0,0,0)):

        if(isinstance(im, str)):
            im = h5py.File(im,'r')['/main'][:]
        if(isinstance(gt, str)):
            gt = fplsynapses.load_from_json(
                gt, pd_voxel.shape, buffer_sz)

        im_max = np.max(im)
        im_min = np.min(im)
        im     = (im - im_min)/(im_max - im_min)

        self.radius   = 75
        self.im       = np.pad(im, ((0,),(self.radius,),(self.radius,)),
                               'constant')
        self.pd_voxel = pd_voxel
        self.gt       = gt

        self.pd       = fplobjdetect.voxel2obj(
            pd_voxel, obj_min_dist, smoothing_sigma,
            volume_offset, buffer_sz)

        self.pd_vs    = ndimage.filters.gaussian_filter(
            self.pd_voxel, smoothing_sigma, truncate=2.0)
        self.pd_voxel = np.pad(self.pd_voxel,
                               ((0,),(self.radius,),(self.radius,)),
                               'constant')
        self.result   = fplobjdetect.obj_pr(
            self.pd['locs'], self.gt['locs'], obj_min_dist,
            allow_mult=allow_mult)

        self.n_pd     = self.pd['conf'].size
        self.n_gt     = self.gt['conf'].size

        self.pd_match = np.sum(self.result.match, axis=1)
        self.gt_match = np.sum(self.result.match, axis=0)

        self.pd_idx   = np.argsort(self.pd['conf'] + self.pd_match)
        self.pd_curr  = max(np.sum(self.pd_match==0) - 1, 0)

        self.gt_mconf = np.argmax(self.result.match, axis=0)
        self.gt_mconf = self.pd['conf'][self.gt_mconf] * self.gt_match
        self.gt_idx   = np.argsort(self.gt_mconf)
        self.gt_curr  = max(np.sum(self.gt_match==0) - 1, 0)

        self.pd['locs'] = self.pd['locs'].astype(int)
        self.gt['locs'] = self.gt['locs'].astype(int)
        self.mode     = 0
        self.overlay  = 1
        self.z        = 0
        self.moved    = True

        self.print_offset = np.asarray(print_offset)-buffer_sz

        fig = plt.figure()
        fig.canvas.mpl_connect('key_press_event', self.keypress)
        self.update()

    def keypress(self, event):
        if(event.key=='right'):
            if self.mode==0 and self.gt_curr < self.n_gt - 1:
                self.gt_curr = self.gt_curr + 1
                self.moved = True
            if self.mode==1 and self.pd_curr < self.n_pd - 1:
                self.pd_curr = self.pd_curr + 1
                self.moved = True
        if(event.key=='left'):
            if self.mode==0 and self.gt_curr > 0:
                self.gt_curr = self.gt_curr - 1
                self.moved = True
            if self.mode==1 and self.pd_curr > 0:
                self.pd_curr = self.pd_curr - 1
                self.moved = True
        if(event.key=='up'):
            self.z = min(self.z+1, self.pd_voxel.shape[0]-1)
        if(event.key=='down'):
            self.z = max(self.z-1, 0)
        if(event.key=='m'):
            self.mode = 1 - self.mode
            self.moved = True
        if(event.key=='o'):
            self.overlay = 1 - self.overlay
        self.update()

    def update(self):
        if self.moved:
            if self.mode == 0:
                self.z = self.gt['locs'][self.gt_idx[self.gt_curr],2]

                score = self.pd_vs[
                    self.z,
                    self.gt['locs'][self.gt_idx[self.gt_curr],1],
                    self.gt['locs'][self.gt_idx[self.gt_curr],0]]
                print('[gt %d/%d conf: %.03f score: %03f (%d,%d,%d)]' %
                      (self.gt_curr+1, self.n_gt,
                       self.gt_mconf[self.gt_idx[self.gt_curr]],
                       score,
                       self.gt['locs'][self.gt_idx[self.gt_curr],0] +
                       self.print_offset[0],
                       self.gt['locs'][self.gt_idx[self.gt_curr],1] +
                       self.print_offset[1],
                       self.z + self.print_offset[2]))
            else:
                self.z = self.pd['locs'][self.pd_idx[self.pd_curr],2]

                score = self.pd_vs[
                    self.z,
                    self.pd['locs'][self.pd_idx[self.pd_curr],1],
                    self.pd['locs'][self.pd_idx[self.pd_curr],0]]
                print('[pd %d/%d matched: %d conf: %.03f score: %.03f (%d,%d,%d)]' %
                      (self.pd_curr+1, self.n_pd,
                       self.pd_match[self.pd_idx[self.pd_curr]],
                       self.pd['conf'][self.pd_idx[self.pd_curr]],
                       score,
                       self.pd['locs'][self.pd_idx[self.pd_curr],0] +
                       self.print_offset[0],
                       self.pd['locs'][self.pd_idx[self.pd_curr],1] +
                       self.print_offset[1],
                       self.z + self.print_offset[2]))
            self.z = int(self.z)
            self.moved = False

        plt.clf()
        if self.mode == 0:
            xx = self.gt['locs'][self.gt_idx[self.gt_curr],0]
            yy = self.gt['locs'][self.gt_idx[self.gt_curr],1]
        else:
            xx = self.pd['locs'][self.pd_idx[self.pd_curr],0]
            yy = self.pd['locs'][self.pd_idx[self.pd_curr],1]
        xx = int(xx)
        yy = int(yy)
        oo = 0.75
        im_slice = oo*self.im[self.z,
                               yy:(yy+2*self.radius+1),
                               xx:(xx+2*self.radius+1)]
        im_slice = np.repeat(im_slice.reshape(im_slice.shape + (1,)),
                             3, axis=2)
        pd_slice = self.pd_voxel[self.z,
                                 yy:(yy+2*self.radius+1),
                                 xx:(xx+2*self.radius+1)]

        if self.overlay:
            im_slice[:,:,0] += (1-oo) * pd_slice
        plt.imshow(im_slice)

        idx = np.logical_and(
            np.logical_and(
                np.abs(self.gt['locs'][:,0] - xx) < self.radius,
                np.abs(self.gt['locs'][:,1] - yy) < self.radius),
            np.abs(self.gt['locs'][:,2] - self.z) < 20).nonzero()
        plt.plot(self.gt['locs'][idx,0] - xx + self.radius,
                 self.gt['locs'][idx,1] - yy + self.radius, 'b.')
        idx = np.logical_and(
            np.logical_and(
                np.abs(self.gt['locs'][:,0] - xx) < self.radius,
                np.abs(self.gt['locs'][:,1] - yy) < self.radius),
            np.abs(self.gt['locs'][:,2] - self.z) < 10).nonzero()
        plt.plot(self.gt['locs'][idx,0] - xx + self.radius,
                 self.gt['locs'][idx,1] - yy + self.radius, 'bo')

        idx = np.logical_and(
            np.logical_and(
                np.abs(self.pd['locs'][:,0] - xx) < self.radius,
                np.abs(self.pd['locs'][:,1] - yy) < self.radius),
            np.abs(self.pd['locs'][:,2] - self.z) < 20).nonzero()
        plt.plot(self.pd['locs'][idx,0] - xx + self.radius,
                 self.pd['locs'][idx,1] - yy + self.radius, 'g.')
        idx = np.logical_and(
            np.logical_and(
                np.abs(self.pd['locs'][:,0] - xx) < self.radius,
                np.abs(self.pd['locs'][:,1] - yy) < self.radius),
            np.abs(self.pd['locs'][:,2] - self.z) < 10).nonzero()
        plt.plot(self.pd['locs'][idx,0] - xx + self.radius,
                 self.pd['locs'][idx,1] - yy + self.radius, 'go')
        plt.draw()

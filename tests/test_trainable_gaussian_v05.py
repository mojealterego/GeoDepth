"""TDD contracts for genuine local spatiotemporal Gaussian parameter optimization."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from omega_geodepth.reconstruction5d import Frame5D
from omega_geodepth.trainable_4dgs import (
    TrainableGaussian4D, GaussianTrainConfig, train_gaussians,
    save_gaussian_checkpoint, load_gaussian_checkpoint,
    init_gaussians_from_depth,
)


def camera():
    return np.array([[24., 0, 9.5], [0., 24., 9.5], [0., 0, 1.]])


def targets_from_teacher():
    teacher = TrainableGaussian4D(
        means_xyz=torch.tensor([[0., 0., 2.4]]),
        velocity=torch.tensor([[.20, 0., 0.]]),
        rgb=torch.tensor([[.8,.12,.05]]),
        sigma_world=torch.tensor([[.14]]),
        opacity=torch.tensor([[.94]]),
        reference_time=0.,
    )
    frames=[]
    with torch.no_grad():
        for t in (0., 1.):
            rgb, alpha, depth=teacher.render(t,camera(),np.eye(4),20,20)
            mask=(alpha.numpy()>.12)
            depth_np=np.where(mask,depth.numpy(),np.nan).astype(np.float32)
            frames.append(Frame5D(t,depth_np,np.rint(rgb.numpy()*255).clip(0,255).astype('uint8'),
                                  camera(),np.eye(4),mask))
    return frames


class GaussianTrainingTests(unittest.TestCase):
    def test_velocity_learned_from_multiple_time_views_not_static_surfel(self):
        frames=targets_from_teacher()
        model=TrainableGaussian4D(
            means_xyz=torch.tensor([[0., 0., 2.4]]),
            velocity=torch.tensor([[0., 0., 0.]]),
            rgb=torch.tensor([[.8,.12,.05]]),
            sigma_world=torch.tensor([[.14]]),
            opacity=torch.tensor([[.94]]),
            reference_time=0.,
        )
        report=train_gaussians(model,frames,GaussianTrainConfig(steps=45,lr=.07,
                            optimize_velocity_only=True,seed=123))
        self.assertLess(report['loss_final'],report['loss_initial']*.75)
        self.assertGreater(float(model.velocity[0,0].detach()),.08)
        self.assertLess(float(model.velocity[0,0].detach()),.35)
        self.assertEqual(report['frames'],2)
        self.assertEqual(report['steps_completed'],45)

    def test_depth_initialization_is_real_backprojection(self):
        img=np.zeros((8,8,3),np.uint8);img[:,:,0]=200
        f=Frame5D(0.,np.ones((8,8),np.float32)*2,img,
                  np.array([[12.,0,3.5],[0,12.,3.5],[0,0,1.]]),np.eye(4),
                  np.ones((8,8),bool))
        m=init_gaussians_from_depth(f,num_splats=16,seed=3)
        self.assertEqual(m.means_xyz.shape,(16,3))
        self.assertTrue(torch.allclose(m.means_xyz[:,2].detach(),torch.full((16,),2.)))
        self.assertEqual(m.reference_time,0.)

    def test_safe_tensors_roundtrip_and_corrupt_checkpoint_fails(self):
        m=TrainableGaussian4D(torch.tensor([[0.,0.,2.]]),torch.zeros(1,3),
            torch.tensor([[.3,.6,.2]]),torch.tensor([[.1]]),torch.tensor([[.9]]),0.0)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            manifest=save_gaussian_checkpoint(m,root,units='arbitrary_scale',
                           provenance='relative_estimated',training_report={'steps_completed':0})
            loaded=load_gaussian_checkpoint(root)
            np.testing.assert_allclose(loaded.means_xyz.detach().numpy(),m.means_xyz.detach().numpy())
            self.assertEqual(manifest['representation'],'trainable_4d_gaussian_scene_v1')
            path=root/'gaussians.safetensors'
            path.write_bytes(path.read_bytes()+b'poison')
            with self.assertRaisesRegex(ValueError,'CHECKPOINT_SHA256_MISMATCH'):
                load_gaussian_checkpoint(root)

    def test_bad_calibration_and_oversized_render_fail_closed(self):
        m=TrainableGaussian4D(torch.tensor([[0.,0.,2.]]),torch.zeros(1,3),
             torch.tensor([[.3,.6,.2]]),torch.tensor([[.1]]),torch.tensor([[.9]]),0.)
        with self.assertRaisesRegex(ValueError,'INTRINSICS'):
            m.render(0.,np.zeros((3,3)),np.eye(4),20,20)
        with self.assertRaisesRegex(ValueError,'RENDER_RESOURCE'):
            m.render(0.,camera(),np.eye(4),2048,2048)

    def test_no_training_on_empty_or_unobserved_depth(self):
        f=targets_from_teacher()[0]
        f.valid[:]=False
        with self.assertRaisesRegex(ValueError,'NO_VALID'):
            init_gaussians_from_depth(f,num_splats=8)

if __name__=='__main__':unittest.main()

class GaussianV05IntegrationTests(unittest.TestCase):
    def test_resize_preserves_pinhole_pixel_center_convention(self):
        from omega_geodepth.trainable_4dgs import resize_frame_for_training
        f=targets_from_teacher()[0]
        original=f.intrinsics.copy()
        f2=resize_frame_for_training(f,10)
        self.assertEqual(f2.depth_z.shape,(10,10))
        self.assertAlmostEqual(f2.intrinsics[0,0],original[0,0]*.5)
        self.assertAlmostEqual(f2.intrinsics[0,2],.5*(original[0,2]+.5)-.5)

    def test_real_cli_train_and_mid_time_render_pipeline(self):
        import json
        from omega_geodepth.trainable_4dgs import main
        frames=targets_from_teacher()
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            records=[]
            for i,frame in enumerate(frames):
                name=f'{i}.npz'
                np.savez_compressed(folder/name,depth_z=frame.depth_z,
                    source_rgb=frame.rgb,intrinsics_k=frame.intrinsics,
                    valid_mask=frame.valid)
                records.append({'time_s':frame.time_s,'npz':name,
                                'pose_c2w':frame.pose_c2w.tolist()})
            (folder/'frames.json').write_text(json.dumps({'units':'arbitrary_scale',
                'provenance':'relative_estimated','frames':records}))
            out=folder/'checkpoint'
            main(['train','--manifest',str(folder/'frames.json'), '--output',str(out),
                  '--splats','10','--steps','5','--lr','.01','--max-side','20'])
            self.assertTrue((out/'gaussians.safetensors').is_file())
            target=folder/'view.png'
            main(['render','--checkpoint',str(out),'--manifest',str(folder/'frames.json'),
                  '--time','.5','--output',str(target),'--max-side','20'])
            self.assertTrue(target.is_file())
            self.assertTrue(target.with_suffix('.npz').is_file())

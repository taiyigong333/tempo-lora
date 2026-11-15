"""回归覆盖缺失相机的dtype与SAM2逐帧更新、末帧读出。"""
import dataclasses
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.tempo_lora.sam import SAM2Stream


class TinyVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.paligemma = SimpleNamespace(config=SimpleNamespace(
            vision_config=SimpleNamespace(image_size=4, patch_size=2),
            text_config=SimpleNamespace(hidden_size=8)))

    def embed_image(self, images, num_frames):
        self.calls += 1
        return torch.ones((images.shape[0]//num_frames,4,8),dtype=torch.bfloat16)

    def embed_language_tokens(self, tokens):
        return torch.ones((*tokens.shape,8),dtype=torch.bfloat16)


class ExactOptimizationTests(unittest.TestCase):
    def test_missing_camera_keeps_embedding_dtype_and_mask(self):
        model=PI0Pytorch.__new__(PI0Pytorch)
        nn.Module.__init__(model)
        model.config=Pi0Config(skip_missing_cameras=False,compact_prefix=False)
        model.gradient_checkpointing_enabled=False
        model.use_sam2_fusion=False
        model.use_action_history=False
        model.action_in_proj=nn.Linear(1,8)
        model.paligemma_with_expert=TinyVision()
        images=[torch.zeros((1,3,3,4,4)) for _ in range(3)]
        masks=[torch.tensor([True]),torch.tensor([True]),torch.tensor([False])]
        args=(images,masks,torch.ones((1,2),dtype=torch.long),torch.ones((1,2),dtype=torch.bool))
        reference=model.embed_prefix(*args)
        self.assertEqual(model.paligemma_with_expert.calls,3)
        model.config=dataclasses.replace(model.config,skip_missing_cameras=True)
        actual=model.embed_prefix(*args)
        self.assertEqual(model.paligemma_with_expert.calls,5)
        self.assertEqual(actual[0].dtype,reference[0].dtype)
        self.assertTrue(torch.equal(actual[0][actual[1]],reference[0][reference[1]]))
        self.assertTrue(torch.equal(actual[1],reference[1]))
        self.assertTrue(torch.equal(actual[2],reference[2]))

    @unittest.skipUnless(torch.cuda.is_available(),'需要CUDA流验证')
    def test_sam_updates_every_frame_and_reads_last(self):
        for latest in (False,True):
            stream=SAM2Stream(nn.Identity().cuda(),latest_features_only=latest)
            calls=[]
            def step(rgb,*,return_tensor=False,return_features=True):
                calls.append((int(rgb[0,0,0]),return_features))
                stream.frame+=1
                if return_features:
                    return torch.full((256,8,8),stream.frame,device='cuda',dtype=torch.float32)
            stream.step=step
            images=[np.full((2,2,3),i,np.uint8) for i in range(4)]
            try:
                actual=stream.encode_async(images)()
                torch.cuda.synchronize()
                self.assertEqual(stream.frame,4)
                self.assertEqual([i for i,_ in calls],[0,1,2,3])
                self.assertEqual([read for _,read in calls],[not latest]*3+[True])
                self.assertEqual(tuple(actual.shape),(1,64,256))
                self.assertTrue(torch.all(actual==4))
            finally:
                stream.executor.shutdown()


if __name__=='__main__':
    unittest.main()

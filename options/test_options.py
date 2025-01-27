from argparse import ArgumentParser

from e4s2024 import DATASETS_ROOT


class TestOptions:

	def __init__(self):
		self.parser = ArgumentParser()
		self.initialize()

	def initialize(self):
		self.parser.add_argument('--exp_dir', type=str, default="/apdcephfs/share_1290939/zhianliu/running_results/our_editing/work_dirs/dummy",help='Path to experiment output directory')
		self.parser.add_argument('--num_seg_cls', type=int, default=12,help='Segmentation mask class number')
		self.parser.add_argument('--remaining_layer_idx', type=int, default=13, help='剩余的几层不用mask')
        # ================= 模型设置 相关 =====================
		self.parser.add_argument('--out_size', type=int, default=1024,help='output image size')      
		self.parser.add_argument('--n_styles', default=11, type=int, help='StyleGAN层数')
		self.parser.add_argument('--fsencoder_type', type=str, default="psp", help='FS Encode网络类型')
		self.parser.add_argument('--extra_encoder_input', type=str, default="diff_map", help='额外的style code补偿Encode网络输入类型') 
        # ================= 数据集 相关 =====================
		self.parser.add_argument('--dataset_root', default='{}/CelebA-HQ'.format(DATASETS_ROOT), type=str, help='dataset root path')
		self.parser.add_argument('--ds_frac', default=1.0, type=float, help='dataset fraction')
		self.parser.add_argument('--test_batch_size', default=1, type=int, help='Batch size for testing and inference')
		self.parser.add_argument('--test_workers', default=4, type=int, help='Number of test/inference dataloader workers')
		self.parser.add_argument('--train_G', default=False, type=bool, help='Whether to train the styleGAN model')
  
		self.parser.add_argument('--output_size', default=1024, type=int, help='Output size of generator')
		self.parser.add_argument('--checkpoint_path', default="/apdcephfs/share_1290939/zhianliu/running_results/our_editing/work_dirs/exp_recon_id0.1_StyleGANWithMask_withoutNorm/checkpoints/iteration_30000.pt", type=str, help='Path to model checkpoint')
		self.parser.add_argument('--save_dir', default="/apdcephfs/share_1290939/zhianliu/running_results/our_editing/work_dirs/exp_recon_StyleGANWithMask_wNorm/testResults", type=str, help='Path to save dir')  
		self.parser.add_argument('--device', default='cuda:0', type=str, help='Which GPU(s) to use')

		self.parser.add_argument('--start_from_latent_avg', action='store_true',default=True, help='Whether to add average latent vector to generate codes from encoder.')
		self.parser.add_argument('--learn_in_w', action='store_true', help='Whether to learn in w space instead of w+')
  
	def parse(self):
		opts = self.parser.parse_args()
		return opts

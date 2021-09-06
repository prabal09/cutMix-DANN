import argparse
import os
from solver import Solver
from torch.backends import cudnn
from data_loader import get_loader
import random, torch
import numpy as np
import datetime
import pdb

cuda_device = 0

manual_seed = 100
random.seed(manual_seed)
torch.manual_seed(manual_seed)
np.random.seed(manual_seed)
random.seed(manual_seed)
os.environ['PYTHONHASHSEED'] = str(0)

if torch.cuda.is_available():
	torch.cuda.manual_seed(manual_seed)
	torch.cuda.manual_seed_all(manual_seed)  
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

def main(config):
	if not os.path.exists(config.s_model_path):
		os.makedirs(config.s_model_path)

	if not os.path.exists(config.src_model_path):
		os.makedirs(config.src_model_path)

	source_loader, target_loader, source_test_loader, target_test_loader = get_loader(config)

	solver = Solver(config, source_loader, target_loader, source_test_loader, target_test_loader)

	if config.method == 'src':
		solver.pretrain()
		solver.test()
		print("Loading best src model............")
		solver.load_best_model()
		solver.test()
	elif config.method == 'dann':
		solver.dann()
	# elif config.method == 'dcm':
	# 	solver.dcm()
	# elif config.method == 'gan1':
	# 	solver.gan1()
	# elif config.method == 'gan2':
	# 	solver.gan2()

	# solver.gan22()
	# solver.train()
	# solver.test()
	# solver.load_best_model()
	solver.tsne(config.tsne_name)
	# solver.get_deep_features()
#python -u main.py &> s_m_dann_g.txt  --source svhn --target mnist --channels 3 --source_testset 1 --total_epochs 200 --clf_test_epoch 5 --method dann --tsne_name s_m_dann_g
if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--p_thresh', type=float, default=0.9) #0.97
	# parser.add_argument('--alignment', type=str, default='global') # | global | local
	parser.add_argument('--method', type=str, default='gan2') # | src | dann | dcm | gan1 | gan2
	parser.add_argument('--e_min', type=int, default=0)
	
	parser.add_argument('--pretrain_epochs', type=int, default=25)
	parser.add_argument('--pretrain_test_epoch', type=int, default=5)
	parser.add_argument('--total_epochs', type=int, default=150)
	parser.add_argument('--clf_test_epoch', type=int, default=10)

	parser.add_argument('--batch_size', type=int, default=128)
	parser.add_argument('--num_workers', type=int, default=8)
	parser.add_argument('--lr', type=float, default=1e-4)
	parser.add_argument('--log_step', type=int, default=50)

	parser.add_argument('--model_id', type=int, default=0)
	parser.add_argument('--model_path', type=str, default='./model')
	parser.add_argument('--db_path', type=str, default='/home/local/ASUAD/schhabr6/DB/digits')
	parser.add_argument('--source', type=str, default='svhn')
	parser.add_argument('--target', type=str, default='mnist')
	parser.add_argument('--source_testset', type=int, default=1)
	parser.add_argument('--target_testset', type=int, default=1)

	parser.add_argument('--init', type=str, default='none') # | none | src | aligned | best_aligned
	parser.add_argument('--save', type=int, default=0)
	parser.add_argument('--tsne_name', type=str, default='tsne')

	parser.add_argument('--image_size', type=int, default=32)
	parser.add_argument('--num_classes', type=int, default=10)
	parser.add_argument('--deep_dim', type=int, default=128)
	parser.add_argument('--fd_h_dim', type=int, default=500)
	parser.add_argument('--channels', type=int, default=3)
	parser.add_argument('--weight_decay', type=float, default=1e-5)

	config = parser.parse_args()
	config.cm=True if config.num_classes<15 else False

	config.src_model_path = os.path.join(config.model_path,config.source + "_" + config.target, str(config.model_id), 'src')

	if config.method == 'src':
		config.s_model_path = config.src_model_path
	# elif config.alignment == 'global':
	# 	config.s_model_path = os.path.join(config.model_path,config.source + "_" + config.target, str(config.model_id), config.method, config.alignment)
	# else:
	# 	config.s_model_path = os.path.join(config.model_path,config.source + "_" + config.target, str(config.model_id), config.method, config.alignment, str(config.e_min))
	config.s_model_path = os.path.join(config.model_path, config.source + "_" + config.target, str(config.model_id),
									   config.method)
	start_time = datetime.datetime.now()
	print("Started at " + str(start_time.strftime('%Y-%m-%d %H:%M:%S')))
	for k in dict(sorted(vars(config).items())).items() : print(k)
	print()
	main(config)
	end_time = datetime.datetime.now()
	print("Ended at " + str(end_time.strftime('%Y-%m-%d %H:%M:%S')))
	duration = end_time - start_time
	print("Duration: " + str(duration))

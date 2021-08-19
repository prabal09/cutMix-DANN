import torch
import torch.nn as nn
import os
import numpy as np
from torch.autograd import Variable
from torch import optim
from torch import Tensor
from model import  encoder, classifier, discriminator
import pdb
from sklearn.metrics import confusion_matrix, accuracy_score
import datetime
from torchsummary import summary
from utils import adjust_alpha, ConditionalEntropyLoss
import seaborn as sns
from MulticoreTSNE import MulticoreTSNE as TSNE
import matplotlib.pyplot as plt
def cutMix(image_batch1,image_batch2):
    l=[];labels = []
    for img1,img2 in zip(image_batch1,image_batch2):
#        break
#        imshow(img1)
#        imshow(img2)
        size = img1.shape
        lam = np.random.beta(1,1)
        bbx1, bby1, bbx2, bby2 = rand_bbox(size,lam)
        img__ = img1.clone()
        img__[0][bbx1:bbx2, bby1:bby2] = img2[0][bbx1:bbx2, bby1:bby2]
        img__[1][bbx1:bbx2, bby1:bby2] = img2[1][bbx1:bbx2, bby1:bby2]
        img__[2][bbx1:bbx2, bby1:bby2] = img2[2][bbx1:bbx2, bby1:bby2]
#        imshow(img__)
        l.append(img__.numpy())
        labels.append(lam)
    l_np = np.asarray(l)
    labels = np.asarray(labels)
    return l_np,labels

class Solver(object):
	def __init__(self, config, source_loader, target_loader, source_test_loader, target_test_loader):
		self.source_loader = source_loader
		self.target_loader = target_loader
		self.source_test_loader = source_test_loader
		self.target_test_loader = target_test_loader

		self.config = config

		self.ce = nn.CrossEntropyLoss()
		self.bce = nn.BCELoss()
		self.cde = ConditionalEntropyLoss()

		self.best_acc = 0
		self.best_train_acc = 0
		self.best_test_acc = 0
		self.time_taken = None
		self.test_no_increase = 0
		self.best_s_test_acc = 0

		self.enc = encoder(self.config)
		self.clf = classifier(self.config)
		self.fd = discriminator(self.config)

		self.is_cuda = torch.cuda.is_available()

		self.c_optimizer = optim.Adam(list(self.enc.parameters()) + list(self.clf.parameters() + list(self.fd.parameters())), self.config.lr, betas=[0.5, 0.999], weight_decay= self.config.weight_decay)
		#self.fd_optimizer = optim.Adam(self.fd.parameters(), self.config.lr, [0.5, 0.999], weight_decay= self.config.weight_decay)

		if self.is_cuda:
			self.enc.cuda()
			self.clf.cuda()
			self.fd.cuda()

		if self.config.init =='src':
			self.enc.load_state_dict(torch.load(os.path.join(self.config.src_model_path, 'best_enc.pkl')))
			self.clf.load_state_dict(torch.load(os.path.join(self.config.src_model_path, 'best_clf.pkl')))

		elif self.config.init == 'aligned':
			self.enc.load_state_dict(torch.load(os.path.join(self.config.s_model_path, 'enc.pkl')))
			self.clf.load_state_dict(torch.load(os.path.join(self.config.s_model_path, 'clf.pkl')))
			self.fd.load_state_dict(torch.load(os.path.join(self.config.s_model_path, 'fd.pkl')))

		elif self.config.init == 'best_aligned':
			self.enc.load_state_dict(torch.load(os.path.join(self.config.s_model_path, 'best_enc.pkl')))
			self.clf.load_state_dict(torch.load(os.path.join(self.config.s_model_path, 'best_clf.pkl')))
			self.fd.load_state_dict(torch.load(os.path.join(self.config.s_model_path, 'best_fd.pkl')))

		print('--------Classifier--------')
		print(self.enc)
		summary(self.enc, input_size=(self.config.channels, self.config.image_size, self.config.image_size))
		print(self.clf)
		summary(self.clf, input_size=(self.config.deep_dim,))

		print('--------FeatureDisc--------')
		print(self.fd)
		summary(self.fd, input_size=(self.config.deep_dim,))

	def load_best_model(self):
		self.enc.load_state_dict(torch.load(os.path.join(self.config.s_model_path, 'best_enc.pkl')))
		self.clf.load_state_dict(torch.load(os.path.join(self.config.s_model_path, 'best_clf.pkl')))

	def reset_grad(self):
		self.c_optimizer.zero_grad()
		self.fd_optimizer.zero_grad()

	def test_dataset(self, db='tgt_test'):
		self.enc.eval()
		self.clf.eval()
		actual = []
		pred = []

		if db.lower() == 'src_train':
			loader = self.source_loader
		elif db.lower() == 'src_test':
			loader = self.source_test_loader
		elif db.lower() == 'tgt_train':
			loader = self.target_loader
		else:
			loader = self.target_test_loader

		for data in loader:
			img, label = data
			if self.is_cuda:
				img = img.cuda()

			with torch.no_grad():
				deep_feat = self.enc(img)
				class_out = self.clf(deep_feat)

			_, predicted = torch.max(class_out.data, 1)
			actual += label.tolist()
			pred += predicted.tolist()
		acc = accuracy_score(y_true=actual, y_pred=pred) * 100
		cm = confusion_matrix(y_true=actual, y_pred=pred, labels=range(self.config.num_classes))
		return acc, cm

	def test(self, db=None):
		s_acc_train, s_acc_test, t_acc_train, t_acc_test = 0.0, 0.0, 0.0, 0.0

		if db=='src' or db is None:
			if self.config.source_testset == 1:
				s_acc_train, s_cm_train = self.test_dataset('src_train')
				print('source_train_acc: %.2f' %(s_acc_train))
				if self.config.cm:	print(s_cm_train)
			else:
				s_acc_train, s_cm_test = 0, 0

			s_acc_test, s_cm_test = self.test_dataset('src_test')
			print('source_test_acc: %.2f' %(s_acc_test))
			if self.config.cm:	print(s_cm_test)

		if db =='tgt' or db is None:
			if self.config.target_testset == 1:
				t_acc_train, t_cm_train = self.test_dataset('tgt_train')
				print('target_train_acc: %.2f' %(t_acc_train))
				if self.config.cm:	print(t_cm_train)
			else:
				t_acc_train, t_cm_train = 0, 0

			t_acc_test, t_cm_test = self.test_dataset('tgt_test')
			print('target_test_acc: %.2f' %(t_acc_test))
			if self.config.cm:	print(t_cm_test)

		if db is None:
			if not self.config.cm: print('source_acc: %.2f, target_train_acc: %.2f, target_test_acc: %.2f' %(s_acc_test, t_acc_train, t_acc_test))

		return s_acc_train, s_acc_test, t_acc_train, t_acc_test

	def get_local_d_outs(self, domain_out, labels):
		y_onehot = torch.FloatTensor(domain_out.shape[0], self.config.num_classes)
		y_onehot.zero_()
		if self.is_cuda:
			y_onehot = y_onehot.cuda()
		y_onehot.scatter_(1, torch.unsqueeze(labels, dim=-1), 1)

		domain_out = domain_out * y_onehot
		domain_out = domain_out.sum(1, keepdim=True)

		return domain_out

	def pretrain(self):

		source_iter = iter(self.source_loader)
		s_iter_per_epoch = len(source_iter)
		total_iters = 0

		print("Source iters per epoch: %d"%(s_iter_per_epoch))

		print("---------------Starting training at " + str(datetime.datetime.now()) + "---------------")
		for epoch in range(self.config.pretrain_epochs):

			self.clf.train()
			self.enc.train()
			self.fd.train()

			for i, (source, s_labels) in enumerate(self.source_loader):
				total_iters += 1

				# load source and target dataset
				if self.is_cuda:
					source, s_labels = source.cuda(), s_labels.cuda()

				s_deep = self.enc(source)
				s_out = self.clf(s_deep)

				s_clf_loss = self.ce(s_out, s_labels)

				loss = s_clf_loss

				self.c_optimizer.zero_grad()
				loss.backward()
				self.c_optimizer.step()

				_, predicted = torch.max(s_out.data, 1)
				b_s_acc = (predicted == s_labels).float().mean()
				s_avg_prob = torch.mean(torch.max(torch.softmax(s_out, dim=1), dim=1)[0])

				if i % self.config.log_step == 0 or i == (s_iter_per_epoch-1):
					print('Ep: %02d/%d, iter: %04d/%04d, total_iters: %05d, s_err: %.4f, s_acc: %.2f, s_p: %.2f' %(epoch+1, self.config.pretrain_epochs, i+1, s_iter_per_epoch, total_iters, s_clf_loss, b_s_acc.item(), s_avg_prob))

			if (epoch+1) % self.config.pretrain_test_epoch == 0:
				if self.config.save == 1:
					enc_path = os.path.join(self.config.src_model_path, 'enc.pkl')
					torch.save(self.enc.state_dict(), enc_path)

					clf_path = os.path.join(self.config.src_model_path, 'clf.pkl')
					torch.save(self.clf.state_dict(), clf_path)

				s_acc_train, s_acc_test, t_acc_train, t_acc_test = self.test(db='src')

				if s_acc_test > self.best_s_test_acc:
					self.best_s_test_acc = s_acc_test

					if self.config.save == 1:
						enc_path = os.path.join(self.config.src_model_path, 'best_enc.pkl')
						torch.save(self.enc.state_dict(), enc_path)

						clf_path = os.path.join(self.config.src_model_path, 'best_clf.pkl')
						torch.save(self.clf.state_dict(), clf_path)

				if t_acc_test > self.best_test_acc:
					self.best_test_acc = t_acc_test
				print("Best source test acc: %.2f, Best tgt test acc: %.2f" %(self.best_s_test_acc, self.best_test_acc))
				# self.tsne(self.config.tsne_name + "_" + str(epoch+1))

	def dann(self):

		source_iter = iter(self.source_loader)
		target_iter = iter(self.target_loader)
		s_iter_per_epoch = len(source_iter)
		t_iter_per_epoch = len(target_iter)
		min_len = min(s_iter_per_epoch, t_iter_per_epoch)
		total_iters = 0

		print("Source iters per epoch: %d"%(s_iter_per_epoch))
		print("Target iters per epoch: %d"%(t_iter_per_epoch))
		print("iters per epoch: %d"%(min(s_iter_per_epoch,t_iter_per_epoch)))

		print("---------------Starting training at " + str(datetime.datetime.now()) + "---------------")
		for epoch in range(self.config.total_epochs):
			self.clf.train()
			self.enc.train()
			self.fd.train()

			for i, (source_data, target_data) in enumerate(zip(self.source_loader, self.target_loader)):
				total_iters += 1
				alpha = adjust_alpha(i, epoch, min_len, self.config.total_epochs)

				# load source and target dataset
				source, s_labels = source_data
				'''
				if self.is_cuda:
					source, s_labels = source.cuda(), s_labels.cuda()
				'''

				target, t_labels = target_data
				'''
				if self.is_cuda:
					target, t_labels = target.cuda(), t_labels.cuda()
				'''

				#fake_label = torch.FloatTensor(self.config.batch_size, 1).fill_(0)
				'''
				if self.is_cuda:
					fake_label = fake_label.cuda()
				'''
				#real_label = torch.FloatTensor(self.config.batch_size, 1).fill_(1)
				'''
				if self.is_cuda:
					real_label = real_label.cuda()
				'''

				cutmix, c_labels = cutMix(source, target)
				cutmix = Tensor(cutmix)
				c_labels = Tensor(c_labels)
				if is_cuda:
					cutmix = cutmix.cuda()


				s_deep = self.enc(source)
				s_out = self.clf(s_deep)

				t_deep = self.enc(target)
				t_out = self.clf(t_deep)


				c_deep = self.enc(cutmix)
				#c_out = self.clf(c_deep)

				s_clf_loss = self.ce(s_out, s_labels)
				c_clf_loss = self.ce(c_out, s_labels)

				#loss = s_clf_loss
				loss = c_clf_loss

				'''
				s_domain_out = self.fd(s_deep, alpha=alpha)
				t_domain_out = self.fd(t_deep, alpha=alpha)
				'''
				domain_out = self.fd(c_deep, alpha=alpha)
				'''
				if self.config.alignment == 'local':
					_, t_ps_labels = torch.max(t_out.data, 1)
					y_probs = torch.max(torch.softmax(t_out, dim=1), dim=1)[0]
					selected = y_probs > self.config.p_thresh
					b_ts_acc = (t_ps_labels == t_labels)[selected].float().mean()

					s_domain_out, t_domain_out = self.get_local_d_outs(s_domain_out, s_labels), self.get_local_d_outs(t_domain_out, t_ps_labels)

					s_domain_err = self.bce(s_domain_out, real_label)

					#-----Loss on all samples-----#
					t_domain_err = self.bce(t_domain_out, fake_label)

					#-----Loss on selected-----#
					if fake_label.shape[0] > 0:
						t_domain_err_selected = self.bce(t_domain_out[selected], fake_label[selected])
					else:
						t_domain_err_selected = torch.Tensor([0]).cuda() if self.cuda else torch.Tensor([0])

					disc_dann_loss = (s_domain_err + t_domain_err)

					dann_loss = (s_domain_err + t_domain_err_selected)

				else:
					s_domain_err = self.bce(s_domain_out, real_label)
					t_domain_err = self.bce(t_domain_out, fake_label)
					dann_loss = s_domain_err + t_domain_err
				'''
				dann_loss = self.bce(domain_out, c_labels)
				loss = loss + dann_loss
				'''
				if self.config.e_min == 1:
					t_loss = self.cde(t_out)  * 0.01
					loss = loss + t_loss
				'''

				self.reset_grad()
				'''
				if self.config.alignment == 'local':
					loss.backward(retain_graph=True)
					self.c_optimizer.step()
					self.reset_grad()
					disc_dann_loss.backward()
					self.fd_optimizer.step()
				else:
					loss.backward()
					self.c_optimizer.step()
					self.fd_optimizer.step()
				'''
				loss.backward()
				self.c_optimizer.step()
				#self.fd_optimizer.step()

				_, predicted = torch.max(s_out.data, 1)
				b_s_acc = (predicted == s_labels).float().mean()
				s_avg_prob = torch.mean(torch.max(torch.softmax(c_out, dim=1), dim=1)[0])

				_, predicted = torch.max(t_out.data, 1)
				b_t_acc = (predicted == t_labels).float().mean()
				t_avg_prob = torch.mean(torch.max(torch.softmax(t_out, dim=1), dim=1)[0])

				'''
				if i % self.config.log_step == 0 or i == (min_len-1):
					print('Ep: %02d/%d, iter: %04d/%04d, total_iters: %05d, s_err: %.4f, s_acc: %.2f, s_p: %.2f, t_acc: %.2f, t_p: %.2f, a: %.3f, sd_err: %.2f, sd_p: %.2f, td_err: %.2f, td_p: %.2f'
						%(epoch+1, self.config.total_epochs, i+1, min_len, total_iters, s_clf_loss, b_s_acc.item(), s_avg_prob, b_t_acc.item(), t_avg_prob, alpha, s_domain_err, s_domain_out.mean(), t_domain_err, t_domain_out.mean()), end='')

					if self.config.e_min == 1:
						print(", t_loss: %.4f"%(t_loss.item()), end='')

					if self.config.alignment == 'local':
						print(", th: %02d(%.2f)"%(selected.sum(), b_ts_acc.item()), end='')

					print()
				'''

			if (epoch+1) % self.config.clf_test_epoch == 0:
				if self.config.save == 1:
					enc_path = os.path.join(self.config.s_model_path, 'enc.pkl')
					torch.save(self.enc.state_dict(), enc_path)

					clf_path = os.path.join(self.config.s_model_path, 'clf.pkl')
					torch.save(self.clf.state_dict(), clf_path)

					fd_path = os.path.join(self.config.s_model_path, 'fd.pkl')
					torch.save(self.fd.state_dict(), fd_path)

				s_acc_train, s_acc_test, t_acc_train, t_acc_test = self.test(db='tgt')

				if t_acc_test > self.best_test_acc:
					self.best_test_acc = t_acc_test
					enc_path = os.path.join(self.config.s_model_path, 'best_enc.pkl')
					torch.save(self.enc.state_dict(), enc_path)

					clf_path = os.path.join(self.config.s_model_path, 'best_clf.pkl')
					torch.save(self.clf.state_dict(), clf_path)

					fd_path = os.path.join(self.config.s_model_path, 'best_fd.pkl')
					torch.save(self.fd.state_dict(), fd_path)
				print("Best tgt test acc: %.2f" %(self.best_test_acc))
				# self.tsne(self.config.tsne_name + "_" + str(epoch+1))


	def tsne(self, fname=None, samples_to_use=1000):
		self.enc.eval()
		self.clf.eval()

		features = []
		labels = []
		for i, data in enumerate(self.source_loader):
			if i *  self.config.batch_size > samples_to_use:
				break
			img, label = data
			if self.is_cuda:
				img = img.cuda()

			with torch.no_grad():
				d_feature = self.enc(img)
				# d_feature = self.enc(img)
				features.append(d_feature.detach().cpu().numpy())
			labels.append(label)
		s_features = np.vstack(features)
		s_label = np.hstack(labels)

		features = []
		labels = []
		for i, data in enumerate(self.target_loader):
			if i *  self.config.batch_size > samples_to_use:
				break
			img, label = data
			if self.is_cuda:
				img = img.cuda()

			with torch.no_grad():
				d_feature = self.enc(img)
				features.append(d_feature.detach().cpu().numpy())
			labels.append(label)
		t_features = np.vstack(features)
		t_label = np.hstack(labels)

		X = np.vstack((s_features, t_features))
		y = np.hstack((s_label, t_label))

		# tnse =  TSNE(n_jobs=10).fit(s_features)
		x_ = TSNE(n_jobs=10).fit_transform(X)
		# x_ = tnse.transform(X)
		names = range(self.config.num_classes) #['Source', 'Target'] # range(10)
		dm = np.hstack((np.ones(s_label.shape[0]) * 0, np.ones(t_label.shape[0]) * 1))
		self.fashion_scatter(x_, y, names, dm, fname)
		return

	def fashion_scatter(self, x, y, names=[], dm=None, fname=None):
		# pdb.set_trace()
		num_classes = len(np.unique(y))
		palette = np.array(sns.color_palette('deep', n_colors=num_classes))

		f = plt.figure(figsize=(8, 8))
		ax = plt.subplot(aspect='equal')

		for i in range(num_classes):
			source_samples = x[np.logical_and(y==i, dm==0)]
			plt.scatter(source_samples[:, 0], source_samples[:, 1], label=names[i], marker='o', color=palette[i], s=10) #, facecolors='none', edgecolors=palette[i], s=20)

			target_samples = x[np.logical_and(y==i, dm==1)]
			plt.scatter(target_samples[:, 0], target_samples[:, 1], label=names[i], marker='x', color=palette[i], s=40) #, linewidth=2.5) #, s=45)#, linewidths=1) #s=50,, facecolors='none'

		plt.xlim(-300, 300)
		plt.ylim(-300, 300)
		ax.axis('off')
		ax.axis('tight')
		# plt.legend() #loc='upper left', fontsize=20, markerscale=6)
		if fname is None:
			plt.savefig('tsne.png')
		else:
			plt.savefig(fname)
		return
		plt.close('all')

	def get_deep_features(self):
		self.enc.eval()
		samples_to_use = 1000
		features = []
		labels = []
		for i, data in enumerate(self.source_test_loader):
			# if i *  self.config.batch_size > samples_to_use:
			# 	break
			img, label = data
			if self.is_cuda:
				img = img.cuda()

			with torch.no_grad():
				d_feature = self.enc(img)
				# d_feature = self.enc(img)
				features.append(d_feature.detach().cpu().numpy())
			labels.append(label)
		s_features = np.vstack(features)
		s_label = np.hstack(labels)

		s_data = np.hstack((s_features, s_label.reshape([-1, 1])))
		print(s_data.shape)
		np.save('src', s_data)

		features = []
		labels = []
		for i, data in enumerate(self.target_test_loader):
			# if i *  self.config.batch_size > samples_to_use:
			# 	break
			img, label = data
			if self.is_cuda:
				img = img.cuda()

			with torch.no_grad():
				d_feature = self.enc(img)
				features.append(d_feature.detach().cpu().numpy())
			labels.append(label)
		t_features = np.vstack(features)
		t_label = np.hstack(labels)

		t_data = np.hstack((t_features, t_label.reshape([-1, 1])))
		print(t_data.shape)
		np.save('tgt', t_data)

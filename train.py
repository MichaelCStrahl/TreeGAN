import torch
import torch.nn as nn
import torch.optim as optim

from data.dataset_benchmark import BenchmarkDataset
from data.shapenet_dataset import ShapenetDataset
from Classificador.classificador import Tnet, PointNetBackbone, PointNetClassHead, PointNetSegHead
from model.gan_network import Generator, Discriminator
from model.gradient_penalty import GradientPenalty
from evaluation.FPD import calculate_fpd

from arguments import Arguments
import plotly.express as px
import pandas as pd
import time
#import visdom
import numpy as np
import plotly.graph_objects as go
import torch.nn.functional as F

CATEGORIES = {
    #'Airplane': 0,
    #'Bag': 1,
    'Cap': 2,
   # 'Car': 3,
    #'Chair': 4,
    'Earphone': 5,
   ## 'Guitar': 6,
   # 'Knife': 7,
   # 'Lamp': 8,
   # 'Laptop': 9,
   ## 'Motorbike': 10,
    #'Mug': 11,
   # 'Pistol': 12,
   ## 'Rocket': 13,
    #'Skateboard': 14,
   # 'Table': 15
    }

class_choice = ['Cap','Earphone']

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class TreeGAN():
    def __init__(self, args):
        self.args = args
        # ------------------------------------------------Dataset---------------------------------------------- #
        self.data = BenchmarkDataset(root=args.dataset_path, npoints=args.point_num, uniform=True, class_choice=class_choice)
        self.dataLoader = torch.utils.data.DataLoader(self.data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)
        print("Training Dataset : {} prepared.".format(len(self.data)))
        # ----------------------------------------------------------------------------------------------------- #
        MODEL_PATH = 'drive/MyDrive/tcc/classificador/cls_model_2class_998.pth'
        self.classifier = PointNetClassHead(num_points=args.point_num, num_global_feats=1024, k=16).to(args.device)
        self.classifier.load_state_dict(torch.load(MODEL_PATH))
        self.classifier.eval()
        # -------------------------------------------------Module---------------------------------------------- #
        self.G = Generator(batch_size=args.batch_size, features=args.G_FEAT, degrees=args.DEGREE, support=args.support).to(args.device)
        self.D = Discriminator(batch_size=args.batch_size, features=args.D_FEAT).to(args.device)             
        
        self.optimizerG = optim.Adam(self.G.parameters(), lr=args.lr, betas=(0, 0.99))
        self.optimizerD = optim.Adam(self.D.parameters(), lr=args.lr, betas=(0, 0.99))

        self.GP = GradientPenalty(args.lambdaGP, gamma=1, device=args.device)
        print("Network prepared.")
    
        # ----------------------------------------------------------------------------------------------------- #
    
    def run(self, save_ckpt=None, load_ckpt=None):        
        color_num = 4
        chunk_size = int(self.args.point_num / color_num)
        colors = np.array([(227,0,27),(231,64,28),(237,120,15),(246,176,44),
                           (252,234,0),(224,221,128),(142,188,40),(18,126,68),
                           (63,174,0),(113,169,156),(164,194,184),(51,186,216),
                           (0,152,206),(16,68,151),(57,64,139),(96,72,132),
                           (172,113,161),(202,174,199),(145,35,132),(201,47,133),
                           (229,0,123),(225,106,112),(163,38,42),(128,128,128)])
        colors = colors[np.random.choice(len(colors), color_num, replace=False)]
        label = torch.stack([torch.ones(chunk_size).type(torch.LongTensor) * inx for inx in range(1,int(color_num)+1)], dim=0).view(-1)

        if load_ckpt is None:
            epoch_log = 0
            iter_log = 0
            
            loss_log = {'G_loss': [], 'D_loss': []}
            loss_legend = list(loss_log.keys())

            metric = {'FPD': []}
        else:
            metric = {}
            loss_log = {'G_loss': [], 'D_loss': []}
            checkpoint = torch.load(load_ckpt)
            self.D.load_state_dict(checkpoint['D_state_dict'])
            self.G.load_state_dict(checkpoint['G_state_dict'])

            epoch_log = checkpoint['epoch']
            iter_log = checkpoint['iter']

            loss_log['G_loss'] = checkpoint['G_loss']
            loss_log['D_loss'] = checkpoint['D_loss']
            loss_legend = list(loss_log.keys())

            metric['FPD'] = checkpoint['FPD']
            
            print("Checkpoint loaded.")
        
        for epoch in range(epoch_log, self.args.epochs):
            iter_total = 0
            pred_class = ''
            pred_prob = 0.0
            for _iter, data in enumerate(self.dataLoader, iter_log):
                # Start Time
                start_time = time.time()
                point, _ = data
                point = point.to(args.device)
                iter_total+=1

                # -------------------- Discriminator -------------------- #
                for d_iter in range(self.args.D_iter):
                    self.D.zero_grad()
                    
                    z = torch.randn(self.args.batch_size, 1, 96).to(args.device)
                    tree = [z]
                    
                    with torch.no_grad():
                        fake_point = self.G(tree)         
                        
                    D_real = self.D(point)
                    D_realm = D_real.mean()

                    D_fake = self.D(fake_point)
                    D_fakem = D_fake.mean()

                    gp_loss = self.GP(self.D, point.data, fake_point.data)
                    
                    d_loss = -D_realm + D_fakem
                    d_loss_gp = d_loss + gp_loss
                    d_loss_gp.backward()
                    self.optimizerD.step()

                loss_log['D_loss'].append(d_loss.item())                  
                
                # ---------------------- Generator ---------------------- #
                self.G.zero_grad()
                
                z = torch.randn(self.args.batch_size, 1, 96).to(args.device)
                tree = [z]
                
                fake_point = self.G(tree)
                G_fake = self.D(fake_point)
                G_fakem = G_fake.mean()
                
                g_loss = -G_fakem
                g_loss.backward()
                self.optimizerG.step()

                loss_log['G_loss'].append(g_loss.item())
                
        
                # --------------------- Visualization -------------------- #

                print("[Epoch/Iter] ", "{:3} / {:3}".format(epoch, _iter),
                      "[ D_Loss ] ", "{: 7.6f}".format(d_loss), 
                      "[ G_Loss ] ", "{: 7.6f}".format(g_loss), 
                      "[ Time ] ", "{:4.2f}s".format(time.time()-start_time))

                if _iter % 20 == 0 and _iter !=0:

            
                    generated_point = self.G.getPointcloud()
                    points = generated_point
                    """plot_X = np.stack([np.arange(len(loss_log[legend])) for legend in loss_legend], 1)
                    plot_Y = np.stack([np.array(loss_log[legend]) for legend in loss_legend], 1)"""
                    plot_X = np.stack([np.arange(len(loss_log[legend])) for legend in loss_legend], 1) 
                    plot_Y = np.stack([np.array(loss_log[legend]) for legend in loss_legend], 1)
                    new_x = [row[0] for row in plot_X]
                    loss_G = [row[0] for row in plot_Y]
                    loss_D = [row[1] for row in plot_Y]
                    
                    #print(new_x)
                    #print(plot_X)
                    #print(loss_G)
                    #print(loss_D)
                    
                    torch.cuda.empty_cache()
                    points = generated_point
            
                    norm_points = ShapenetDataset.normalize_points(points)
                    
                    norm_points = ShapenetDataset.normalize_points(points)
                    with torch.no_grad():
                        norm_points = norm_points.unsqueeze(0).transpose(2, 1).to(DEVICE)
                        preds, _, _ = self.classifier(norm_points)
                        preds = torch.softmax(preds, dim=1)
                        pred_choice = preds.squeeze().argmax()
                    pred_class = list(CATEGORIES.keys())[pred_choice.cpu().numpy()]
                    pred_prob = preds[0, pred_choice]
                    print(f'The predicted class is: {pred_class}, with probability: {pred_prob}')
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=new_x, y=loss_G,
                                        mode='lines',
                                        name='Loss G'))
                    fig.add_trace(go.Scatter(x=new_x, y=loss_D,
                                        mode='lines',
                                        name='Loss D'))
                    fig.show()
                    generated_point_cpu = generated_point.to('cpu').detach().numpy()
                    x = generated_point_cpu[:, 2]
                    y = generated_point_cpu[:, 0]
                    z = generated_point_cpu[:, 1]
                    trace = go.Scatter3d(
                        x=x,
                        y=y,
                        z=z,
                        mode='markers',
                        marker=dict(
                            size=2,
                            color=colors,
                            colorscale='Viridis',  # Adjust the colorscale as per your preference
                            opacity=1,
                        )
                    )
                    data = [trace]
                    # Create the layout for the plot
                    layout = go.Layout(
                        title='Generated Pointcloud',
                        scene=dict(
                            xaxis_title='X',
                            yaxis_title='Y',
                            zaxis_title='Z'
                        ),
                        width=800,
                        height=800
                    )
                    # Create the figure using data and layout
                    fig = go.Figure(data=data, layout=layout)

                    # Show the plot
                    fig.show()

                    """self.vis.line(X=plot_X, Y=plot_Y, win=1,
                                  opts={'title': 'TreeGAN Loss', 'legend': loss_legend, 'xlabel': 'Iteration', 'ylabel': 'Loss'})"""

                    """self.vis.scatter(X=generated_point[:,torch.LongTensor([2,0,1])], Y=label, win=2,
                                     opts={'title': "Generated Pointcloud", 'markersize': 2, 'markercolor': colors, 'webgl': True})"""

                    """if len(metric['FPD']) > 0:
                        self.vis.line(X=np.arange(len(metric['FPD'])), Y=np.array(metric['FPD']), win=3, 
                                      opts={'title': "Frechet Pointcloud Distance", 'legend': ["FPD best : {}".format(np.min(metric['FPD']))]})"""

                    print('Figures are saved.')

            # ---------------------- Save checkpoint --------------------- #
            if epoch % 1 == 0 and not save_ckpt == None:
                print(f'teste: {pred_class}, teste: {pred_prob}')
                torch.save({
                        'epoch': epoch,
                        'iter': _iter,
                        'D_state_dict': self.D.state_dict(),
                        'G_state_dict': self.G.state_dict(),
                        'D_loss': loss_log['D_loss'],
                        'G_loss': loss_log['G_loss'],
                        'FPD': metric['FPD']
                }, save_ckpt+str(epoch)+pred_class+'.pt')

                print('Checkpoint is saved.')
                
                
                
            # ---------------- Frechet Pointcloud Distance --------------- #
            if epoch % 1 == 0:
                fake_pointclouds = torch.Tensor([])
                for i in range(100): # batch_size * 100
                    z = torch.randn(self.args.batch_size, 1, 96).to(self.args.device)
                    tree = [z]
                    with torch.no_grad():
                        sample = self.G(tree).cpu()
                    fake_pointclouds = torch.cat((fake_pointclouds, sample), dim=0)

                fpd = calculate_fpd(fake_pointclouds, batch_size=100, dims=1808, device=self.args.device)
                metric['FPD'].append(fpd)
                print('[{:4} Epoch] Frechet Pointcloud Distance <<< {:.10f} >>>'.format(epoch, fpd))
                class_name = class_choice if class_choice is not None else 'all'
                #torch.save(fake_pointclouds, '/content/drive/MyDrive/ResultadosLothar/Generated/treeGCN3class_{}_{}.pt'.format(str(epoch), class_name))
                del fake_pointclouds
                
            
                
                    

if __name__ == '__main__':
    args = Arguments().parser().parse_args()

    args.device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')

    SAVE_CHECKPOINT = args.ckpt_path + args.ckpt_save if args.ckpt_save is not None else None
    LOAD_CHECKPOINT = args.ckpt_path + args.ckpt_load if args.ckpt_load is not None else None

    print(SAVE_CHECKPOINT, LOAD_CHECKPOINT)

    model = TreeGAN(args)
    model.run(save_ckpt=SAVE_CHECKPOINT, load_ckpt=LOAD_CHECKPOINT)

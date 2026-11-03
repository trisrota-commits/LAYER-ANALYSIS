"""Publication-oriented schematic figures; vector PDF plus PNG previews.
No experimental values are invented. Earlier versions remain untouched.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent
INK, EDGE, FILL = '#273344', '#667486', '#F3F4F6'
FORWARD, BACKWARD = '#356B9A', '#805496'
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11, 'pdf.fonttype':42})

def canvas(height):
    fig, ax = plt.subplots(figsize=(10,height))
    fig.subplots_adjust(left=.025,right=.975,top=.98,bottom=.03)
    ax.set(xlim=(0,10),ylim=(0,6))
    ax.axis('off')
    return fig,ax

def text(ax,x,y,s,size=11,bold=False,color=INK):
    ax.text(x,y,s,ha='center',va='center',fontsize=size,
            weight='bold' if bold else 'normal',color=color,linespacing=1.4)

def box(ax,x,y,w,h,s,size=11):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.025,rounding_size=.07',
                              facecolor=FILL,edgecolor=EDGE,linewidth=1))
    text(ax,x+w/2,y+h/2,s,size)

def arrow(ax,x,y,xx,yy,color=EDGE):
    ax.add_patch(FancyArrowPatch((x,y),(xx,yy),arrowstyle='-|>',mutation_scale=12,
                                linewidth=1.15,color=color))

def line(ax,points,color=EDGE,dashed=False):
    ax.plot([p[0] for p in points],[p[1] for p in points],color=color,
            lw=1.15,ls='--' if dashed else '-')

def save(fig,name):
    for ext in ('pdf','png'):
        p=ROOT/f'{name}.{ext}'
        if p.exists(): raise FileExistsError(p)
        fig.savefig(p,dpi=200,facecolor='white')
    plt.close(fig)

# Persistence diagram: semantic payload identity, not a claim about file naming.
f,a=canvas(3.8)
for x,s in [(.15,'Candidate\ncheckpoint'),(2.7,'Layer\nmeasurements'),(5.25,'Checkpoint record\nlayer_stat payload'),(7.8,'Database\npersistence')]:
    box(a,x,4.5,2.05,1,s,10.5)
for x in (.15,2.7,5.25): arrow(a,x+2.1,5,x+2.5,5)
text(a,1.3,3.65,'Analysis checkpoints',10,True)
text(a,1.3,2.9,'1, 2, 3, 4, 5,\n10, 15, …, 50',10)
arrow(a,3.725,4.45,3.725,3.85)
box(a,2.6,1.05,4.35,2.7,'')
text(a,4.775,3.42,'Selected layer fields',10,True)
for x,s in [(3.12,'Layer'),(4.45,'taylor_imp'),(5.85,'grad_norm'),(6.65,'…')]: text(a,x,2.83,s,9,True)
line(a,[(2.75,2.52),(6.8,2.52)])
for y,vals in [(2.1,['conv1',r'$T_1$',r'$g_1$','…']),(1.48,['head',r'$T_2$',r'$g_2$','…'])]:
    for x,s in zip([3.12,4.45,5.85,6.65],vals):text(a,x,y,s,10)
text(a,4.775,.52,'Schematic rows; symbols are not measured results.',9)
arrow(a,8.825,4.45,8.825,3.85)
box(a,7.35,2.5,2.5,1.25,'layer_stat\nper_layer_stat',11)
text(a,8.6,1.8,'Linked to the\ntraining record',10)
save(f,'checkpoint_storage')

# Forward/backward and explicit elementwise reduction.
f,a=canvas(4.5)
box(a,.25,4.35,2.6,1,'Training minibatch\n(inputs and labels)',10.5)
box(a,3.7,4.35,2.6,1,r'Layer activations $A_l$',11)
box(a,7.15,4.35,2.6,1,r'Minibatch loss $\mathcal{L}_{\mathcal{B}}$',11)
arrow(a,2.9,4.86,3.65,4.86,FORWARD)
arrow(a,6.35,4.86,7.1,4.86,FORWARD)
text(a,5,5.72,'Forward pass',10,color=FORWARD)
line(a,[(8.45,4.3),(8.45,3.65),(5.8,3.65)],BACKWARD)
arrow(a,5.8,3.65,5.8,4.3,BACKWARD)
text(a,7.3,3.23,r'Backward: $\partial\mathcal{L}_{\mathcal{B}}/\partial A_l$',10,color=BACKWARD)
line(a,[(4.2,4.3),(4.2,3.0),(1.7,3.0)])
arrow(a,1.7,3.0,1.7,2.55)
text(a,2.65,3.36,'Activations and gradients',9)
box(a,.25,1.65,2.9,.85,r'Element-wise $A_l\odot\nabla_{A_l}\mathcal{L}_{\mathcal{B}}$',10.5)
box(a,3.7,1.65,2.6,.85,'Absolute value',11)
box(a,6.85,1.65,2.9,.85,'Average within layer',11)
arrow(a,3.2,2.075,3.65,2.075)
arrow(a,6.35,2.075,6.8,2.075)
text(a,5,.97,r'$T_l=\frac{1}{|A_l|}\sum_u\left|A_{l,u}\frac{\partial\mathcal{L}_{\mathcal{B}}}{\partial A_{l,u}}\right|$',15)
text(a,5,.25,'Finite layer scores → unweighted layer average → one value per checkpoint',10)
save(f,'taylor_calculation')

# Explicitly held-out inference; targets feed evaluation, never prediction.
f,a=canvas(4.7)
arrow(a,.4,5.2,6.2,5.2)
for x,e in [(1,1),(3,2),(5,3)]:
    a.plot(x,5.2,'o',ms=6,color=EDGE)
    text(a,x,5.65,f'Epoch {e}',11,True)
    arrow(a,x,5.1,x,4.65)
    box(a,x-.8,3.9,1.6,.7,'Accuracy\nTaylor',10.5)
text(a,6.55,5.2,'…',18)
box(a,7.15,4.15,2.6,1.4,'Recorded target\nBest accuracy on LEMUR\nNB201: see caption',10)
for x in (1,3,5):line(a,[(x,3.85),(x,3.45)])
line(a,[(1,3.45),(5,3.45)])
arrow(a,3,3.45,3,2.95)
box(a,.2,1.1,1.8,1.8,'Metadata\nTask · dataset\nMetric · budget',9.5)
box(a,2.5,1.75,2.6,1.15,'Already fitted\npredictor',11)
arrow(a,2.05,2.325,2.45,2.325)
box(a,5.65,1.75,2.25,1.15,'Predicted\naccuracy',11)
arrow(a,5.15,2.325,5.6,2.325)
box(a,8.4,1.75,1.35,1.15,'Evaluate\nprediction',10)
arrow(a,7.95,2.325,8.35,2.325)
line(a,[(8.95,4.1),(8.95,2.96)],dashed=True)
text(a,7.9,3.5,'Compare',9)
text(a,5,.53,'Held-out records: the recorded target is not an input to the predictor.',10)
save(f,'early_epoch_timeline')
print('Created three revised vector PDFs and PNG previews in',ROOT)

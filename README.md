# Automated Mathematical Equation Discovery for Visual Analysis

Submitted to JMLR (04/2021)

arXiv: https://arxiv.org/abs/2104.08633

## Abstract
Finding the best mathematical equation to deal with the different challenges found in complex scenarios requires a thorough understanding of the scenario and a trial and error process carried out by experts. In recent years, most state-of-the-art equation discovery methods have been widely applied in modeling and identification systems. However, equation discovery approaches can be very useful in computer vision, particularly in the field of feature extraction. In this paper, we focus on recent AI advances to present a novel framework for automatically discovering equations from scratch with little human intervention to deal with the different challenges encountered in real-world scenarios. In addition, our proposal can reduce human bias by proposing a search space design through generative network instead of hand-designed. As a proof of concept, the equations discovered by our framework are used to distinguish moving objects from the background in video sequences. Experimental results show the potential of the proposed approach and its effectiveness in discovering the best equation in video sequences.

## Citation
```
@misc{silva2021automated,
      title={Automated Mathematical Equation Structure Discovery for Visual Analysis}, 
      author={Caroline Pacheco do Espírito Silva and José A. M. Felippe De Souza and Antoine Vacavant and Thierry Bouwmans and Andrews Cordolino Sobral},
      year={2021},
      eprint={2104.08633},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Requirements
Make sure you have installed all of the following prerequisites on your development machine:

* Python 3.8.3

* Numpy

* Numba

* OpenCV

* BGSLibrary: pip install pybgs

* Chocolate: pip install git+https://github.com/AIworx-Labs/chocolate@master

## Usage
 
Please to run the  equation_mutation.py script use the command below :
```
python equation_mutation.py -m -f dataset/mutate.txt -i $dataset/skiting/input -g $dataset/skiting/groundtruth -r 0.15
```

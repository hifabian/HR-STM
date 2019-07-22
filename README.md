# HR-STM Simulation

This repository provides an implementation of high-resolution scanning tunnelling 
microscopy (HR-STM) simulations.
The method is based on the works of Bardeen (Bardeen`s Tunnelling Theory), Chen 
(Derivative Rule) and Hapala et al. (Probe Particle Model).

To run the simulation, execute `run.py` as
```sh
mpirun -np <NUMBER OF PROCESSES> python3 run.py
``` 
which will run it using MPI.
The input values are listed when running
```
python3 run.py -h
```
together with a small explanation.

## References
* J. Bardeen, "Tunnelling from a many-particle point of view", Phys. Rev. Lett. 6 (1961)
* C. J. Chen, "Tunneling matrix elements in three-dimensional space: The derivative rule 
  and the sum rule", Phys. Rev. B 42 (1990)
* G. Mándi, and K. Palotás, "Chen’s derivative rule revisited: Role of tip-orbital 
  interference in STM," Phys. Rev. B 91 (2015)
* P. Hapala, G. Kichin, C. Wagner, F. S. Tautz, R. Temirov, and P. Jelínek, "Mechanism of 
  high-resolution STM/AFM imaging with functionalized tips", Phys. Rev. B 90 (2014)
* P. Hapala, R. Temirov, F. S. Tautz, and P. Jelínek, "Origin of high-resolution IETS-STM
  images of organic molecules with functionalized tips", Phys. Rev. Lett. 113 (2014)
* O. Krejčı́, P. Hapala, M. Ondráček, and P. Jelı́nek, "Principles and simulations of 
  high-resolution STM imaging with a flexible tip apex", Phys. Rev. B 95 (2017)

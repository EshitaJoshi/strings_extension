/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
   
   This file was modified by Mierk Schwabe and Eshita Joshi from pair_coul_debye to 
   include, in addition to the debye interaction potential, the potential due to two positive wakes
   of charge w_c and distance r_w away from the particle:

	\phi_{w\pm} = \frac{-w_{c}q}{4\pi\epsilon_{0}|\mathbf{r} \mp \mathbf{r_{w}}|}\exp(-\kappa_{eff} |\mathbf{r} \mp \mathbf{r_{w}}|),

   where r is the interparticle distance, q is the particle charge and
   
	\kappa_{eff} = \sqrt{\frac{\kappa^{2}}{(1 + M_{th}^{2}\cos^{2}\zeta)} + \kappa^{2}_{e}}.

	 input parameter for this potential are M_th (thermal Mach number of the ions),
		lambda (plasma screening length - the same as used for Debye potential),
		the wake charge and distance,
		and the direction of the electric field - for now always in x direction!
		
------------------------------------------------------------------------- */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pair_coul_string.h"
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairCoulString::PairCoulString(LAMMPS *lmp) : PairCoulCut(lmp) {}

/* ---------------------------------------------------------------------- */

//use full neighbors list since force in non-newtonian
void PairCoulString::init_style()
{
	
	int irequest = neighbor->request(this,instance_me);
	neighbor->requests[irequest]->half = 0;
	neighbor->requests[irequest]->full = 1;
	
}

/* ---------------------------------------------------------------------- */

void PairCoulString::compute(int eflag, int vflag)
{
  int i,j,ii,jj,inum,jnum,itype,jtype;
  double qtmp,xtmp,ytmp,ztmp,delx,dely,delz,ecoul,fpair;
  double rsq,r2inv,r,rinv,forcecoul,forcedip1,forcedip2,factor_coul,factordip,screening;
  double theta,kappasqinv,kappainv,kappainvfac,costhetasq,forcefac;
  int *ilist,*jlist,*numneigh,**firstneigh;
  double costh, kappaeff, kappaeff2inv, Mtheff, kappasq, kappaesq;
  double xwake1,xwake2,rsqw1,rsqw2,delwx1,delwx2,rw1,rw2,r2invw1,r2invw2; //wake x positions
  
  ecoul = 0.0;
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = 0;

  double **x = atom->x;
  double **f = atom->f;
  double *q = atom->q;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_coul = force->special_coul;
  int newton_pair = force->newton_pair;
  double qqrd2e = force->qqrd2e;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  kappasq = kappa*kappa;
  kappaesq = kappae*kappae;
  kappasqinv = 1/kappasq;
  kappainv = sqrt(kappasqinv);
  
  //kappainvfac = 0.1*kappainv;
  
  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
	  
	//printf("atom nr. %d\n",ii) ; 
	  
    i = ilist[ii];
    qtmp = q[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

	xwake1 = xtmp - wake_delta;
	xwake2 = xtmp + wake_delta;	
	
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_coul = special_coul[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
     
      jtype = type[j];
	  
      if (rsq < cutsq[itype][jtype]) {
        r2inv = 1.0/rsq;
        r = sqrt(rsq);
        rinv = 1.0/r;
		
		//calculate distance to wakes
		delwx1 = xwake1 - x[j][0];
		rsqw1 = delwx1*delwx1 + dely*dely + delz*delz;
		rw1 = sqrt(rsqw1);
		r2invw1 = 1/rsqw1;
		
		delwx2 = xwake2 - x[j][0];
		rsqw2 = delwx2*delwx2 + dely*dely + delz*delz;
		rw2 = sqrt(rsqw2);	
		r2invw2 = 1/rsqw2;
		
		theta =  acos(delx*rinv); //angle to ion flow (fixed to x-direction for now)
		//printf("delx: %g, r2inv: %g\n",delx, r2inv);
		//printf("in loop: theta: %g, cos(theta): %g\n", theta, cos(theta));
		
		costh = cos(theta);
		costhetasq = costh*costh;
		
		//calculate effective screening length depending on angle to ion velocity and Mth
		Mtheff = Mth * costh; //effective ion velocity
		
		forcefac = qqrd2e * scale[itype][jtype] * qtmp*q[j];
		
		//effective screening length
		kappaeff = sqrt(kappasq/float(1+Mtheff*Mtheff) + kappaesq); //effective screening length
		kappaeff2inv = 1/(kappaeff*kappaeff);
		screening = exp(-kappaeff*r);
		forcecoul = factor_coul*forcefac * screening * (kappaeff + rinv) * r2inv;
		//printf("kappa: %g\n",kappaeff);
					
		/*  no effective screening length
		screening = exp(-kappa*r);
		forcecoul = forcefac * screening * (kappa + rinv); */
		
		factordip = (-1)*wake_z*forcefac*factor_coul; //charge opposite to that of particle
		
		/* set dipole force within particle radius to the value at the distance equal to 
		the particle radius in order to avoid infinite attractive force */
		
		if (rw1 < partd) { //set dipole force to value at cutoff for smaller radii
			forcedip1 = factordip*exp(-kappaeff*partd)*(kappaeff + 1.0/partd) / (partd*partd); 
		} else {
			if (rw2 < partd) { //set dipole force to value at cutoff for smaller radii
			forcedip2 = factordip*exp(-kappaeff*partd)*(kappaeff + 1.0/partd) / (partd*partd); 
			}
			else {
				forcedip1 =  factordip*exp(-kappaeff*rw1)*(kappaeff + 1.0/rw1) * r2invw1; 
				forcedip2 =  factordip*exp(-kappaeff*rw2)*(kappaeff + 1.0/rw2) * r2invw2; 
			}
		}
		
		fpair = forcecoul + forcedip1 + forcedip2;		
	
		/*debug 
		printf("xwake1: %g, delwx1: %g, rw1: %g\n", xwake1,delwx1,rw1);
		printf("xwake2: %g, delwx2: %g, rw2: %g\n", xwake2,delwx2,rw2);
		printf("screening length: %g, distance: %g\n", 1/kappaeff, r);
		printf("theta: %g, cos(theta): %g\n",theta,cos(theta));
		printf("fdip1: %g\n",delwx1*forcedip1);
		printf("fdip2: %g\n",delwx2*forcedip2);
		printf("fcoul y: %g\n\n",forcecoul*dely);
		printf("ftot: %g\n",(delx*forcecoul+delwx1*forcedip1+delwx2*forcedip2));
		printf("...\n");
		*/
		
        f[i][0] += (delx*forcecoul+delwx1*forcedip1+delwx2*forcedip2);
        f[i][1] += dely*fpair;
        f[i][2] += delz*fpair;
        /*if (newton_pair || j < nlocal) {
          f[j][0] -= (delx*forcecoul+delwx1*forcedip1+delwx2*forcedip2);
          f[j][1] -= dely*fpair;
          f[j][2] -= delz*fpair;
	  }*/ //remove because of full neighbor list required

        if (eflag) ecoul = factor_coul * qqrd2e *
            scale[itype][jtype] * qtmp*q[j] * rinv * screening;

        if (evflag) ev_tally(i,j,nlocal,newton_pair,
                             0.0,ecoul,fpair,delx,dely,delz);
      }
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairCoulString::settings(int narg, char **arg)
{
  if (narg != 7) error->all(FLERR,"Illegal pair_style command");  //increase to 4 if direction of Efield as input

  kappa = utils::numeric(FLERR,arg[0], false, lmp); //inverse ion screening length
  kappae =  utils::numeric(FLERR,arg[1], false, lmp); //inverse electron screening length
  cut_global = utils::numeric(FLERR,arg[2], false, lmp);
  wake_z = utils::numeric(FLERR,arg[3], false, lmp); //fraction of charge on particle that is on each wake; e.g. 0.6
  wake_delta = utils::numeric(FLERR,arg[4], false, lmp); //distance of wake from particle in m (should be less than lambda)
  Mth = utils::numeric(FLERR,arg[5], false, lmp); //thermal velocity of ions
  partd = utils::numeric(FLERR,arg[6], false, lmp); //particle radius!!

  // reset cutoffs that have been explicitly set

  //printf("!!!!!!!!!!!!!!!!!!!!Reading settings\n");	
  
  if (allocated) {
    int i,j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i; j <= atom->ntypes; j++)
        if (setflag[i][j]) cut[i][j] = cut_global;
  }
}

/* ----------------------------------------------------------------------
  proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairCoulString::write_restart_settings(FILE *fp)
{
  fwrite(&cut_global,sizeof(double),1,fp);
  fwrite(&wake_delta,sizeof(double),1,fp);
  fwrite(&wake_z,sizeof(double),1,fp);
  fwrite(&kappa,sizeof(double),1,fp);
  fwrite(&kappae,sizeof(double),1,fp);
  fwrite(&Mth,sizeof(double),1,fp);
  fwrite(&partd,sizeof(double),1,fp);
  fwrite(&offset_flag,sizeof(int),1,fp);
  fwrite(&mix_flag,sizeof(int),1,fp);
}

/* ----------------------------------------------------------------------
  proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairCoulString::read_restart_settings(FILE *fp)
{
  if (comm->me == 0) {
    fread(&cut_global,sizeof(double),1,fp);
	fread(&wake_delta,sizeof(double),1,fp);
	fread(&wake_z,sizeof(double),1,fp);
    fread(&kappa,sizeof(double),1,fp);
	fread(&kappae,sizeof(double),1,fp);
	fread(&Mth,sizeof(double),1,fp);
	fread(&partd,sizeof(double),1,fp);
    fread(&offset_flag,sizeof(int),1,fp);
    fread(&mix_flag,sizeof(int),1,fp);
  }
  MPI_Bcast(&cut_global,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&kappa,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&offset_flag,1,MPI_INT,0,world);
  MPI_Bcast(&mix_flag,1,MPI_INT,0,world);
}

/* ---------------------------------------------------------------------- */

double PairCoulString::single(int i, int j, int itype, int jtype,
                           double rsq, double factor_coul, double factor_lj, double theta,
                           double &fforce)
{
  double r2inv,r,rinv,forcecoul,phicoul,screening,forcedipole,qsq,kappasq;

  r2inv = 1.0/rsq;
  r = sqrt(rsq);
  rinv = 1.0/r;
  screening = exp(-kappa*r);
  qsq = atom->q[i]*atom->q[j];
  kappasq = kappa*kappa;
  forcecoul = force->qqrd2e * qsq * screening * (kappa + rinv);
  
  printf("/n !!!entered single pro!!!\n");
  return 0; //if this is actually used, include fdipole!
  //forcedipole = 3*0.43 * qsq * Mth^2 /(kappasq * rsq)*(3*(cos(theta))^2-1)	
  fforce = factor_coul*forcecoul * r2inv;

  phicoul = force->qqrd2e * atom->q[i]*atom->q[j] * rinv * screening;
  return factor_coul*phicoul;
}
